"""数据管线：分词、SFT 指令样本与预训练语料、批内 padding。

支持三种输入：
    1) JSONL/CSV，字段 q/a（MiniMind SFT 原始格式）
    2) JSONL，字段 prompt/question + chosen（指令格式）
    3) JSONL，字段 text（纯文本预训练语料）

约定：SFT 只在 assistant 回复 span 上计算 loss，prompt 部分 label 置 -100。
"""

import csv
import json
import os
import random
from typing import Dict, Iterator, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

IGNORE_INDEX = -100
DEFAULT_ASSETS = os.path.abspath("assets")


def load_tokenizer(path: str = DEFAULT_ASSETS):
    """从目录加载 fast tokenizer；缺 pad_token 时退化为 eos。"""
    if not os.path.exists(os.path.join(path, "tokenizer.json")):
        raise FileNotFoundError(
            f"{path}/tokenizer.json 不存在，请先执行：uv run python scripts/download_assets.py"
        )
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer 必须配置 eos_token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _read_records(path: str, limit: Optional[int] = None) -> Iterator[dict]:
    """逐条读取，避免额外复制完整 CSV / JSONL 到内存。"""
    if limit is not None and limit < 0:
        raise ValueError("limit must be nonnegative")
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = (
            csv.DictReader(f)
            if str(path).lower().endswith(".csv")
            else (json.loads(line) for line in f if line.strip())
        )
        for i, row in enumerate(rows):
            if limit and i >= limit:
                break
            if not isinstance(row, dict):
                raise ValueError(f"{path}: record {i + 1} must be a JSON object")
            yield row


def _to_messages(record: dict) -> Optional[List[dict]]:
    """把一条记录归一化成 [{'role','content'}, ...]；纯文本记录返回 None。"""
    if "messages" in record:
        messages = record["messages"]
        if not isinstance(messages, list) or not messages:
            return None
        if any(
            not isinstance(m, dict)
            or m.get("role") not in {"system", "user", "assistant"}
            or not isinstance(m.get("content"), str)
            for m in messages
        ):
            return None
        return messages if messages[-1]["role"] == "assistant" else None

    q = record.get("q") or record.get("question") or record.get("prompt") or record.get("instruction")
    a = record.get("a") or record.get("chosen") or record.get("output") or record.get("response")
    if not isinstance(q, str) or not isinstance(a, str) or not q.strip() or not a.strip():
        return None

    messages: List[dict] = []
    history = record.get("history")
    if isinstance(history, str):
        try:
            history = json.loads(history or "[]")
        except json.JSONDecodeError:
            history = []
    # 兼容部分 MiniMind CSV 中多包一层列表的历史。
    if isinstance(history, list) and len(history) == 1 and isinstance(history[0], list):
        if history[0] and isinstance(history[0][0], list):
            history = history[0]
    for turn in history if isinstance(history, list) else []:
        if isinstance(turn, (list, tuple)) and len(turn) == 2:
            if all(isinstance(content, str) for content in turn):
                messages += [{"role": "user", "content": turn[0]}, {"role": "assistant", "content": turn[1]}]
        elif (
            isinstance(turn, dict)
            and turn.get("role") in {"user", "assistant", "system"}
            and isinstance(turn.get("content"), str)
        ):
            messages.append(turn)
    if isinstance(record.get("system"), str) and record["system"].strip():
        messages.insert(0, {"role": "system", "content": record["system"]})
    if record.get("instruction") and isinstance(record.get("input"), str) and record["input"]:
        q += "\n" + record["input"]
    messages += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    return messages


class SupDataset(Dataset):
    """指令微调数据集：label 只在最后一条 assistant 回复上非 -100。"""

    def __init__(self, path: str, tokenizer, sequence_length: int = 512, limit: Optional[int] = None):
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.examples: List[Dict[str, List[int]]] = []
        skipped = 0
        for record in _read_records(path, limit):
            messages = _to_messages(record)
            if not messages:
                skipped += 1
                continue
            enc = self._encode(messages)
            if enc is None:
                skipped += 1
                continue
            self.examples.append(enc)
        self.skipped = skipped
        if not self.examples:
            raise ValueError(f"{path} 中没有任何可用的指令样本")

    def _encode(self, messages: List[dict]) -> Optional[Dict[str, List[int]]]:
        # 完整对话 vs 去掉最后一条 assistant 回复的前缀，两者长度差即监督区间。
        full = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt = self.tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        if not full.strip() or not messages[-1].get("content", "").strip():
            return None
        full_ids = self.tokenizer(full, add_special_tokens=False)["input_ids"]
        prompt_len = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if prompt_len >= len(full_ids):
            return None
        input_ids = full_ids[: self.sequence_length]
        labels = [IGNORE_INDEX] * min(prompt_len, len(input_ids)) + input_ids[prompt_len : len(input_ids)]
        if not any(label != IGNORE_INDEX for label in labels[1:]):
            return None
        return {"input_ids": input_ids, "labels": labels}

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class PretrainDataset(Dataset):
    """预训练数据集：整段语料拼接后切成定长块，label 即 input_ids 本身。

    语料字段优先级 text > content > q/a（MiniMind 的 SFT csv 去掉对话壳即可当预训练语料）。
    """

    def __init__(self, path: str, tokenizer, sequence_length: int = 512, limit: Optional[int] = None):
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        self.sequence_length = sequence_length
        self.blocks = []
        pending = []
        total_tokens = 0
        for record in _read_records(path, limit):
            text = record.get("text") or record.get("content")
            if not text:
                q, a = record.get("q"), record.get("a")
                text = f"{q}\n{a}" if q and a else None
            if isinstance(text, str) and text.strip():
                ids = tokenizer(text.strip(), add_special_tokens=False)["input_ids"] + [
                    tokenizer.eos_token_id
                ]
                total_tokens += len(ids)
                pending.extend(ids)
                full = len(pending) // sequence_length * sequence_length
                self.blocks.extend(pending[i : i + sequence_length] for i in range(0, full, sequence_length))
                pending = pending[full:]
        if not self.blocks:
            raise ValueError(f"{path}: 可用语料仅 {total_tokens} token，不足一个 {sequence_length} 长度的块")
        self.dropped_tokens = len(pending)

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        ids = self.blocks[idx]
        return {"input_ids": ids, "labels": list(ids)}


class PadCollator:
    """可 pickle 的 collator，支持 Windows DataLoader worker。"""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attention_mask = [], [], []
        for b in batch:
            pad = max_len - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [self.pad_token_id] * pad)
            labels.append(b["labels"] + [IGNORE_INDEX] * pad)
            attention_mask.append([1] * len(b["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def make_collate(pad_token_id: int):
    return PadCollator(pad_token_id)


def split_dataset(dataset: Dataset, val_size: int = 256, seed: int = 42):
    """按样本索引切 train/val，保证 val 不与 train 重叠。"""
    n = len(dataset)
    if val_size < 0 or n < 1:
        raise ValueError("dataset must be nonempty and val_size nonnegative")
    val_size = min(val_size, n - 1)
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    val_idx, train_idx = indices[:val_size], indices[val_size:]
    return torch.utils.data.Subset(dataset, train_idx), torch.utils.data.Subset(dataset, val_idx)
