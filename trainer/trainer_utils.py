"""训练公共设施：设备选择、参数量、学习率调度、检查点、loss 曲线。

约定：
    * 检查点目录为 output_dir/<run_name>/<step>，内含 model.pth + opt.pth + meta.json
    * 训练指标以 JSONL 追加写入 output_dir/<run_name>/loss.jsonl
    * loss 曲线直接渲染成 SVG（不引入 matplotlib 依赖）
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from dataset.lm_dataset import make_collate
from model.model import MokioMindConfig, MokioMindForCausalLM


def configure_stdio():
    """Keep Windows terminals from failing on generated Unicode text."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def human(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


def cosine_lr(step: int, total_steps: int, base_lr: float, warmup_steps: int, min_lr: float = 1e-6) -> float:
    """线性 warmup + cosine 衰减，total_steps<=0 时退化为恒定 lr。"""
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    if total_steps <= warmup_steps:
        return base_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float = 1.0):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # norm 参数与 bias 不参与 weight decay，小模型上更稳。
        if param.ndim <= 1 or "norm" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
        betas=(0.9, 0.95),
    )


class RunDirectory:
    """一次训练的输出目录：checkpoint / loss.jsonl / samples.txt 都落在这里。"""

    def __init__(self, output_dir: str, run_name: str):
        self.root = os.path.join(output_dir, run_name)
        os.makedirs(self.root, exist_ok=True)
        self.loss_path = os.path.join(self.root, "loss.jsonl")
        self.samples_path = os.path.join(self.root, "samples.txt")

    @property
    def latest(self) -> Optional[str]:
        if not os.path.isdir(self.root):
            return None
        steps = [
            int(d)
            for d in os.listdir(self.root)
            if d.isdigit()
            and all(os.path.exists(os.path.join(self.root, d, f)) for f in ("model.pth", "meta.json"))
        ]
        return os.path.join(self.root, str(max(steps))) if steps else None


def save_checkpoint(
    model: torch.nn.Module,
    optimizer,
    run_dir: RunDirectory,
    step: int,
    meta: Dict,
    tokenizer=None,
    rng_state=None,
) -> str:
    ckpt_dir = os.path.join(run_dir.root, str(step))
    os.makedirs(ckpt_dir, exist_ok=True)
    model.config.save_pretrained(ckpt_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(os.path.join(ckpt_dir, "tokenizer"))

    # 只存 tensor，避免把自定义类写进 pickle，加载时无需还原原始模块路径。
    def save_tensors(value, name):
        target = os.path.join(ckpt_dir, name)
        torch.save(value, target + ".part")
        os.replace(target + ".part", target)

    save_tensors({k: v.detach().cpu() for k, v in model.state_dict().items()}, "model.pth")
    save_tensors(optimizer.state_dict(), "opt.pth")
    if rng_state is not None:
        save_tensors(rng_state, "rng.pth")
    meta_path = os.path.join(ckpt_dir, "meta.json")
    with open(meta_path + ".part", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    os.replace(meta_path + ".part", meta_path)
    return ckpt_dir


def load_model_weights(model: torch.nn.Module, path: str) -> Dict:
    """从 checkpoint 目录或 .pth 文件读取权重，返回载入统计。"""
    weights_file = os.path.join(path, "model.pth") if os.path.isdir(path) else path
    if not os.path.exists(weights_file):
        raise FileNotFoundError(f"找不到检查点权重：{weights_file}\n请先训练并保存本项目的检查点。")
    state = torch.load(weights_file, map_location="cpu", weights_only=True)
    result = model.load_state_dict(state, strict=True)
    return {
        "loaded": len(state),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }


def load_optimizer_state(optimizer, path: str) -> bool:
    opt_file = os.path.join(path, "opt.pth") if os.path.isdir(path) else None
    if opt_file and os.path.exists(opt_file):
        optimizer.load_state_dict(torch.load(opt_file, map_location="cpu", weights_only=True))
        return True
    return False


class MetricsLogger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.records: List[Dict] = []
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                self.records = [json.loads(line) for line in f if line.strip()]

    def log(self, **kwargs) -> Dict:
        kwargs.setdefault("time", round(time.time(), 3))
        self.records.append(kwargs)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(kwargs, ensure_ascii=False) + "\n")
        return kwargs

    def curve_svg(
        self,
        out_path: str,
        key: str = "loss",
        title: str = "training loss",
        width: int = 900,
        height: int = 420,
    ) -> Optional[str]:
        """把 loss.jsonl 渲染成折线图；记录不足 2 条时返回 None。"""
        points = [(r.get("step", i), r[key]) for i, r in enumerate(self.records) if key in r]
        if len(points) < 2:
            return None
        pad_l, pad_r, pad_t, pad_b = 70, 24, 46, 46
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        # 留出上下边界，避免曲线贴边。
        span = (y_max - y_min) or 1.0
        y_lo, y_hi = y_min - span * 0.08, y_max + span * 0.08

        def sx(v):
            return pad_l + (v - x_min) / ((x_max - x_min) or 1) * (width - pad_l - pad_r)

        def sy(v):
            return height - pad_b - (v - y_lo) / (y_hi - y_lo) * (height - pad_t - pad_b)

        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
            f'<text x="{pad_l}" y="26" font-family="Segoe UI,Microsoft YaHei,sans-serif" font-size="17" fill="#111827">{title}</text>',
        ]
        for i in range(5):
            gv = y_lo + (y_hi - y_lo) * i / 4
            gy = sy(gv)
            parts.append(
                f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{width - pad_r}" y2="{gy:.1f}" stroke="#e5e7eb"/>'
            )
            parts.append(
                f'<text x="{pad_l - 10}" y="{gy + 4:.1f}" text-anchor="end" '
                f'font-family="Segoe UI,sans-serif" font-size="12" fill="#6b7280">{gv:.3f}</text>'
            )
        for i in range(5):
            gx = x_min + (x_max - x_min) * i / 4
            parts.append(
                f'<text x="{sx(gx):.1f}" y="{height - pad_b + 22}" text-anchor="middle" '
                f'font-family="Segoe UI,sans-serif" font-size="12" fill="#6b7280">{int(gx)}</text>'
            )
        path = " ".join(
            ("M" if i == 0 else "L") + f"{sx(x):.1f} {sy(y):.1f}" for i, (x, y) in enumerate(points)
        )
        parts.append(f'<path d="{path}" fill="none" stroke="#2563eb" stroke-width="2"/>')
        parts.append(
            f'<text x="{width - pad_r}" y="{height - 12}" text-anchor="end" '
            f'font-family="Segoe UI,sans-serif" font-size="12" fill="#6b7280">step</text>'
        )
        parts.append(
            f'<text x="{pad_l}" y="{height - 12}" font-family="Segoe UI,sans-serif" '
            f'font-size="12" fill="#6b7280">min {min(ys):.4f} / max {max(ys):.4f} / {len(points)} points</text>'
        )
        parts.append("</svg>")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(parts))
        return out_path


def append_samples(path: str, text: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def init_model(config: MokioMindConfig, checkpoint: Optional[str] = None, device: str = "cpu"):
    """新建模型，严格检查 checkpoint 的参数名称和形状。"""
    model = MokioMindForCausalLM(config)
    if checkpoint:
        report = load_model_weights(model, checkpoint)
        print(
            f"[weights] {checkpoint} -> loaded {report['loaded']} tensors, "
            f"missing {len(report['missing_keys'])}, unexpected {len(report['unexpected_keys'])}"
        )
        if report["missing_keys"] or report["unexpected_keys"]:
            print(f"  missing={report['missing_keys'][:5]} unexpected={report['unexpected_keys'][:5]}")
    return model.to(device)


def read_model_config(path: str) -> MokioMindConfig:
    """从 checkpoint 的 meta.json 还原训练时的模型配置。"""
    meta = os.path.join(path, "meta.json") if os.path.isdir(path) else path
    if os.path.isdir(path) and not os.path.exists(meta):
        meta = os.path.join(path, "config.json")
    if not os.path.exists(meta):
        raise FileNotFoundError(f"找不到 {meta}；推理需要带 meta.json 的 checkpoint 或 config.json")
    with open(meta, encoding="utf-8") as f:
        data = json.load(f)
    data = data.get("config", data)
    return MokioMindConfig(**{k: v for k, v in data.items() if k != "model_type"})


def format_speed(tokens_per_sec: float, steps_per_sec: float, seq_len: int) -> str:
    return f"{tokens_per_sec:,.0f} tok/s ({steps_per_sec:.2f} step/s, {seq_len} tok/step)"


@dataclass
class TrainConfig:
    epochs: int = 1
    batch_size: int = 8
    accumulation_steps: int = 2
    learning_rate: float = 5e-4
    warmup_steps: int = 50
    weight_decay: float = 1.0
    max_grad_norm: float = 1.0
    log_interval: int = 10
    eval_interval: int = 200
    save_interval: int = 500
    eval_batches: int = 20
    sample_interval: int = 0
    num_workers: int = 0
    max_steps: int = 0
    device: str = "cpu"
    resume: bool = False
    sample_prompts: Optional[List[str]] = None
    seed: int = 42

    def __post_init__(self):
        for name in ("epochs", "batch_size", "accumulation_steps", "log_interval", "eval_batches"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "warmup_steps",
            "eval_interval",
            "save_interval",
            "sample_interval",
            "num_workers",
            "max_steps",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.learning_rate <= 0 or self.max_grad_norm <= 0 or self.weight_decay < 0:
            raise ValueError("invalid learning rate, gradient norm or weight decay")


@torch.no_grad()
def evaluate(model, loader, device: str, max_batches: int) -> float:
    """在验证集上取前 max_batches 个 batch 的平均 loss。"""
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        tokens = int((labels[:, 1:] != -100).sum())
        total += float(loss) * tokens
        count += tokens
    model.train(was_training)
    return total / max(1, count)


def run_sample(
    model,
    tokenizer,
    run_dir: RunDirectory,
    step: int,
    prompts: List[str],
    device: str,
    max_new_tokens: int = 48,
) -> List[str]:
    """生成若干回复，既写文件也返回，便于训练中途肉眼检查。"""
    was_training = model.training
    model.eval()
    outputs = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").to(device)
        generated = model.generate(
            ids["input_ids"],
            max_new_tokens=min(
                max_new_tokens, model.config.max_position_embeddings - ids["input_ids"].shape[1]
            ),
            temperature=0.85,
            top_p=0.85,
            eos_token_id=tokenizer.eos_token_id,
        )
        reply = tokenizer.decode(generated[0], skip_special_tokens=True)
        outputs.append(f"[step {step}] Q: {prompt}\nA: {reply.strip()}")
    append_samples(run_dir.samples_path, "\n" + "\n\n".join(outputs) + "\n")
    model.train(was_training)
    return outputs


def run_training(model, tokenizer, train_ds, val_ds, cfg: TrainConfig, run_dir: RunDirectory) -> Dict:
    """Train with token-weighted accumulation and restartable epoch/batch cursors."""
    if not len(train_ds):
        raise ValueError("training dataset is empty")
    digest = hashlib.sha256()
    for dataset in (train_ds, val_ds):
        digest.update(str(0 if dataset is None else len(dataset)).encode())
        if dataset is not None:
            for idx in range(len(dataset)):
                digest.update(json.dumps(dataset[idx], sort_keys=True).encode())
    data_signature = digest.hexdigest()
    vocab_signature = (
        hashlib.sha256(json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()).hexdigest()
        if hasattr(tokenizer, "get_vocab")
        else None
    )
    model.to(cfg.device)
    model.train()
    optimizer = build_optimizer(model, cfg.learning_rate, cfg.weight_decay)
    collate = make_collate(tokenizer.pad_token_id)
    generator = torch.Generator()
    loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate,
        generator=generator,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=cfg.batch_size, collate_fn=collate, num_workers=cfg.num_workers)
        if val_ds is not None and len(val_ds)
        else None
    )
    steps_per_epoch = math.ceil(len(loader) / cfg.accumulation_steps)
    total_steps = cfg.epochs * steps_per_epoch
    if cfg.max_steps:
        total_steps = min(total_steps, cfg.max_steps)
    step, start_epoch, start_batch, total_tokens = 0, 0, 0, 0
    best_val, last_train_loss = None, None
    if cfg.resume:
        latest = run_dir.latest
        if not latest:
            raise FileNotFoundError("--resume requires an existing checkpoint in this run directory")
        with open(os.path.join(latest, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        previous = meta.get("train_config", {})
        if meta.get("data_signature") != data_signature or meta.get("vocab_signature") != vocab_signature:
            raise ValueError("dataset or tokenizer changed; use --checkpoint to start a new run")
        for key in ("batch_size", "accumulation_steps", "seed", "num_workers"):
            if key in previous and previous[key] != getattr(cfg, key):
                raise ValueError(f"cannot change {key} while resuming")
        if "next_epoch" not in meta:
            raise ValueError("legacy checkpoint lacks a training cursor; use --checkpoint to start a new run")
        load_model_weights(model, latest)
        if not load_optimizer_state(optimizer, latest):
            raise FileNotFoundError(f"missing optimizer state in {latest}")
        state = torch.load(os.path.join(latest, "rng.pth"), map_location="cpu", weights_only=True)
        torch.set_rng_state(state["torch"])
        if cfg.device.startswith("cuda") and "cuda" in state:
            torch.cuda.set_rng_state_all(state["cuda"])
        step = meta["step"]
        start_epoch, start_batch = meta["next_epoch"], meta["next_batch"]
        total_tokens = meta.get("tokens", 0)
        best_val, last_train_loss = meta.get("best_val_loss"), meta.get("train_loss")
        print(f"[resume] {latest}, step={step}, epoch={start_epoch}, batch={start_batch}")
    else:
        if run_dir.latest or os.path.exists(run_dir.loss_path):
            raise FileExistsError("run already exists; use --resume or choose a new --run-name")
        torch.manual_seed(cfg.seed)

    logger = MetricsLogger(run_dir.loss_path)
    # Discard metrics written after the last complete checkpoint (e.g. after an interruption).
    if cfg.resume:
        logger.records = [r for r in logger.records if r.get("step", 0) <= step]
        with open(logger.path, "w", encoding="utf-8") as f:
            for record in logger.records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    run_start = time.time()
    lr = optimizer.param_groups[0]["lr"]
    next_epoch, next_batch = start_epoch, start_batch
    optimizer.zero_grad(set_to_none=True)

    def checkpoint():
        meta = {
            "step": step,
            "next_epoch": next_epoch,
            "next_batch": next_batch,
            "lr": lr,
            "tokens": total_tokens,
            "train_loss": last_train_loss,
            "best_val_loss": best_val,
            "config": model.config.to_dict(),
            "train_config": asdict(cfg),
            "data_signature": data_signature,
            "vocab_signature": vocab_signature,
        }
        state = {"torch": torch.get_rng_state()}
        if cfg.device.startswith("cuda"):
            state["cuda"] = torch.cuda.get_rng_state_all()
        return save_checkpoint(model, optimizer, run_dir, step, meta, tokenizer, state)

    for epoch in range(start_epoch, cfg.epochs):
        if step >= total_steps:
            break
        generator.manual_seed(cfg.seed + epoch)
        accumulated_tokens, accumulated_loss, micro_batches = 0, 0.0, 0
        for batch_idx, batch in enumerate(loader):
            if epoch == start_epoch and batch_idx < start_batch:
                continue
            batch = {key: value.to(cfg.device) for key, value in batch.items()}
            target_tokens = int((batch["labels"][:, 1:] != -100).sum())
            if target_tokens == 0:
                raise ValueError("batch has no next-token supervision")
            out = model(**batch)
            if not torch.isfinite(out.loss):
                raise FloatingPointError("non-finite training loss")
            (out.loss * target_tokens).backward()
            accumulated_tokens += target_tokens
            accumulated_loss += out.loss.detach().item() * target_tokens
            total_tokens += int(batch["attention_mask"].sum())
            micro_batches += 1
            next_epoch, next_batch = epoch, batch_idx + 1
            if next_batch == len(loader):
                next_epoch, next_batch = epoch + 1, 0
            if micro_batches < cfg.accumulation_steps and batch_idx + 1 < len(loader):
                continue
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.div_(accumulated_tokens)
            lr = cosine_lr(step, total_steps, cfg.learning_rate, cfg.warmup_steps)
            set_lr(optimizer, lr)
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm, error_if_nonfinite=True)
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            last_train_loss = round(accumulated_loss / accumulated_tokens, 5)
            accumulated_tokens, accumulated_loss, micro_batches = 0, 0.0, 0
            if step % cfg.log_interval == 0 or step == 1 or step == total_steps:
                logger.log(
                    step=step,
                    lr=round(lr, 8),
                    loss=last_train_loss,
                    grad_norm=round(grad_norm, 4),
                    tokens=total_tokens,
                )
                print(f"step {step}/{total_steps} | loss {last_train_loss:.4f} | lr {lr:.2e}")
            if val_loader is not None and cfg.eval_interval and step % cfg.eval_interval == 0:
                val_loss = evaluate(model, val_loader, cfg.device, cfg.eval_batches)
                best_val = val_loss if best_val is None else min(best_val, val_loss)
                logger.log(step=step, val_loss=round(val_loss, 5))
                print(f"  -> val loss {val_loss:.4f}")
            if cfg.sample_interval and cfg.sample_prompts and step % cfg.sample_interval == 0:
                for line in run_sample(model, tokenizer, run_dir, step, cfg.sample_prompts, cfg.device):
                    print("  " + line.replace("\n", "\n  "))
            if cfg.save_interval and step % cfg.save_interval == 0:
                print(f"  -> checkpoint saved: {checkpoint()}")
            if step >= total_steps:
                break
    val_loss = evaluate(model, val_loader, cfg.device, cfg.eval_batches) if val_loader else None
    if val_loss is not None:
        best_val = val_loss if best_val is None else min(best_val, val_loss)
        logger.log(step=step, val_loss=round(val_loss, 5))
    ckpt = checkpoint()
    return {
        "params": count_params(model),
        "train_samples": len(train_ds),
        "val_samples": 0 if val_ds is None else len(val_ds),
        "steps": step,
        "total_steps": total_steps,
        "tokens": total_tokens,
        "final_train_loss": last_train_loss,
        "val_loss": None if val_loss is None else round(val_loss, 5),
        "best_val_loss": None if best_val is None else round(best_val, 5),
        "checkpoint": ckpt,
        "curve": logger.curve_svg(os.path.join(run_dir.root, "loss_curve.svg")),
        "elapsed_sec": round(time.time() - run_start, 1),
    }


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """预训练与 SFT 共用的命令行参数。"""
    group = parser.add_argument_group("model")
    group.add_argument("--hidden-size", type=int, default=512)
    group.add_argument("--intermediate-size", type=int, default=None)
    group.add_argument("--num-hidden-layers", type=int, default=8)
    group.add_argument("--num-attention-heads", type=int, default=8)
    group.add_argument("--num-key-value-heads", type=int, default=2)
    group.add_argument("--max-position-embeddings", type=int, default=32768)
    group.add_argument("--rope-theta", type=int, default=1000000)
    group.add_argument("--vocab-size", type=int, default=None)
    group.add_argument("--dropout", type=float, default=0.0)
    group.add_argument("--no-flash-attention", dest="flash_attention", action="store_false")
    group.add_argument("--use-moe", action="store_true")
    group.add_argument("--n-routed-experts", type=int, default=4)
    group.add_argument("--n-shared-experts", type=int, default=1)
    group.add_argument("--num-experts-per-tok", type=int, default=2)
    group.set_defaults(flash_attention=True)

    group = parser.add_argument_group("data")
    group.add_argument("--data-path", type=str, required=True)
    group.add_argument("--tokenizer-path", type=str, default=None)
    group.add_argument("--seq-length", type=int, default=512)
    group.add_argument("--max-samples", type=int, default=0, help="只取前 N 条，0 表示全量")
    group.add_argument("--val-size", type=int, default=256)

    group = parser.add_argument_group("train")
    group.add_argument("--run-name", type=str, default=None)
    group.add_argument("--output-dir", type=str, default="output")
    group.add_argument("--epochs", type=int, default=1)
    group.add_argument("--batch-size", type=int, default=8)
    group.add_argument("--accumulation-steps", type=int, default=2)
    group.add_argument("--learning-rate", type=float, default=5e-4)
    group.add_argument("--warmup-steps", type=int, default=50)
    group.add_argument("--weight-decay", type=float, default=1.0)
    group.add_argument("--max-grad-norm", type=float, default=1.0)
    group.add_argument("--log-interval", type=int, default=10)
    group.add_argument("--eval-interval", type=int, default=200)
    group.add_argument("--save-interval", type=int, default=500)
    group.add_argument("--eval-batches", type=int, default=20)
    group.add_argument("--sample-interval", type=int, default=0)
    group.add_argument("--sample-prompt", action="append", default=[])
    group.add_argument("--num-workers", type=int, default=0)
    group.add_argument("--max-steps", type=int, default=0)
    group.add_argument("--device", type=str, default=None)
    group.add_argument("--seed", type=int, default=42)
    group.add_argument("--num-threads", type=int, default=4, help="CPU intra-op threads")
    group.add_argument("--resume", action="store_true")
    group.add_argument("--checkpoint", type=str, default=None, help="仅载入权重的 checkpoint 目录或 .pth")
    return parser


def config_from_args(args, tokenizer=None) -> MokioMindConfig:
    torch.manual_seed(args.seed)
    if args.num_threads < 1:
        raise ValueError("--num-threads must be positive")
    torch.set_num_threads(args.num_threads)
    checkpoint = args.checkpoint
    if args.resume:
        if checkpoint or not args.run_name:
            raise ValueError("--resume requires --run-name and cannot be combined with --checkpoint")
        checkpoint = RunDirectory(args.output_dir, args.run_name).latest
        if not checkpoint:
            raise FileNotFoundError("no checkpoint to resume")
    config = (
        read_model_config(checkpoint)
        if checkpoint and os.path.isdir(checkpoint)
        else MokioMindConfig(
            dropout=args.dropout,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            max_position_embeddings=args.max_position_embeddings,
            num_attention_heads=args.num_attention_heads,
            num_hidden_layers=args.num_hidden_layers,
            num_key_value_heads=args.num_key_value_heads,
            vocab_size=args.vocab_size or (len(tokenizer) if tokenizer is not None else 6400),
            rope_theta=args.rope_theta,
            flash_attention=args.flash_attention,
            use_moe=args.use_moe,
            num_experts_per_tok=args.num_experts_per_tok,
            n_routed_experts=args.n_routed_experts,
            n_shared_experts=args.n_shared_experts,
        )
    )
    if tokenizer is not None:
        if config.vocab_size != len(tokenizer):
            raise ValueError("model vocabulary size does not match tokenizer")
        config.bos_token_id, config.eos_token_id = tokenizer.bos_token_id, tokenizer.eos_token_id
    if not 2 <= args.seq_length <= config.max_position_embeddings:
        raise ValueError("--seq-length must be between 2 and max_position_embeddings")
    return config


def tokenizer_from_args(args):
    from dataset.lm_dataset import load_tokenizer

    path = args.tokenizer_path
    checkpoint = args.checkpoint
    if args.resume and args.run_name:
        checkpoint = RunDirectory(args.output_dir, args.run_name).latest
    if path is None and checkpoint and os.path.isdir(os.path.join(checkpoint, "tokenizer")):
        path = os.path.join(checkpoint, "tokenizer")
    return load_tokenizer(path) if path else load_tokenizer()


def train_config_from_args(args) -> TrainConfig:
    return TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        eval_batches=args.eval_batches,
        sample_interval=args.sample_interval,
        num_workers=args.num_workers,
        max_steps=args.max_steps,
        device=args.device or pick_device(),
        resume=args.resume,
        sample_prompts=list(args.sample_prompt),
        seed=args.seed,
    )
