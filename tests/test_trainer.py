"""训练/数据/推理链路的可运行性校验。

使用临时 byte tokenizer，不需要外部资源或网络。
"""

import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.lm_dataset import (
    IGNORE_INDEX,
    PretrainDataset,
    SupDataset,
    make_collate,
    split_dataset,
)
from model.model import MoEFeedForward, MokioMindConfig, MokioMindForCausalLM
from trainer.trainer_utils import (
    MetricsLogger,
    RunDirectory,
    TrainConfig,
    build_optimizer,
    cosine_lr,
    load_model_weights,
    read_model_config,
    run_training,
    save_checkpoint,
)

SAMPLES = [
    {"history": "[]", "q": "你好", "a": "你好，有什么可以帮你？"},
    {"history": "[]", "q": "天空是什么颜色", "a": "天空通常是蓝色的。"},
    {"history": "[]", "q": "水的化学式", "a": "水的化学式是 H2O。"},
    {"history": "[]", "q": "一年有几个月", "a": "一年有十二个月。"},
    {"history": "[]", "q": "介绍一下北京", "a": "北京是中国的首都，历史悠久。"},
    {"history": "[]", "q": "2+3 等于几", "a": "2+3 等于 5。"},
    {"history": '[[["你是谁？","我是一个助手。"]]]', "q": "你会什么", "a": "我可以回答问题。"},
    {"history": "[]", "q": "太阳从哪边升起", "a": "太阳从东边升起。"},
]


@pytest.fixture
def sft_file(tmp_path):
    path = tmp_path / "sft.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for record in SAMPLES:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return str(path)


@pytest.fixture
def small_model():
    torch.manual_seed(0)
    config = MokioMindConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        flash_attention=False,
    )
    return MokioMindForCausalLM(config)


# ---------- 数据集 ----------


def test_sft_labels_only_supervise_answer(tokenizer, sft_file):
    dataset = SupDataset(sft_file, tokenizer, sequence_length=512)
    assert len(dataset) == len(SAMPLES)
    for example in dataset.examples:
        ids, labels = example["input_ids"], example["labels"]
        supervised = [i for i, label in zip(ids, labels) if label != IGNORE_INDEX]
        masked = [label for label in labels if label != IGNORE_INDEX]
        # label 与 input 在监督区间逐 token 对齐。
        assert supervised == masked
        # 前缀（含 <|im_start|>user）全部被 mask 掉。
        first_supervised = labels.index(next(label for label in labels if label != IGNORE_INDEX))
        assert first_supervised >= 3
        prefix_text = tokenizer.decode(ids[:first_supervised])
        assert "user" in prefix_text


def test_sft_multiturn_history_is_masked(tokenizer, sft_file):
    dataset = SupDataset(sft_file, tokenizer, sequence_length=512)
    multi = dataset.examples[6]
    supervised_text = tokenizer.decode(
        [i for i, label in zip(multi["input_ids"], multi["labels"]) if label != IGNORE_INDEX]
    )
    assert "我可以回答问题" in supervised_text
    assert "我是一个助手" not in supervised_text


def test_collate_pads_and_masks(tokenizer, sft_file):
    dataset = SupDataset(sft_file, tokenizer, sequence_length=512)
    lengths = [len(e["input_ids"]) for e in dataset.examples]
    batch = make_collate(tokenizer.pad_token_id)([dataset[0], dataset[1], dataset[6]])
    assert batch["input_ids"].shape == (3, max(lengths[0], lengths[1], lengths[6]))
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    # 每行尾部 padding 位：attention_mask=0 且 label=-100。
    for row, sample in zip(batch["attention_mask"], [dataset[0], dataset[1], dataset[6]]):
        pad_start = len(sample["input_ids"])
        assert int(row[:pad_start].sum()) == pad_start
        assert int(row[pad_start:].sum()) == 0
    assert (batch["labels"][:, -1] == IGNORE_INDEX).any()


def test_pretrain_blocks_are_full_length(tokenizer, tmp_path):
    path = tmp_path / "pt.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for i in range(80):
            f.write(
                json.dumps(
                    {"text": "深度学习依靠反向传播自动学习特征表示。" * 4 + str(i)}, ensure_ascii=False
                )
                + "\n"
            )
    dataset = PretrainDataset(str(path), tokenizer, sequence_length=32)
    assert len(dataset) > 5
    assert all(len(block) == 32 for block in dataset.blocks)
    assert dataset[0]["labels"] == dataset[0]["input_ids"]


def test_pretrain_falls_back_to_qa_columns(tokenizer, sft_file):
    dataset = PretrainDataset(sft_file, tokenizer, sequence_length=16)
    assert len(dataset) >= 1


def test_split_is_disjoint(tokenizer, sft_file):
    dataset = SupDataset(sft_file, tokenizer, sequence_length=512)
    train, val = split_dataset(dataset, val_size=2, seed=7)
    assert not (set(train.indices) & set(val.indices))
    assert len(train) + len(val) == len(dataset)


# ---------- 学习率 / 优化器 ----------


def test_cosine_lr_warmup_then_decay():
    assert cosine_lr(0, 100, 1e-3, 10) == pytest.approx(1e-4)
    assert cosine_lr(9, 100, 1e-3, 10) == pytest.approx(1e-3)
    assert cosine_lr(10, 100, 1e-3, 10) == pytest.approx(1e-3)
    assert cosine_lr(55, 100, 1e-3, 10) == pytest.approx(5e-4, rel=1e-2)
    assert cosine_lr(100, 100, 1e-3, 10) < 1e-5
    # 超出总步数后不应反弹。
    assert cosine_lr(200, 100, 1e-3, 10) <= cosine_lr(100, 100, 1e-3, 10) + 1e-9


def test_weight_decay_skips_norm_and_bias(small_model):
    optimizer = build_optimizer(small_model, lr=1e-3, weight_decay=1.0)
    decay_ids = {id(p) for p in optimizer.param_groups[0]["params"]}
    no_decay_ids = {id(p) for p in optimizer.param_groups[1]["params"]}
    assert decay_ids.isdisjoint(no_decay_ids)
    assert len(decay_ids | no_decay_ids) == sum(1 for p in small_model.parameters() if p.requires_grad)
    for name, param in small_model.named_parameters():
        if "norm" in name or param.ndim <= 1:
            assert id(param) in no_decay_ids, name


# ---------- 前向 / 反向 ----------


def test_forward_backward_shapes(small_model):
    input_ids = torch.randint(0, 6400, (2, 24))
    labels = input_ids.clone()
    labels[:, :6] = IGNORE_INDEX
    out = small_model(input_ids=input_ids, labels=labels)
    assert out.logits.shape == (2, 24, 6400)
    assert out.loss.dim() == 0
    out.loss.backward()
    grads = [p.grad for p in small_model.parameters() if p.grad is not None]
    assert len(grads) == sum(1 for _ in small_model.parameters())
    assert all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_ignored_prefix_does_not_contribute_to_loss(small_model):
    small_model.eval()
    input_ids = torch.randint(0, 6400, (1, 20))
    full = small_model(input_ids=input_ids, labels=input_ids.clone()).loss.item()
    masked = small_model(
        input_ids=input_ids, labels=[IGNORE_INDEX] * 10 + input_ids[0, 10:].tolist()
    ).loss.item()
    assert masked != full


def test_padding_positions_are_ignored(small_model):
    small_model.eval()
    ids = torch.randint(10, 6400, (1, 16))
    labels = ids.clone()
    single = small_model(input_ids=ids, labels=labels).loss.item()
    padded_ids = torch.cat([ids, torch.zeros(1, 8, dtype=torch.long)], dim=1)
    padded_labels = torch.cat([labels, torch.full((1, 8), IGNORE_INDEX, dtype=torch.long)], dim=1)
    attention_mask = torch.cat(
        [torch.ones(1, 16, dtype=torch.long), torch.zeros(1, 8, dtype=torch.long)], dim=1
    )
    padded = small_model(
        input_ids=padded_ids, attention_mask=attention_mask, labels=padded_labels
    ).loss.item()
    assert padded == pytest.approx(single, abs=1e-4)


def test_single_batch_overfit_reduces_loss(small_model, tokenizer, sft_file):
    dataset = SupDataset(sft_file, tokenizer, sequence_length=512)
    batch = make_collate(tokenizer.pad_token_id)([dataset[i] for i in range(4)])
    optimizer = torch.optim.Adam(small_model.parameters(), lr=5e-3)
    first, last = None, None
    for i in range(80):
        out = small_model(input_ids=batch["input_ids"], labels=batch["labels"])
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if i == 0:
            first = out.loss.item()
        last = out.loss.item()
    assert last < first * 0.35, f"overfit 失败：{first:.4f} -> {last:.4f}"


# ---------- MoE ----------


def moe_config():
    return MokioMindConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        flash_attention=False,
        use_moe=True,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
    )


def test_moe_forward_returns_aux_loss():
    torch.manual_seed(0)
    model = MokioMindForCausalLM(moe_config())
    ids = torch.randint(0, 6400, (2, 16))
    out = model(input_ids=ids, labels=ids.clone())
    assert out.aux_loss is not None and out.aux_loss.dim() == 0
    assert torch.isfinite(out.loss) and out.loss.item() > 0
    out.loss.backward()
    routed = [n for n, _ in model.named_parameters() if "experts" in n]
    assert routed, "MoE 层未生成专家参数"


def test_moe_selects_topk_distinct_experts():
    torch.manual_seed(0)
    config = moe_config()
    moe = MoEFeedForward(config)
    x = torch.randn(2, 8, config.hidden_size)
    y, aux = moe(x)
    assert y.shape == x.shape
    assert torch.isfinite(aux) and aux.item() > 0
    scores = torch.softmax(moe.gate(x.view(-1, config.hidden_size)), dim=-1)
    topk_i = scores.topk(moe.topk, dim=-1).indices
    assert topk_i.shape == (2 * 8, config.num_experts_per_tok)
    assert (topk_i.diff(dim=-1) != 0).all(), "同一 token 选中了重复专家"


def test_moe_aux_loss_penalizes_collapsed_routing():
    """路由坍缩（负载与概率同时集中）时辅助损失应显著高于均衡路由。"""
    config = moe_config()
    moe = MoEFeedForward(config)
    n_tokens, n_experts = 16, config.n_routed_experts
    uniform = torch.full((n_tokens, n_experts), 1.0 / n_experts)
    concentrated = uniform.clone()
    concentrated[:, 0] = 0.94
    concentrated[:, 1:] = 0.06 / (n_experts - 1)
    balanced = torch.tensor([[(i * 2) % n_experts, (i * 2 + 1) % n_experts] for i in range(n_tokens)])
    collapsed = torch.zeros(n_tokens, 2, dtype=torch.long)
    b, s = 1, n_tokens
    aux_balanced = moe._aux_loss(uniform, balanced, b, s).item()
    aux_collapsed = moe._aux_loss(concentrated, collapsed, b, s).item()
    assert aux_collapsed > aux_balanced * 1.5
    # 完美均衡时 aux = alpha * n_experts * sum(1/n^2) = alpha
    assert aux_balanced == pytest.approx(config.aux_loss_alpha, rel=1e-3)


def test_moe_more_params_than_dense():
    dense = sum(
        p.numel()
        for p in MokioMindForCausalLM(
            MokioMindConfig(
                hidden_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=512,
                flash_attention=False,
            )
        ).parameters()
    )
    moe = sum(p.numel() for p in MokioMindForCausalLM(moe_config()).parameters())
    assert moe > dense


# ---------- 检查点 / 配置往返 ----------


def test_checkpoint_roundtrip(small_model, tmp_path):
    optimizer = build_optimizer(small_model, 1e-3)
    run_dir = RunDirectory(str(tmp_path), "unit")
    before = small_model.output.weight.detach().clone()
    path = save_checkpoint(
        small_model, optimizer, run_dir, 5, {"step": 5, "config": small_model.config.to_dict()}
    )
    small_model.output.weight.data.fill_(0.0)
    report = load_model_weights(small_model, path)
    assert report["missing_keys"] == [] and report["unexpected_keys"] == []
    assert torch.equal(small_model.output.weight.detach(), before)
    assert run_dir.latest == path

    restored = read_model_config(path)
    assert restored.hidden_size == small_model.config.hidden_size
    assert restored.intermediate_size == small_model.config.intermediate_size
    assert restored.use_moe == small_model.config.use_moe


def test_missing_checkpoint_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model_weights(torch.nn.Linear(2, 2), str(tmp_path / "nope"))


# ---------- 训练闭环 ----------


def test_run_training_end_to_end(tmp_path, tokenizer, sft_file):
    torch.manual_seed(0)
    dataset = SupDataset(sft_file, tokenizer, sequence_length=512)
    train, val = split_dataset(dataset, val_size=2, seed=1)
    config = MokioMindConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        flash_attention=False,
        vocab_size=len(tokenizer),
    )
    model = MokioMindForCausalLM(config)
    run_dir = RunDirectory(str(tmp_path), "e2e")
    summary = run_training(
        model,
        tokenizer,
        train,
        val,
        TrainConfig(
            epochs=40,
            batch_size=2,
            accumulation_steps=1,
            learning_rate=5e-3,
            warmup_steps=2,
            log_interval=1,
            eval_interval=999,
            save_interval=999,
            eval_batches=2,
            max_steps=40,
            device="cpu",
            sample_interval=20,
            sample_prompts=["你好"],
        ),
        run_dir,
    )
    assert summary["steps"] == 40
    losses = [
        json.loads(line)["loss"] for line in open(run_dir.loss_path, encoding="utf-8") if '"loss"' in line
    ]
    assert len(losses) == 40
    assert losses[-1] < losses[0], f"loss 未下降：{losses[0]} -> {losses[-1]}"
    assert summary["final_train_loss"] == pytest.approx(losses[-1], abs=1e-6)
    assert os.path.exists(os.path.join(summary["checkpoint"], "model.pth"))
    assert os.path.exists(summary["curve"])
    assert "<svg" in open(summary["curve"], encoding="utf-8").read()
    assert os.path.exists(run_dir.samples_path)
    # 采样文本必须真的落到文件里，且带步数标记。
    samples = open(run_dir.samples_path, encoding="utf-8").read()
    assert "[step 20]" in samples and "Q: 你好" in samples


def test_metrics_logger_curve_svg(tmp_path):
    logger = MetricsLogger(str(tmp_path / "loss.jsonl"))
    for step in range(1, 11):
        logger.log(step=step, loss=9.0 - 0.3 * step)
    out = logger.curve_svg(str(tmp_path / "curve.svg"))
    content = open(out, encoding="utf-8").read()
    assert content.startswith("<svg")
    assert content.count("L") >= 9
    assert "10 points" in content
    # 单点不足以成图。
    single = MetricsLogger(str(tmp_path / "one.jsonl"))
    single.log(step=1, loss=1.0)
    assert single.curve_svg(str(tmp_path / "one.svg")) is None
