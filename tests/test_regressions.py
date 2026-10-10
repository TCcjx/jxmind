import copy
import json
import pickle
import zipfile
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from dataset.lm_dataset import SupDataset, make_collate, split_dataset
from main import fit_context
from model.model import MokioMindConfig, MokioMindForCausalLM
from scripts import download_assets
from trainer.trainer_utils import RunDirectory, TrainConfig, load_model_weights, run_training


def tiny_model(**kwargs):
    return MokioMindForCausalLM(
        MokioMindConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=32,
            max_position_embeddings=64,
            **kwargs,
        )
    )


@pytest.mark.parametrize("flash", [False, True])
def test_causal_mask_with_padding_never_reads_future(flash):
    torch.manual_seed(0)
    model = tiny_model(flash_attention=flash).eval()
    ids = torch.randint(0, 32, (2, 12))
    changed = ids.clone()
    changed[:, 6:] = (changed[:, 6:] + 1) % 32
    mask = torch.ones_like(ids)
    mask[1, -2:] = 0
    with torch.no_grad():
        before, after = model(ids, attention_mask=mask), model(changed, attention_mask=mask)
    torch.testing.assert_close(before.logits[:, :6], after.logits[:, :6])


@pytest.mark.parametrize("flash", [False, True])
def test_cached_logits_match_full_forward(flash):
    torch.manual_seed(3)
    model = tiny_model(flash_attention=flash).eval()
    ids = torch.randint(0, 32, (2, 10))
    with torch.no_grad():
        expected = model(ids).logits
        first = model(ids[:, :6], use_cache=True)
        second = model(
            ids[:, 6:],
            past_kv_list=first.past_key_values,
            attention_mask=torch.ones_like(ids),
            use_cache=True,
        )
    torch.testing.assert_close(second.logits, expected[:, 6:], atol=1e-6, rtol=1e-5)
    assert second.past_key_values[0][0].shape[2] == 10


def test_generation_cache_and_no_cache_match():
    model = tiny_model().eval()
    ids = torch.tensor([[1, 3, 7], [1, 4, 9]])
    cached = model.generate(ids, temperature=0, max_new_tokens=8)
    plain = model.generate(ids, temperature=0, max_new_tokens=8, use_cache=False)
    assert torch.equal(cached, plain)


def test_generation_stops_each_batch_row_at_eos(monkeypatch):
    model = tiny_model()
    calls = 0

    def forward(ids, **kwargs):
        nonlocal calls
        logits = torch.full((2, ids.shape[1], 32), -100.0)
        logits[0, -1, 2 if calls == 0 else 4] = 100
        logits[1, -1, 5 if calls == 0 else 2] = 100
        calls += 1
        return SimpleNamespace(logits=logits, past_key_values=[None])

    monkeypatch.setattr(model, "forward", forward)
    assert model.generate(
        torch.ones(2, 1, dtype=torch.long), temperature=0, eos_token_id=2, max_new_tokens=6
    ).tolist() == [[2, 2], [5, 2]]


def test_context_limits_raise_clear_errors():
    model = tiny_model()
    with pytest.raises(ValueError, match="max_position_embeddings"):
        model(torch.ones(1, 65, dtype=torch.long))
    with pytest.raises(ValueError, match="max_position_embeddings"):
        model.generate(torch.ones(1, 60, dtype=torch.long), max_new_tokens=5)


def test_fully_ignored_labels_have_finite_zero_loss():
    model = tiny_model()
    ids = torch.ones(2, 8, dtype=torch.long)
    loss = model(ids, labels=torch.full_like(ids, -100)).loss
    assert loss.item() == 0
    loss.backward()
    assert torch.isfinite(model.output.weight.grad).all()


def test_sft_drops_truncated_and_nonassistant_samples(tmp_path, tokenizer):
    path = tmp_path / "sft.jsonl"
    records = [
        {"q": "x" * 200, "a": "too long"},
        {"q": "hi", "a": "ok"},
        {"messages": [{"role": "user", "content": "no answer"}]},
    ]
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    dataset = SupDataset(path, tokenizer, sequence_length=64)
    assert len(dataset) == 1 and dataset.skipped == 2
    labels = dataset[0]["labels"]
    assert any(label != -100 for label in labels[1:])


def test_single_sample_and_zero_validation_keep_training_data():
    train, val = split_dataset([1], val_size=256)
    assert len(train) == 1 and len(val) == 0
    train, val = split_dataset([1, 2], val_size=0)
    assert len(train) == 2 and len(val) == 0


def test_context_removes_complete_turns_and_preserves_system(tokenizer):
    history = [
        {"role": "system", "content": "brief"},
        {"role": "user", "content": "x" * 100},
        {"role": "assistant", "content": "y" * 100},
        {"role": "user", "content": "hi"},
    ]
    kept = fit_context(tokenizer, history, 64)
    assert [m["role"] for m in kept] == ["system", "user"]
    with pytest.raises(ValueError, match="过长"):
        fit_context(tokenizer, [{"role": "user", "content": "x" * 200}], 64)


def test_incompatible_checkpoint_is_rejected(tmp_path):
    path = tmp_path / "bad.pth"
    torch.save({"other.weight": torch.ones(1)}, path)
    with pytest.raises(RuntimeError):
        load_model_weights(tiny_model(), str(path))


def test_custom_rope_scaling_roundtrips():
    scaling = {"type": "yarn", "factor": 4, "original_max_position_embeddings": 16}
    config = MokioMindConfig(rope_scaling=scaling)
    assert MokioMindConfig(**config.to_dict()).rope_scaling == scaling


def test_ffn_rounding_does_not_add_an_extra_block():
    assert MokioMindConfig(hidden_size=192).intermediate_size == 512


def test_collator_survives_windows_spawn_pickle():
    collate = pickle.loads(pickle.dumps(make_collate(0)))
    batch = collate([{"input_ids": [1, 2], "labels": [-100, 2]}])
    assert batch["input_ids"].tolist() == [[1, 2]]


def test_archive_extracts_only_csv_to_the_requested_path(tmp_path):
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../../sft_data_single.csv", "q,a\nhi,hello\n")
        z.writestr("../../unrelated.txt", "do not extract")
    target = tmp_path / "data" / "sft_data_single.csv"
    download_assets.extract_csv(archive, target)
    assert target.read_text() == "q,a\nhi,hello\n"
    assert list(target.parent.iterdir()) == [target]


def test_requested_data_download_failure_returns_nonzero(tmp_path, monkeypatch):
    monkeypatch.setattr(
        download_assets.sys,
        "argv",
        [
            "download",
            "--with-data",
            "--assets-dir",
            str(tmp_path / "assets"),
            "--data-dir",
            str(tmp_path / "data"),
        ],
    )
    monkeypatch.setattr(download_assets, "fetch", lambda url, dest: not url.endswith(".zip"))
    assert download_assets.main() == 1


def train_cfg(**kwargs):
    return TrainConfig(
        epochs=1,
        learning_rate=1e-3,
        warmup_steps=0,
        weight_decay=0,
        log_interval=1,
        eval_interval=0,
        save_interval=0,
        **kwargs,
    )


def test_accumulation_equals_full_batch_with_variable_token_counts(tmp_path):
    torch.manual_seed(8)
    initial = tiny_model()
    full, accumulated = copy.deepcopy(initial), copy.deepcopy(initial)
    samples = [
        {"input_ids": [1, 4, 6], "labels": [-100, 4, 6]},
        {"input_ids": [1, 2, 3, 5, 7], "labels": [-100, -100, 3, 5, 7]},
    ]
    tokenizer = SimpleNamespace(pad_token_id=0, save_pretrained=lambda path: None)
    for model, batch_size, accumulation, name in [(full, 2, 1, "full"), (accumulated, 1, 2, "acc")]:
        summary = run_training(
            model,
            tokenizer,
            samples,
            None,
            train_cfg(batch_size=batch_size, accumulation_steps=accumulation),
            RunDirectory(str(tmp_path), name),
        )
        assert summary["steps"] == 1
    for a, b in zip(full.parameters(), accumulated.parameters()):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-4)


def test_partial_batch_and_accumulation_are_not_dropped(tmp_path):
    samples = [{"input_ids": [1, 4, 6], "labels": [1, 4, 6]}] * 5
    tokenizer = SimpleNamespace(pad_token_id=0, save_pretrained=lambda path: None)
    summary = run_training(
        tiny_model(),
        tokenizer,
        samples,
        None,
        train_cfg(batch_size=2, accumulation_steps=2),
        RunDirectory(str(tmp_path), "partial"),
    )
    assert summary["steps"] == 2 and summary["tokens"] == 15


def test_resume_matches_uninterrupted_training(tmp_path, monkeypatch):
    torch.manual_seed(10)
    initial = tiny_model(dropout=0.1)
    full, interrupted, resumed = (copy.deepcopy(initial) for _ in range(3))
    samples = [{"input_ids": [1, i + 2, 7, 8], "labels": [1, i + 2, 7, 8]} for i in range(7)]
    tokenizer = SimpleNamespace(pad_token_id=0, save_pretrained=lambda path: None)
    cfg = replace(train_cfg(batch_size=2, accumulation_steps=1), save_interval=1)
    run_training(full, tokenizer, samples, None, cfg, RunDirectory(str(tmp_path), "full"))
    original = interrupted.forward
    calls = 0

    def fail_after_checkpoint(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("simulated interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(interrupted, "forward", fail_after_checkpoint)
    directory = RunDirectory(str(tmp_path), "resume")
    with pytest.raises(RuntimeError, match="simulated"):
        run_training(interrupted, tokenizer, samples, None, cfg, directory)
    summary = run_training(resumed, tokenizer, samples, None, replace(cfg, resume=True), directory)
    assert summary["steps"] == 4 and summary["tokens"] == 28
    for a, b in zip(full.parameters(), resumed.parameters()):
        torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-6)
    changed = copy.deepcopy(samples)
    changed[0]["input_ids"][1] = 22
    with pytest.raises(ValueError, match="dataset or tokenizer changed"):
        run_training(resumed, tokenizer, changed, None, replace(cfg, resume=True), directory)
