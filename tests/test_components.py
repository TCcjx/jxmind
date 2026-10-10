"""针对 model/model.py 中已实现组件的可运行校验。

用法：
    uv run python tests/test_components.py
存在 pytest 时也可直接 `uv run pytest tests/`。
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import (  # noqa: E402
    MokioMindConfig,
    RMSNorm,
    apply_rotary_pos_emb,
    precompute_freqs,
)

HEAD_DIM = 64
YARN = {
    "beta_fast": 32,
    "beta_slow": 1,
    "factor": 16,
    "original_max_position_embeddings": 2048,
    "attention_factor": 1.0,
    "type": "yarn",
}

RESULTS = []


def check(name, fn):
    fn()
    RESULTS.append((name, "PASS"))
    print(f"PASS  {name}")


def angles_at_position_1(freqs_cos, freqs_sin):
    """t=1 时首半区即为角 θ_i；用 atan2 反演，避免 arccos 在小角度处丢失精度。"""
    half = freqs_cos.shape[-1] // 2
    return torch.atan2(freqs_sin[1][:half], freqs_cos[1][:half])


def _yarn_ratios():
    end = 2048 * 16
    pc, ps = precompute_freqs(dim=HEAD_DIM, end=end, rope_base=1e6)
    sc, ss = precompute_freqs(dim=HEAD_DIM, end=end, rope_base=1e6, rope_scaling=YARN)
    return angles_at_position_1(sc, ss) / angles_at_position_1(pc, ps)


# ---------------------------------------------------------------- RMSNorm
def rms_norm_is_scale_invariant():
    torch.manual_seed(0)
    norm = RMSNorm(HEAD_DIM, eps=1e-5)
    with torch.no_grad():
        norm.weight.fill_(1.0)
    x = torch.randn(4, 6, HEAD_DIM) * 50
    y = norm(x)
    rms = y.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4), rms


def rms_norm_learnable_weight_scales_output():
    torch.manual_seed(0)
    norm = RMSNorm(HEAD_DIM)
    x = torch.randn(2, 3, HEAD_DIM)
    with torch.no_grad():
        norm.weight.fill_(1.0)
        base = norm(x)
        norm.weight.fill_(2.0)
        doubled = norm(x)
    assert torch.allclose(doubled, base * 2, atol=1e-5)


def rms_norm_preserves_input_dtype():
    norm = RMSNorm(HEAD_DIM).half()
    out = norm(torch.randn(2, 4, HEAD_DIM, dtype=torch.float16))
    assert out.dtype == torch.float16, out.dtype


def rms_norm_is_differentiable():
    norm = RMSNorm(HEAD_DIM)
    x = torch.randn(2, 3, HEAD_DIM, requires_grad=True)
    norm(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert torch.isfinite(norm.weight.grad).all()


# ---------------------------------------------------------------- 标准 RoPE
def rope_cos_sin_are_unit_magnitude():
    cos, sin = precompute_freqs(dim=HEAD_DIM, end=64, rope_base=1e6)
    assert cos.shape == (64, HEAD_DIM) and sin.shape == cos.shape
    assert torch.allclose(cos.pow(2) + sin.pow(2), torch.ones_like(cos), atol=1e-5)


def rope_position_zero_is_identity():
    cos, sin = precompute_freqs(dim=HEAD_DIM, end=8, rope_base=1e6)
    q = torch.randn(1, 2, 1, HEAD_DIM)
    q0, _ = apply_rotary_pos_emb(q, q, cos[:1], sin[:1])
    assert torch.allclose(q0, q, atol=1e-6)


def rope_shift_invariance_of_scores():
    """整体平移所有位置，QK 分数矩阵不变 —— RoPE 只编码相对距离。"""
    seq, delta = 16, 5
    cos, sin = precompute_freqs(dim=HEAD_DIM, end=seq + delta, rope_base=1e6)
    torch.manual_seed(0)
    q = torch.randn(1, 1, seq, HEAD_DIM)
    k = torch.randn(1, 1, seq, HEAD_DIM)

    qa, ka = apply_rotary_pos_emb(q, k, cos[:seq], sin[:seq])
    qb, kb = apply_rotary_pos_emb(q, k, cos[delta : delta + seq], sin[delta : delta + seq])

    def dot_matrix(a, b):
        return torch.einsum("md,nd->mn", a[0, 0], b[0, 0])

    base = dot_matrix(qa, ka)
    shifted = dot_matrix(qb, kb)
    err = (base - shifted).abs().max().item()
    assert err < 1e-3, f"shift broke relative-position invariance, max abs err={err:.2e}"


def rope_position_ids_matches_slicing():
    seq = 8
    ids = torch.arange(seq).unsqueeze(0)
    cos, sin = precompute_freqs(dim=HEAD_DIM, end=seq, rope_base=1e6)
    q = torch.randn(1, 2, seq, HEAD_DIM)
    a = apply_rotary_pos_emb(q, q, cos[:seq], sin[:seq])
    b = apply_rotary_pos_emb(q, q, cos, sin, position_ids=ids)
    assert torch.allclose(a[0], b[0], atol=1e-6) and torch.allclose(a[1], b[1], atol=1e-6)


def rope_output_shape_matches_query():
    """两种调用约定都必须保持 [B, H, S, D]，禁止静默广播出多余维度。"""
    batch, heads, seq = 3, 2, 7
    table_cos, table_sin = precompute_freqs(dim=HEAD_DIM, end=seq, rope_base=1e6)
    q = torch.randn(batch, heads, seq, HEAD_DIM)
    ids = torch.arange(seq).unsqueeze(0).expand(batch, seq)

    for label, args, kwargs in (
        ("sliced 2-D", (q, q, table_cos[:seq], table_sin[:seq]), {}),
        ("batched 3-D", (q, q, table_cos[:seq].unsqueeze(0), table_sin[:seq].unsqueeze(0)), {}),
        ("position_ids", (q, q, table_cos, table_sin), {"position_ids": ids}),
    ):
        out_q, out_k = apply_rotary_pos_emb(*args, **kwargs)
        assert out_q.shape == q.shape, f"{label}: got {tuple(out_q.shape)}"
        assert out_k.shape == q.shape, f"{label}: got {tuple(out_k.shape)}"
        assert torch.isfinite(out_q).all(), label


def rope_half_split_layout():
    """rotate_half 约定：前半与后半共用同一组频率。"""
    cos, sin = precompute_freqs(dim=HEAD_DIM, end=4, rope_base=1e6)
    half = HEAD_DIM // 2
    assert torch.allclose(cos[:, :half], cos[:, half:], atol=1e-6)


# ---------------------------------------------------------------- YaRN
def yarn_leaves_high_freq_untouched():
    ratios = _yarn_ratios()
    assert torch.allclose(ratios[:6], torch.ones(6), atol=1e-4), ratios[:6]


def yarn_compresses_low_freq_by_factor():
    ratios = _yarn_ratios()
    expect = torch.full((32 - 14,), 1 / 16)
    assert torch.allclose(ratios[14:], expect, rtol=1e-3), ratios[14:]


def yarn_ramp_is_monotonic():
    ratios = _yarn_ratios()
    mid = ratios[5:15]
    assert bool((mid.diff() <= 1e-6).all()), mid
    assert bool((ratios > 1 / 16 - 1e-3).all()) and bool((ratios <= 1 + 1e-6).all()), ratios


def yarn_is_noop_within_original_window():
    plain, _ = precompute_freqs(dim=HEAD_DIM, end=2048, rope_base=1e6)
    scaled, _ = precompute_freqs(dim=HEAD_DIM, end=2048, rope_base=1e6, rope_scaling=YARN)
    assert torch.allclose(plain, scaled, atol=1e-6)


def yarn_attention_factor_compensates():
    """attention_factor 对 cos/sin 做均匀缩放，用于补偿长上下文下注意力分布的发散。"""
    cos1, sin1 = precompute_freqs(dim=HEAD_DIM, end=4096, rope_base=1e6, rope_scaling=YARN)
    boosted = dict(YARN, attention_factor=2.0)
    cos2, sin2 = precompute_freqs(dim=HEAD_DIM, end=4096, rope_base=1e6, rope_scaling=boosted)
    assert torch.allclose(cos2, cos1 * 2, atol=1e-6)
    assert torch.allclose(sin2, sin1 * 2, atol=1e-6)
    assert cos1.abs().max() <= 1.0 + 1e-6 and cos2.abs().max() <= 2.0 + 1e-6


# ---------------------------------------------------------------- Config
def config_roundtrips_through_json():
    cfg = MokioMindConfig(hidden_size=512, num_hidden_layers=4, use_moe=True)
    restored = MokioMindConfig(**cfg.to_diff_dict())
    assert restored.hidden_size == 512 and restored.num_hidden_layers == 4
    assert restored.use_moe is True and restored.model_type == "mokiomind"


def config_rope_scaling_tracks_flag():
    assert MokioMindConfig().rope_scaling is None
    yarn_cfg = MokioMindConfig(inference_rope_scaling=True).rope_scaling
    assert yarn_cfg["type"] == "yarn" and yarn_cfg["factor"] == 16
    assert yarn_cfg["beta_fast"] == 32 and yarn_cfg["beta_slow"] == 1
    assert yarn_cfg["original_max_position_embeddings"] == 2048


def config_default_geometry():
    cfg = MokioMindConfig()
    assert cfg.hidden_size // cfg.num_attention_heads == 64
    assert cfg.num_key_value_heads == 2 and cfg.num_attention_heads % cfg.num_key_value_heads == 0
    assert cfg.vocab_size == 6400 and cfg.max_position_embeddings == 32768


if __name__ == "__main__":
    torch.manual_seed(0)
    tests = [
        v for k, v in sorted(globals().items()) if k.startswith(("rms_norm_", "rope_", "yarn_", "config_"))
    ]
    failures = []
    for fn in tests:
        try:
            check(fn.__name__, fn)
        except Exception as exc:  # noqa: BLE001
            failures.append((fn.__name__, exc))
            print(f"FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
