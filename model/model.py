import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


class MokioMindConfig(PretrainedConfig):  # !模型配置初始化
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: Optional[int] = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        rope_scaling: Optional[dict] = None,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        # SwiGLU 的 8/3·d 经验值，向上取整到 64 的倍数：hidden_size 512 -> 1408
        if self.intermediate_size is None:
            self.intermediate_size = math.ceil(8 * self.hidden_size / (3 * 64)) * 64

        self.rope_scaling = rope_scaling or (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )

        for name in (
            "hidden_size",
            "intermediate_size",
            "max_position_embeddings",
            "num_attention_heads",
            "num_key_value_heads",
            "num_hidden_layers",
            "vocab_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if hidden_size % num_attention_heads or num_attention_heads % num_key_value_heads:
            raise ValueError("hidden_size / attention heads and Q / KV heads must be divisible")
        if (hidden_size // num_attention_heads) % 2:
            raise ValueError("RoPE requires an even head dimension")
        if not 0 <= dropout < 1 or rms_norm_eps <= 0 or rope_theta <= 1:
            raise ValueError("invalid dropout, rms_norm_eps or rope_theta")
        if hidden_act not in {"silu", "gelu", "relu"}:
            raise ValueError(f"unsupported hidden_act: {hidden_act}")
        if use_moe and not (1 <= num_experts_per_tok <= n_routed_experts and n_shared_experts >= 0):
            raise ValueError("invalid MoE expert counts")
        if scoring_func not in {"softmax", "sigmoid"} or aux_loss_alpha < 0:
            raise ValueError("invalid MoE scoring function or auxiliary loss weight")
        if self.rope_scaling is not None:
            for name in (
                "factor",
                "original_max_position_embeddings",
                "beta_fast",
                "beta_slow",
                "attention_factor",
            ):
                if self.rope_scaling.get(name, 1) <= 0:
                    raise ValueError(f"rope_scaling.{name} must be positive")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (self.weight * x.float() * self._norm(x.float())).type_as(x)


def precompute_freqs(
    dim: int,
    end: int = int(32 * 1024),
    rope_base: float = 1e6,
    rope_scaling: Optional[dict] = None,
):
    # 1. 初始化标准 RoPE 频率。
    # torch.arange(0, dim, 2) 生成 [0, 2, 4, ... dim-2]
    # 计算出的 freqs 就是标准的 1 / (base ** (2i / d))
    freqs, attn_factor = (
        1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)),
        1.0,
    )

    if rope_scaling is not None:
        # 2. 从配置字典中提取 YaRN 的超参数
        # orig_max: 模型预训练时的原始最大长度（例如 Llama-2 是 2048 或 4096）
        # factor: 要扩展的倍数 s (比如从 2k 扩展到 32k，factor 就是 16)
        # beta_fast (对应论文中的 α): 高频边界，波长比例大于此值的维度不缩放
        # beta_slow (对应论文中的 β): 低频边界，波长比例小于此值的维度全量缩放
        # attn_factor: 注意力温度补偿，由于距离拉长导致注意力分布发散（变平缓），需要乘上一个系数让注意力重新“聚焦”
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0),
            rope_scaling.get("beta_slow", 1.0),
            rope_scaling.get("attention_factor", 1.0),
        )

        # 只有当要推断的长度大于原始训练长度时，才应用缩放
        if end / orig_max > 1.0:
            # 3. 使用前文推导的公式，定义波长比例 b 到维度索引 i 的映射函数
            def inv_dim(b):
                return (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))

            # 4. 计算高频区和低频区的维度切分点
            # low: 不需要缩放的高频部分的最高索引
            # high: 需要完全缩放的低频部分的最低索引
            low, high = (
                max(math.floor(inv_dim(beta_fast)), 0),
                min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1),
            )

            # 5. 计算混合因子 γ (Ramp)
            # 在 low 之前，ramp 为 0；在 high 之后，ramp 为 1；在 low 和 high 之间，线性过渡。
            # clamp 函数限制了数值只能在 [0, 1] 之间。
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001),
                0,
                1,
            )

            # 6. 频率融合公式：f'(i) = f(i) * ((1-γ) + γ/s)
            # 当 ramp=0 时（高频）：系数为 1，保持原频率不变。
            # 当 ramp=1 时（低频）：系数为 1/factor，即对频率进行线性插值缩放。
            # ramp在0-1之间时：平滑过渡。
            freqs = freqs * (1 - ramp + ramp / factor)

    # 7. 根据目标长度 end，生成位置索引向量 t
    t = torch.arange(end, device=freqs.device)

    # 8. 计算外积：将位置 t 与处理好的频率 freqs 相乘，得到每个位置的旋转角度 θ
    freqs = torch.outer(t, freqs).float()

    # 9. 计算 Cos 和 Sin，并应用注意力补偿系数 (attn_factor)
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor

    return freqs_cos, freqs_sin


# !旋转位置编码
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1)

    # cos/sin 可以是完整位置表；传入 position_ids 时先按需索引。
    if position_ids is not None:
        cos, sin = cos[position_ids], sin[position_ids]
    # 允许直接传 [seq_len, head_dim] 的切片：补出 batch 维，避免与 head 维静默广播成高维张量。
    elif cos.dim() == 2:
        cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed.type_as(q), k_embed.type_as(k)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA：把每个 KV 头复制 n_rep 份以对齐 Q 头数。[B, n_kv, S, D] -> [B, n_kv*n_rep, S, D]"""
    if n_rep == 1:
        return x
    b, n_kv, s, d = x.shape
    return x[:, :, None].expand(b, n_kv, n_rep, s, d).reshape(b, n_kv * n_rep, s, d)


def build_attention_mask(q_len: int, k_len: int, dtype, device) -> torch.Tensor:
    """加性因果掩码；k_len > q_len 时（KV cache 已存历史）query 仍可看见全部过去。"""
    min_value = torch.finfo(dtype).min
    mask = torch.full((q_len, k_len), min_value, dtype=dtype, device=device)
    past = k_len - q_len
    rows = torch.arange(q_len, device=device)[:, None] + past
    cols = torch.arange(k_len, device=device)[None, :]
    return mask.masked_fill(cols <= rows, 0.0)


class Attention(nn.Module):
    def __init__(self, config: MokioMindConfig, layer_id: int):
        super().__init__()
        if config.num_attention_heads % config.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        self.layer_id = layer_id
        self.n_heads = config.num_attention_heads
        self.n_kv = config.num_key_value_heads
        self.head_dim = config.hidden_size // self.n_heads
        self.n_rep = self.n_heads // self.n_kv
        self.wq = nn.Linear(config.hidden_size, self.head_dim * self.n_heads, bias=False)
        self.wk = nn.Linear(config.hidden_size, self.head_dim * self.n_kv, bias=False)
        self.wv = nn.Linear(config.hidden_size, self.head_dim * self.n_kv, bias=False)
        self.wo = nn.Linear(self.head_dim * self.n_heads, config.hidden_size, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.attn_dropout = config.dropout
        self.flash = config.flash_attention

    def forward(
        self,
        x,
        cos,
        sin,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        b, s, _ = x.shape
        q = self.wq(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, s, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, s, self.n_kv, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        # 缓存累积后的完整 k/v，past_len 才能由 shape[2] 直接读出。
        cache = (k, v)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # SDPA 的 is_causal 只在自注意力（无 cache、无 padding）时成立，其余情况走显式掩码。
        if self.flash and attention_mask is None and past_kv is None and s > 1:
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.attn_dropout if self.training else 0.0, is_causal=True
            )
        else:
            mask = build_attention_mask(s, k.shape[2], q.dtype, q.device)[None, None]
            if attention_mask is not None:
                mask = (mask + attention_mask).clamp_min(torch.finfo(q.dtype).min)
            if self.flash:
                out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask, dropout_p=self.attn_dropout if self.training else 0.0
                )
            else:
                scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim) + mask
                attn = F.softmax(scores.float(), dim=-1).type_as(scores)
                if self.training and self.attn_dropout:
                    attn = F.dropout(attn, p=self.attn_dropout)
                out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(b, s, -1)
        return self.resid_dropout(self.wo(out)), cache


class FeedForward(nn.Module):
    """SwiGLU：w1 为 gate，w3 为 up，w2 为 down，与上游权重命名一致。"""

    def __init__(self, config: MokioMindConfig):
        super().__init__()
        acts = {"silu": F.silu, "gelu": F.gelu, "relu": F.relu}
        if config.hidden_act not in acts:
            raise ValueError(f"unsupported hidden_act: {config.hidden_act}")
        self.act = acts[config.hidden_act]
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w2(self.act(self.w1(x)) * self.w3(x))), None


class MoEFeedForward(nn.Module):
    def __init__(self, config: MokioMindConfig):
        super().__init__()
        self.n_experts = config.n_routed_experts
        self.topk = min(config.num_experts_per_tok, config.n_routed_experts)
        self.alpha = config.aux_loss_alpha
        self.scoring_func = config.scoring_func
        self.norm_topk_prob = config.norm_topk_prob
        self.seq_aux = config.seq_aux
        self.gate = nn.Linear(config.hidden_size, self.n_experts, bias=False)
        self.experts = nn.ModuleList([FeedForward(config) for _ in range(config.n_routed_experts)])
        self.shared_experts = nn.ModuleList([FeedForward(config) for _ in range(config.n_shared_experts)])

    def _aux_loss(self, scores, topk_i, b, s):
        k = self.topk
        onehot = F.one_hot(topk_i, self.n_experts).sum(dim=1).float()
        if self.seq_aux:
            load = onehot.view(b, s, self.n_experts).sum(1) / (s * k)
            probs = scores.view(b, s, self.n_experts).mean(1)
            return self.alpha * self.n_experts * (load.detach() * probs).sum(-1).mean()
        else:
            load = onehot.sum(0) / (b * s * k)
            probs = scores.mean(0)
        return self.alpha * self.n_experts * (load.detach() * probs).sum()

    def forward(self, x):
        b, s, h = x.shape
        flat = x.view(-1, h)
        logits = self.gate(flat)
        scores = F.softmax(logits, dim=-1) if self.scoring_func == "softmax" else F.sigmoid(logits)
        topk_w, topk_i = scores.topk(self.topk, dim=-1)
        if self.norm_topk_prob and self.topk > 1:
            topk_w = topk_w / (topk_w.sum(-1, keepdim=True) + 1e-9)

        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            hit = topk_i == e
            if not hit.any():
                continue
            tok, slot = hit.nonzero(as_tuple=True)
            weight = topk_w[tok, slot].unsqueeze(-1)
            out.index_add_(0, tok, weight * expert(flat[tok])[0])

        for shared in self.shared_experts:
            out = out + shared(flat)[0]

        return out.view(b, s, h), self._aux_loss(scores, topk_i, b, s)


class TransformerBlock(nn.Module):
    def __init__(self, config: MokioMindConfig, layer_id: int):
        super().__init__()
        eps = config.rms_norm_eps
        self.attention_norm = RMSNorm(config.hidden_size, eps)
        self.attention = Attention(config, layer_id)
        self.ffn_norm = RMSNorm(config.hidden_size, eps)
        self.feed_forward = MoEFeedForward(config) if config.use_moe else FeedForward(config)

    def forward(self, x, cos, sin, attention_mask=None, position_ids=None, past_kv=None):
        attn, cache = self.attention(self.attention_norm(x), cos, sin, attention_mask, position_ids, past_kv)
        x = x + attn
        ffn_out, aux = self.feed_forward(self.ffn_norm(x))
        return x + ffn_out, cache, aux


class MokioMindModel(nn.Module):
    def __init__(self, config: MokioMindConfig):
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        head_dim = config.hidden_size // config.num_attention_heads
        freqs_cos, freqs_sin = precompute_freqs(
            head_dim, config.max_position_embeddings, config.rope_theta, config.rope_scaling
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_kv_list: Optional[List] = None,
        use_cache: bool = False,
    ):
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have shape [batch, nonempty sequence]")
        b, s = input_ids.shape
        if past_kv_list is not None and len(past_kv_list) != len(self.layers):
            raise ValueError("KV cache must contain one entry per layer")
        past_len = 0 if not past_kv_list else int(past_kv_list[0][0].shape[2])
        if past_len + s > self.config.max_position_embeddings:
            raise ValueError("sequence plus KV cache exceeds max_position_embeddings")
        h = self.embed(input_ids)

        if position_ids is None:
            position_ids = torch.arange(past_len, past_len + s, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(b, s)
        if position_ids.min() < 0 or position_ids.max() >= self.config.max_position_embeddings:
            raise ValueError("position_ids exceed the RoPE table")

        if attention_mask is not None and attention_mask.dim() == 2:
            if attention_mask.shape != (b, past_len + s):
                raise ValueError("attention_mask must cover the complete sequence including KV cache")
            # [B, S] 的 0/1 padding 掩码 -> 加性 [B, 1, 1, S]
            neg = torch.finfo(h.dtype).min
            attention_mask = (1.0 - attention_mask[:, None, None, :].to(h.dtype)) * neg

        past_kv_list = past_kv_list or [None] * len(self.layers)
        caches, aux_total = [], 0.0
        for layer, past in zip(self.layers, past_kv_list):
            h, cache, aux = layer(h, self.freqs_cos, self.freqs_sin, attention_mask, position_ids, past)
            caches.append(cache)
            if aux is not None:
                aux_total = aux_total + aux
        return self.norm(h), (caches if use_cache else None), aux_total


@dataclass
class MokioMindCausalLMOutput(CausalLMOutputWithPast):
    aux_loss: Optional[torch.Tensor] = None


class MokioMindForCausalLM(nn.Module):
    _no_split_modules = ["TransformerBlock"]

    def __init__(self, config: MokioMindConfig):
        super().__init__()
        self.config = config
        self.model = MokioMindModel(config)
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        labels=None,
        past_kv_list=None,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden, caches, aux = self.model(input_ids, attention_mask, position_ids, past_kv_list, use_cache)
        logits = self.output(hidden)

        loss = None
        if labels is not None:
            labels = torch.as_tensor(labels, dtype=torch.long, device=input_ids.device)
            if labels.ndim == 1 and input_ids.shape[0] == 1:
                labels = labels.unsqueeze(0)
            if labels.shape != input_ids.shape:
                raise ValueError("labels must match input_ids")
            shift_logits = logits[:, :-1, :].contiguous().view(-1, self.config.vocab_size)
            shift_labels = labels[:, 1:].contiguous().view(-1)
            loss = (
                F.cross_entropy(shift_logits.float(), shift_labels, ignore_index=-100)
                if (shift_labels != -100).any()
                else logits.sum() * 0.0
            )
            if aux != 0.0:
                loss = loss + aux
        return MokioMindCausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=caches,
            aux_loss=None if aux == 0 else aux,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens: int = 128,
        temperature: float = 0.85,
        top_p: float = 0.85,
        top_k: int = None,
        eos_token_id: int = None,
        use_cache: bool = True,
    ):
        self.eval()
        if input_ids.ndim != 2 or input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
            raise ValueError("generation requires a nonempty [batch, sequence] prompt")
        if max_new_tokens < 0 or temperature < 0 or not 0 < top_p <= 1 or (top_k is not None and top_k < 1):
            raise ValueError("invalid generation parameters")
        if input_ids.shape[1] + max_new_tokens > self.config.max_position_embeddings:
            raise ValueError("prompt plus max_new_tokens exceeds max_position_embeddings")
        past = None
        prompt_len = input_ids.shape[1]
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        for _ in range(max_new_tokens):
            if use_cache:
                # 首步必须整段 prefill，之后每步只喂新生成的 token。
                step_input = input_ids if past is None else input_ids[:, -1:]
                out = self.forward(step_input, past_kv_list=past, use_cache=True)
                logits, past = out.logits, out.past_key_values
            else:
                cond = input_ids[:, -self.config.max_position_embeddings :]
                logits = self.forward(cond).logits
            logits = logits[:, -1, :]
            if temperature == 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature

                if top_k:
                    keep = torch.topk(logits, min(top_k, logits.size(-1))).values[..., -1:]
                    logits = logits.masked_fill(logits < keep, float("-inf"))
                if top_p < 1.0:
                    sorted_logits, sorted_i = torch.sort(logits, dim=-1, descending=True)
                    sorted_probs = F.softmax(sorted_logits, dim=-1)
                    remove = (sorted_probs.cumsum(dim=-1) - sorted_probs) >= top_p
                    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                    logits = torch.full_like(logits, float("-inf")).scatter_(1, sorted_i, sorted_logits)
                next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

            if eos_token_id is not None:
                next_id = torch.where(finished[:, None], eos_token_id, next_id)
                finished |= next_id.squeeze(-1) == eos_token_id
            input_ids = torch.cat([input_ids, next_id], dim=-1)
            if finished.all():
                break
        return input_ids[:, prompt_len:]

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)
