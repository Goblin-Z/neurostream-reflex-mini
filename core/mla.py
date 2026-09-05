"""
core/mla.py — DeepSeek-V2 MLA（Multi-head Latent Attention）+ YaRN RoPE 移植。

来源：deepseek-ai/DeepSeek-V2-Lite 官方 modeling_deepseek.py（DeepSeek-V2-Lite
的 q_lora_rank=null，Q 不做低秩压缩）。权重名与官方 safetensors 一致，直接加载。

与 core/attention.py 的 MultiHeadAttention 接口兼容：
  forward(x, attention_mask=None, mem_kv=None, rope_offset=0) -> [B, T, d_model]
  - mem_kv: L4 记忆 KV（展开的 key/value，全可见前缀）
  - rope_offset: 增量解码起始位置（past 长度）
  - _kv_cache_enabled / _last_kv: L4 轮次 KV 缓存（展开态 key/value）

MLA 说明：
  - Q: q_proj → [B,T,16,192]（qk_nope=128 + qk_rope=64），不压缩
  - KV: kv_a_proj_with_mqa → compressed_kv(512) + k_pe(64)（所有头共享）
        kv_b_proj(kv_a_layernorm(compressed_kv)) → [B,T,16,256] → k_nope(128)+v(128)
  - RoPE 只旋转 q_pe/k_pe：interleaved 布局（配对 (2d,2d+1) 用频率 f_d），
    与 transformers 5.x built-in 的复数 view_as_complex 语义逐位一致；
    ⚠ 官方 remote code 是 half-split rotate_half，与 built-in 不一致，
      verify 金标准为 built-in，本实现与之对齐
  - softmax_scale = q_head_dim^-0.5（与 built-in 对齐，不含 mscale²）

注意：本项目以"展开的 key/value"作为 past/记忆 KV（逻辑与 GQA 层一致，
简化增量解码与 L4 记忆复用；官方缓存压缩态以省显存，本实现展开态
KV 每 token 每层 = 16×192 + 16×128 ≈ 10KB bf16，8K 上下文约 2.2GB，可接受）。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config.model_config import ReflexConfig
from core.rmsnorm import RMSNorm


# ── YaRN 辅助函数（官方实现原样移植）──

def yarn_find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def yarn_find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    low = math.floor(yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_linear_ramp_mask(min_v, max_v, dim):
    if min_v == max_v:
        max_v += 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32) - min_v) / (max_v - min_v)
    return torch.clamp(linear_func, 0, 1)


class DeepseekYarnRope(nn.Module):
    """
    DeepSeek-V2 YaRN 旋转位置编码（与 transformers 5.x built-in 逐位对齐）。

    - 只旋转 q_pe/k_pe（qk_rope_head_dim 维）
    - interleaved 布局：维度配对 (2d, 2d+1) 用频率 f_d 旋转（built-in 的
      复数 view_as_complex 语义）；cos/sin 取缓存【前 D//2 列】（缓存
      emb=cat((freqs,freqs))，前 D//2 列即频率 0..D/2-1 的 cos/sin）。
      ⚠ 坑：官方 remote code（modeling_deepseek.py）是 half-split
      rotate_half；transformers 5.x built-in 是 interleaved 复数——两者
      不一致，verify 金标准为 built-in，本实现与之对齐。
    - YaRN 频率混合（freq_extra / freq_inter + 线性 ramp 掩码）
    - cos/sin 乘 _mscale（本配置 mscale=mscale_all_dim → _mscale=1.0）
    """

    def __init__(self, dim: int, theta: float = 10000.0, max_seq_len: int = 8192,
                 scaling_factor: float = 40.0, original_max_position_embeddings: int = 4096,
                 beta_fast: float = 32, beta_slow: float = 1,
                 mscale: float = 0.707, mscale_all_dim: float = 0.707):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim

        freq_extra = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        freq_inter = 1.0 / (
            scaling_factor * theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        low, high = yarn_find_correction_range(
            beta_fast, beta_slow, dim, theta, original_max_position_embeddings)
        inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2)
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        _mscale = float(
            yarn_get_mscale(scaling_factor, mscale)
            / yarn_get_mscale(scaling_factor, mscale_all_dim)
        )
        self._mscale = _mscale
        self._update_cache(max_seq_len)

    def _update_cache(self, seq_len: int):
        if not hasattr(self, '_cached_seq_len') or seq_len > self._cached_seq_len:
            t = torch.arange(seq_len, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq)          # [seq, dim/2]
            emb = torch.cat((freqs, freqs), dim=-1)         # [seq, dim]
            self._cached_cos = (emb.cos() * self._mscale)
            self._cached_sin = (emb.sin() * self._mscale)
            self._cached_seq_len = seq_len

    def forward(self, x, offset: int = 0):
        """
        对 q_pe/k_pe 应用 interleaved 旋转（transformers 5.x built-in 语义）。
        x: [B, H, T, dim]（或 [B, T, H, dim] 由 H==1 处理）
        offset: 增量解码起始位置
        """
        *_, T, D = x.shape
        self._update_cache(offset + T)
        # cos/sin 保持 fp32（built-in 的 freqs_cis 是 complex64），
        # 旋转在 fp32 中完成、最后才转回 x.dtype —— 与 built-in 相同路径；
        # 此前在 bf16 中旋转引入每层 ~1e-3 确定性差异，27 层累积 +
        # 路由分叉 → verify top-1 被拖到 ~78%
        cos = self._cached_cos[offset:offset + T].to(x.device)
        sin = self._cached_sin[offset:offset + T].to(x.device)
        if x.dim() == 4:
            cos = cos.unsqueeze(0).unsqueeze(0)   # [1,1,T,d]
            sin = sin.unsqueeze(0).unsqueeze(0)
        else:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)

        # interleaved：维度配对 (x[2d], x[2d+1]) 用频率 f_d 的 cos/sin。
        # ⚠ cos/sin 必须取【前 D//2 列】（=频率 0..D/2-1），不能取 0::2 列
        #   （会混入后一半重复列，数值差 ~5.5）；也不能用 half-split
        #   （built-in 是复数 interleaved，half-split 差 ~6.1）。
        x_f = x.float()
        a = x_f[..., 0::2]
        b = x_f[..., 1::2]
        c = cos[..., : D // 2]
        s = sin[..., : D // 2]
        rot_a = a * c - b * s
        rot_b = a * s + b * c
        out = torch.stack([rot_a, rot_b], dim=-1).reshape(*x.shape)
        return out.to(x.dtype)


class DeepseekMLA(nn.Module):
    """
    DeepSeek-V2-Lite MLA 注意力（q_lora_rank=null 版）。

    权重名与官方 safetensors 一致：
      q_proj / kv_a_proj_with_mqa / kv_a_layernorm / kv_b_proj / o_proj
    """

    def __init__(self, config: ReflexConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim

        self.q_proj = nn.Linear(config.d_model, self.n_heads * self.q_head_dim,
                                bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(
            config.d_model, config.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_a_layernorm = RMSNorm(config.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.v_head_dim, config.d_model,
                                bias=False)
        self.attention_dropout = getattr(config, 'attention_dropout', 0.0)

        # softmax scale：q_head_dim^-0.5（与 transformers 5.x built-in 对齐；
        # built-in 的 DeepseekV2Attention.scaling 不含 mscale²，mscale 仅经
        # rope attention_scaling 使用，本配置 mscale=mscale_all_dim → 无缩放。
        # ⚠ 官方 remote code 是 ×mscale²，与 built-in 不一致——verify 金标准
        #   为 built-in，故不乘 mscale²）
        self.softmax_scale = self.q_head_dim ** (-0.5)

        rope_scaling = getattr(config, 'rope_scaling', None) or {}
        # YaRN rope（interleaved，与 transformers 5.x built-in 复数语义对齐）
        self.rope = DeepseekYarnRope(
            dim=self.qk_rope_head_dim,
            theta=getattr(config, 'rope_theta', 10000.0),
            max_seq_len=config.max_seq_len,
            scaling_factor=rope_scaling.get('factor', 1.0),
            original_max_position_embeddings=rope_scaling.get(
                'original_max_position_embeddings', config.max_seq_len),
            beta_fast=rope_scaling.get('beta_fast', 32),
            beta_slow=rope_scaling.get('beta_slow', 1),
            mscale=rope_scaling.get('mscale', 1),
            mscale_all_dim=rope_scaling.get('mscale_all_dim', 0),
        )

        # L4 记忆: KV 缓存开关与最近一轮 KV（展开态，与 MHA 接口一致）
        self._kv_cache_enabled = False
        self._last_kv = None

    def forward(self, x, attention_mask=None, mem_kv=None, rope_offset=0):
        """
        x: [B, T, d_model]
        attention_mask: [B, T] (1=valid, 0=padding)
        mem_kv: (mem_k, mem_v) — 展开的 key/value（[B, H, T_mem, 192/128]）
        rope_offset: 增量解码起始位置
        Returns: [B, T, d_model]
        """
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim],
                                   dim=-1)

        compressed_kv, k_pe = torch.split(
            self.kv_a_proj_with_mqa(x),
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = k_pe.view(B, T, 1, self.qk_rope_head_dim).transpose(1, 2)

        kv = (self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
              .view(B, T, self.n_heads,
                    self.qk_nope_head_dim + self.v_head_dim)
              .transpose(1, 2))
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # YaRN RoPE（half-split，官方 rotate_half 语义）
        q_pe = self.rope(q_pe, offset=rope_offset)
        k_pe = self.rope(k_pe, offset=rope_offset)
        # MLA: k_pe 为单头共享（[B,1,T,64]），拼接前广播到所有头
        # （官方用 new_empty + 广播赋值实现，此处等价）
        k_pe = k_pe.expand(B, self.n_heads, T, self.qk_rope_head_dim).contiguous()

        q = torch.cat([q_nope, q_pe], dim=-1)   # [B, H, T, 192]
        k = torch.cat([k_nope, k_pe], dim=-1)   # [B, H, T, 192]

        # L4: 缓存本轮 KV（展开态，detach）
        if getattr(self, '_kv_cache_enabled', False):
            self._last_kv = (k.detach(), v.detach())

        # L4: 拼接历史对话 KV（记忆区无 causal，全可见）
        if mem_kv is not None:
            mem_k, mem_v = mem_kv
            k = torch.cat([mem_k, k], dim=2)
            v = torch.cat([mem_v, v], dim=2)

        # 注意力（scale = q_head_dim^-0.5，与 transformers 5.x built-in 一致）
        dropout_p = self.attention_dropout if self.training else 0.0
        T_total = k.size(-2)
        T_mem = T_total - T
        attn_mask = torch.zeros((1, 1, T, T_total),
                                device=x.device, dtype=q.dtype)
        if attention_mask is not None:
            pad = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
            attn_mask[:, :, :, T_mem:] = attn_mask[:, :, :, T_mem:].masked_fill(
                pad.expand(B, 1, T, T), float('-inf'))
        # 显式 causal mask + is_causal=False：与 transformers 5.x sdpa 接口
        # 完全同路径（built-in 在 mask 非 None 时 is_causal 被置 False；
        # is_causal=True 在 CUDA flash 内核走不同分支，产生确定性数值差异，
        # 27 层累积 → verify max|Δ|≈3.7、top-1 78%）
        causal = torch.triu(
            torch.full((T, T), float('-inf'), device=x.device, dtype=q.dtype),
            diagonal=1)
        attn_mask[:, :, :, T_mem:] += causal.unsqueeze(0)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
            is_causal=False, scale=self.softmax_scale)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.o_proj(out)
        return out
