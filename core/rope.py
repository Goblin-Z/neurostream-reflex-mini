import torch
import torch.nn as nn
import math


class RoPE(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Applies rotary transformations to Q and K after head splitting,
    before attention computation.  Encodes relative position via
    rotation in the complex plane.

    Advantages over learned absolute position embedding:
    - No position_embedding buffer (fixes overflow for long sequences)
    - Better length generalization (extrapolates via NTK-aware scaling)
    - Relative position encoding built into the rotation

    Partial rotary (Qwen3.x 兼容): 通过 rotary_dim 只旋转前 rotary_dim 维，
    其余维度原样通过。inv_freq 的分母为 rotary_dim（与 Qwen3 实现一致）。
    """

    def __init__(self, head_dim: int, max_seq_len: int = 4096,
                 theta: float = 10000.0, rotary_dim: int = None):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        # 旋转维度（Qwen3.x partial rotary：rotary_dim = int(head_dim * factor) < head_dim；
        # 未指定时旋转全部 head_dim 维）
        self.rotary_dim = rotary_dim or head_dim
        assert 0 < self.rotary_dim <= head_dim, \
            f'rotary_dim {self.rotary_dim} must be in (0, {head_dim}]'

        # Pre-compute inverse frequencies: 1 / theta^(2i/d) for i in [0, d/2)
        inv_freq = 1.0 / (theta ** (
            torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim
        ))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # Pre-compute cached sin/cos for max_seq_len
        self._update_cache(max_seq_len)

    def _update_cache(self, seq_len: int):
        """Update sin/cos cache if needed for longer sequences."""
        if not hasattr(self, '_cached_seq_len') or seq_len > self._cached_seq_len:
            positions = torch.arange(seq_len, dtype=torch.float32,
                                     device=self.inv_freq.device)
            freqs = torch.outer(positions, self.inv_freq)  # [seq_len, rotary_dim/2]
            # Duplicate to match rotary_dim: [seq_len, rotary_dim]
            emb = torch.cat([freqs, freqs], dim=-1)
            self._cached_cos = emb.cos()
            self._cached_sin = emb.sin()
            self._cached_seq_len = seq_len

    def forward(self, x, seq_len: int = None, offset: int = 0):
        """
        Apply rotary embedding to x.

        x: [B, n_heads, T, head_dim] or [B, T, n_heads, head_dim]
        offset: 起始位置（增量解码时 = past 长度；全量前向为 0）
        Returns: same shape with rotary transformation applied.

        Rotation: for each pair (x[2i], x[2i+1]) within the first rotary_dim
        dimensions (partial rotary: 其余维度原样通过):
            x'[2i]   = x[2i] * cos(θ) - x[2i+1] * sin(θ)
            x'[2i+1] = x[2i] * sin(θ) + x[2i+1] * cos(θ)
        """
        *_, T, D = x.shape

        if seq_len is None:
            seq_len = T
        self._update_cache(offset + seq_len)

        cos = self._cached_cos[offset:offset + T].to(x.device, dtype=x.dtype)  # [T, rotary_dim]
        sin = self._cached_sin[offset:offset + T].to(x.device, dtype=x.dtype)

        # Reshape for broadcasting: [1, T, 1, D] or [T, D] depending on x shape
        if x.dim() == 4:  # [B, n_heads, T, head_dim]
            cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, T, D]
            sin = sin.unsqueeze(0).unsqueeze(0)
        else:  # [B, T, D]
            cos = cos.unsqueeze(0)  # [1, T, D]
            sin = sin.unsqueeze(0)

        # Rotate half: (x1, x2) -> (x1*cos - x2*sin, x1*sin + x2*cos)
        rd = self.rotary_dim
        x1 = x[..., :rd // 2]
        x2 = x[..., rd // 2:rd]

        cos1 = cos[..., :rd // 2]
        sin1 = sin[..., rd // 2:rd]

        rotated = torch.cat([
            x1 * cos1 - x2 * sin1,
            x1 * sin1 + x2 * cos1,
        ], dim=-1)

        if rd < D:  # partial rotary: 尾部维度原样拼接
            return torch.cat([rotated, x[..., rd:]], dim=-1)
        return rotated


def apply_rope(q, k, rope_module):
    """
    Convenience function to apply RoPE to both Q and K.

    q: [B, n_heads, T, head_dim]
    k: [B, n_kv_heads, T, head_dim]
    Returns: (q_rotated, k_rotated)
    """
    seq_len = q.size(-2)
    return rope_module(q, seq_len), rope_module(k, seq_len)
