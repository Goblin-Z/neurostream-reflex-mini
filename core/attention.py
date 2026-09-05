import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend

from config.model_config import ReflexConfig
from core.rmsnorm import RMSNorm
from core.rope import RoPE


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with modern enhancements:
    - Grouped Query Attention (GQA): fewer KV heads for memory/speed
    - Rotary Position Embedding (RoPE): replaces absolute position embedding
    - QK-Norm: RMSNorm on Q and K for training stability
    - Flash Attention: uses F.scaled_dot_product_attention when available

    Architecture:
        q_proj:  d_model -> n_heads * head_dim
        kv_proj: d_model -> 2 * n_kv_heads * head_dim
        o_proj:  n_heads * head_dim -> d_model

    GQA: n_kv_heads < n_heads, KV heads are repeated to match Q heads.
    """

    def __init__(self, config: ReflexConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        # 显式 head_dim（Qwen3.x: head_dim 可不等于 d_model // n_heads）；0 = 自动
        self.head_dim = getattr(config, 'head_dim', 0) or (config.d_model // config.n_heads)
        # Qwen3.x attn_output_gate: q_proj 输出双倍宽度，一半 query 一半 sigmoid 门控
        self.attn_gate = getattr(config, 'attn_gate', False)
        self.q_out_dim = self.n_heads * self.head_dim * (2 if self.attn_gate else 1)

        # GQA: number of KV heads (defaults to n_heads if not specified)
        self.n_kv_heads = getattr(config, 'n_kv_heads', config.n_heads)
        if self.n_kv_heads > self.n_heads:
            self.n_kv_heads = self.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads  # repeat factor

        # Projections
        self.q_proj = nn.Linear(config.d_model, self.q_out_dim,
                                bias=False)
        self.kv_proj = nn.Linear(config.d_model, 2 * self.n_kv_heads * self.head_dim,
                                 bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.d_model,
                                bias=False)

        # QK-Norm: per-head RMSNorm on Q and K
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        # L4 记忆: KV 缓存开关与最近一轮 KV（由 model/pipeline 控制）
        self._kv_cache_enabled = False
        self._last_kv = None

        # RoPE
        rope_theta = getattr(config, 'rope_theta', 10000.0)
        partial = getattr(config, 'partial_rotary_factor', 1.0)
        rotary_dim = int(self.head_dim * partial) if partial < 1.0 else None
        self.rope = RoPE(self.head_dim,
                         max_seq_len=config.max_seq_len,
                         theta=rope_theta,
                         rotary_dim=rotary_dim)

        self.attn_dropout = getattr(config, 'attention_dropout', config.dropout)
        self.out_dropout = nn.Dropout(config.dropout)

        # Scale factor
        self.scale = self.head_dim ** -0.5

    def forward(self, x, attention_mask=None, is_causal=True, mem_kv=None,
                rope_offset=0):
        """
        x: [B, T, d_model]
        attention_mask: [B, T] (1 = valid, 0 = padding)
        mem_kv: (mem_k, mem_v) from MemoryBank — 历史对话 KV，
                拼接参与注意力（L4 内容记忆）。mem_k/mem_v: [B, H, T_mem, hd]
        rope_offset: 增量解码起始位置（past 长度）；全量前向为 0
        Returns: [B, T, d_model]
        """
        B, T, _ = x.shape

        # Q projection (attn_gate: 双倍输出，一半 query 一半 gate)
        q = self.q_proj(x)  # [B, T, q_out_dim]
        if self.attn_gate:
            # Qwen3.x 布局: [B, T, n_heads, head_dim*2] → chunk 后半为 gate
            q = q.view(B, T, self.n_heads, self.head_dim * 2)
            query_states, gate = torch.chunk(q, 2, dim=-1)
            q = query_states.transpose(1, 2)
            gate = gate.reshape(B, T, -1)  # [B, T, n_heads*head_dim]
        else:
            q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            gate = None
        # q: [B, n_heads, T, head_dim]

        # KV projection
        kv = self.kv_proj(x)  # [B, T, 2 * n_kv_heads * head_dim]
        k, v = kv.chunk(2, dim=-1)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # k, v: [B, n_kv_heads, T, head_dim]

        # QK-Norm (per-head RMSNorm)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE on Q and K
        seq_len = q.size(-2)
        q = self.rope(q, seq_len, offset=rope_offset)
        k = self.rope(k, seq_len, offset=rope_offset)

        # L4: 缓存本轮 KV（detach，供轮次结束写入 MemoryBank）
        if getattr(self, '_kv_cache_enabled', False):
            self._last_kv = (k.detach(), v.detach())

        # L4: 拼接历史对话 KV（记忆区无 causal，全可见）
        if mem_kv is not None:
            mem_k, mem_v = mem_kv
            k = torch.cat([mem_k, k], dim=2)
            v = torch.cat([mem_v, v], dim=2)

        # GQA: repeat KV heads to match Q heads (must be contiguous for Flash Attention)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1).contiguous()
            v = v.repeat_interleave(self.n_rep, dim=1).contiguous()
        else:
            k = k.contiguous()
            v = v.contiguous()

        # Q must also be contiguous after RoPE transformation
        q = q.contiguous()

        # Flash Attention via F.scaled_dot_product_attention
        # Use is_causal=True for causal mask (enables Flash Attention kernel).
        # Only fall back to explicit attn_mask for padding (non-causal cases).
        # Apply dropout during training; 0 during eval to always enable Flash.
        dropout_p = self.attn_dropout if self.training else 0.0
        
        if attention_mask is not None or mem_kv is not None:
            # Combined mask: padding + causal over current span;
            # memory span is fully visible (no causal, no padding mask).
            T_total = k.size(-2)
            T_mem = T_total - T
            attn_mask = torch.zeros(
                (B, self.n_heads, T, T_total),
                device=x.device, dtype=q.dtype,
            )
            if attention_mask is not None:
                # padding 屏蔽当前区
                pad = (attention_mask == 0).unsqueeze(1).unsqueeze(2)  # [B,1,1,T]
                attn_mask[:, :, :, T_mem:] = attn_mask[:, :, :, T_mem:].masked_fill(
                    pad.expand(B, self.n_heads, T, T), float('-inf'))
            if is_causal:
                causal = torch.triu(
                    torch.full((T, T), float('-inf'), device=x.device, dtype=q.dtype),
                    diagonal=1,
                )
                attn_mask[:, :, :, T_mem:] += causal.unsqueeze(0)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask,
                dropout_p=dropout_p, is_causal=False,
            )
        else:
            # No padding/memory: use is_causal flag (enables Flash Attention)
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )
        # out: [B, n_heads, T, head_dim]

        # Merge heads and output projection
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        # Qwen3.x attn_output_gate: out = o_proj(out * sigmoid(gate))
        if gate is not None:
            out = out * torch.sigmoid(gate)
        out = self.out_dropout(self.o_proj(out))
        return out
