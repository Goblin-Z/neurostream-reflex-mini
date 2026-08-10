import torch
import torch.nn as nn
import torch.nn.functional as F

from core.rmsnorm import RMSNorm


class BlockDeltaAttnRes(nn.Module):
    """
    Block Delta Attention Residuals (based on Kimi AttnRes + Delta improvement).

    Replaces fixed additive residual accumulation between blocks with
    learned softmax attention over block-level delta representations.

    Key design (fusion of three papers):
    - AttnRes (Kimi 2026): softmax attention over previous layer outputs
    - Delta AttnRes (2026): attend over deltas, not cumulative states
      -> higher routing contrast (max weight ~0.6 vs ~0.2)
    - Low-Rank AttnRes (2026): low-rank keys for efficiency

    Sink mitigation: deltas are L2-normalized before K projection so
    routing is direction-based, not magnitude-based.  V uses
    unnormalized deltas to preserve full representation information.

    Architecture:
        Q = W_q(RMSNorm(h_current))              # [B, T, rank]
        K = W_k(normalize(delta_j))              # [B, T, rank] per source
        V = W_v(delta_j)                         # [B, T, d_model] per source
        attn = softmax(Q . K^T / sqrt(rank))     # [B, T, n_sources]
        out = sum_j attn_j * V_j                # [B, T, d_model]
        h = h_current + post_norm(out_proj(out))

    Initialization: W_v = 0, post_norm scale = 1e-3
    -> starts as near-standard Transformer, learns to use AttnRes over time.
    """

    def __init__(self, d_model: int, rank: int):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.scale = rank ** -0.5

        # Q from current state (via RMSNorm for stability)
        self.q_norm = RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, rank, bias=False)

        # K from normalized deltas (low-rank for efficiency)
        self.k_proj = nn.Linear(d_model, rank, bias=False)

        # V from unnormalized deltas (full-dimensional to preserve information)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # PostDA-Norm: scale initialized to 1e-3 so AttnRes starts near-zero
        self.post_norm = RMSNorm(d_model)

        # Initialize: Q, K, V, out_proj all with small normal
        # post_norm.weight=1e-3 controls the overall magnitude (AttnRes starts near-zero)
        # 不能有任何层为零，否则梯度链条断裂
        nn.init.normal_(self.q_proj.weight, std=1.0 / d_model)
        nn.init.normal_(self.k_proj.weight, std=1.0 / rank)
        nn.init.normal_(self.v_proj.weight, std=1.0 / d_model)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        self.post_norm.weight.data.fill_(1e-3)

        # Cache for routing stats (not a learnable parameter)
        self._last_attn_weights = None

    def forward(self, h_current, block_outputs, memory_bank=None):
        """
        Apply Delta Attention Residual aggregation.

        Args:
            h_current: [B, T, d_model] - current hidden state (after block)
            block_outputs: list of [B, T, d_model] - outputs at each block
                           boundary, starting with h_0 (embedding output).
                           Length >= 2 (at least h_0 and current block output).
            memory_bank: L2 长期记忆——语义记忆向量作为额外 source，
                         模型注意力自行决定回忆权重（方向 B）。

        Returns: [B, T, d_model] - h_current + post_norm(attn_output)
        """
        n_sources = len(block_outputs)
        if n_sources < 2:
            return h_current

        # Compute deltas: delta_0 = h_0, delta_j = h_j - h_{j-1}
        deltas = [block_outputs[0]]
        for j in range(1, n_sources):
            deltas.append(block_outputs[j] - block_outputs[j - 1])

        # L2 记忆 source: 语义记忆向量作为额外 delta（模型注意力自由分配）
        mem_vectors = []
        mem_indices = []
        n_base = len(deltas)   # 基础 delta 数量（记忆列从 n_base 开始）
        if memory_bank is not None:
            mem_vectors, mem_indices = memory_bank.retrieve_with_index(
                h_current)
            for mv in mem_vectors:
                deltas.append(mv.unsqueeze(0).unsqueeze(0).expand(
                    h_current.size(0), h_current.size(1), -1))

        # Q from current state
        q = self.q_proj(self.q_norm(h_current))  # [B, T, rank]

        # K from L2-normalized deltas (sink mitigation: direction-based routing)
        keys = []
        for delta in deltas:
            norm = delta.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            normalized = delta / norm
            keys.append(self.k_proj(normalized))
        k = torch.stack(keys, dim=2)  # [B, T, n_sources, rank]

        # V from unnormalized deltas (full-dimensional, preserves magnitude)
        vals = [self.v_proj(delta) for delta in deltas]
        v = torch.stack(vals, dim=2)  # [B, T, n_sources, d_model]

        # Attention: [B, T, rank] x [B, T, n_sources, rank] -> [B, T, n_sources]
        scores = torch.einsum('btr,btnr->btn', q, k) * self.scale
        attn = F.softmax(scores, dim=-1)  # [B, T, n_sources]

        # Cache for routing stats
        self._last_attn_weights = attn.detach()

        # 自发固化: 记忆 source 的注意力权重回传（行为驱动的 salience 累积）
        if memory_bank is not None and mem_indices.numel() > 0:
            with torch.no_grad():
                # 每条记忆的聚焦度 = 注意力在该 source 列上的均值
                mem_attn = attn[:, :, n_base:].mean(dim=(0, 1))  # [k]
                memory_bank.accumulate_salience(mem_indices, mem_attn)

        # Weighted sum: [B, T, n_sources] x [B, T, n_sources, d] -> [B, T, d]
        out = torch.einsum('btn,btnd->btd', attn, v)  # [B, T, d_model]

        # Output projection + PostDA-Norm + residual
        out = self.out_proj(out)
        return h_current + self.post_norm(out)

    def get_routing_stats(self):
        """
        Return routing statistics from the last forward pass.

        Returns dict with:
          - 'max_weight': mean of max attention weight per token (routing contrast)
          - 'entropy': mean attention entropy (routing diversity)
          - 'n_sources': number of attention sources
        Returns None if no forward pass has been made.
        """
        if self._last_attn_weights is None:
            return None
        attn = self._last_attn_weights  # [B, T, n_sources]
        max_weight = attn.max(dim=-1).values.mean().item()
        entropy = -(attn.clamp(min=1e-8) * attn.clamp(min=1e-8).log()).sum(dim=-1).mean().item()
        return {
            'max_weight': max_weight,
            'entropy': entropy,
            'n_sources': attn.size(-1),
        }


class AttnResStack(nn.Module):
    """
    Manages Block Delta Attention Residuals across the entire model.

    Creates one BlockDeltaAttnRes module per inter-block boundary.
    During forward, the model calls this to apply AttnRes at each
    block boundary.
    """

    def __init__(self, n_layers: int, block_size: int, d_model: int,
                 rank: int):
        super().__init__()
        self.block_size = block_size
        n_boundaries = (n_layers - 1) // block_size
        self.modules_list = nn.ModuleList([
            BlockDeltaAttnRes(d_model, rank)
            for _ in range(n_boundaries)
        ])

    def apply(self, boundary_idx, h_current, block_outputs,
              memory_bank=None):
        """Apply AttnRes at the given boundary index."""
        if boundary_idx >= len(self.modules_list):
            return h_current
        return self.modules_list[boundary_idx](
            h_current, block_outputs, memory_bank=memory_bank)

    @property
    def n_boundaries(self):
        return len(self.modules_list)

    def get_all_routing_stats(self):
        """Return routing stats for all boundaries."""
        stats = []
        for i, module in enumerate(self.modules_list):
            s = module.get_routing_stats()
            if s is not None:
                stats.append((i, s))
        return stats
