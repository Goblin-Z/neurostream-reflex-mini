import torch
import torch.nn as nn
import torch.nn.functional as F

from config.model_config import ReflexConfig
from typing import Optional, Tuple


class Router(nn.Module):
    """
    MoE router with state-conditioned gating (h_t → direct expert bias).

    v2 change: replaces EndoSphere soft-fusion with direct h_t → expert logits
    conditioning.  h_t (the recurrent state from SelfModel) produces a per-expert
    bias via a learnable projection, making expert selection genuinely dependent
    on the model's current "thinking state".

    Architecture:
        base_logits  = x @ gate_weight + gate_bias
        state_logits = h_t @ h_to_bias_weight     (when h_state is provided)
        logits       = base_logits + state_logits
        probs        = top-k softmax(logits)

    h_to_bias_weight is zero-initialised so the model starts from the base
    routing and gradually learns state-dependent routing over time.
    """

    def __init__(self, config: ReflexConfig):
        super().__init__()
        self.top_k = config.top_k
        self.d_model = config.d_model

        initial_n = config.n_stable + config.n_plastic
        self.gate_weight = nn.Parameter(
            torch.randn(config.d_model, initial_n) * 0.02
        )
        self.gate_bias = nn.Parameter(torch.zeros(initial_n))

        # h_t → expert bias: [d_model, n_experts], zero-initialised
        self.h_to_bias_weight = nn.Parameter(
            torch.zeros(config.d_model, initial_n)
        )

        self.verify_threshold = nn.Parameter(
            torch.tensor(config.verify_threshold_init)
        )
        self.verify_min_interval = config.verify_min_interval

        self.register_buffer('activation_running_mean', torch.zeros(initial_n))
        self.register_buffer('activation_running_var', torch.ones(initial_n))
        self.ema_momentum = 0.99

        # Gating entropy + expert utilization (Phase B: non-procedural growth triggers)
        self.register_buffer('expert_util_ema', torch.ones(initial_n) / initial_n)
        self._last_gating_entropy = torch.tensor(0.0)
        self._last_aux_loss = None
        self._util_momentum = 0.99

    # ── Dynamic gate management ──

    def add_column(self) -> int:
        col = self.gate_weight.size(1)
        new_w = torch.randn(self.d_model, 1, device=self.gate_weight.device) * 0.02
        self.gate_weight = nn.Parameter(
            torch.cat([self.gate_weight, new_w], dim=1)
        )
        new_b = torch.zeros(1, device=self.gate_bias.device)
        self.gate_bias = nn.Parameter(
            torch.cat([self.gate_bias, new_b], dim=0)
        )
        new_h = torch.zeros(self.d_model, 1, device=self.h_to_bias_weight.device)
        self.h_to_bias_weight = nn.Parameter(
            torch.cat([self.h_to_bias_weight, new_h], dim=1)
        )
        # Append to buffers (not replace) to maintain correct size
        self._update_buffer('activation_running_mean',
                            torch.cat([self.activation_running_mean,
                                       torch.zeros(1, device=self.gate_weight.device)]))
        self._update_buffer('activation_running_var',
                            torch.cat([self.activation_running_var,
                                       torch.ones(1, device=self.gate_weight.device)]))
        new_util = torch.ones(1, device=self.gate_weight.device) / self.gate_weight.size(1)
        self._update_buffer('expert_util_ema',
                            torch.cat([self.expert_util_ema, new_util]))
        return col

    def remove_column(self, col: int):
        keep = [i for i in range(self.gate_weight.size(1)) if i != col]
        self.gate_weight = nn.Parameter(self.gate_weight[:, keep])
        self.gate_bias = nn.Parameter(self.gate_bias[keep])
        self.h_to_bias_weight = nn.Parameter(
            self.h_to_bias_weight[:, keep]
        )
        self._update_buffer('activation_running_mean',
                            self.activation_running_mean[keep])
        self._update_buffer('activation_running_var',
                            self.activation_running_var[keep])
        self._update_buffer('expert_util_ema',
                            self.expert_util_ema[keep])

    def _update_buffer(self, name, new_tensor):
        if name in self._buffers:
            self._buffers[name] = new_tensor
        else:
            self.register_buffer(name, new_tensor)

    # ── Forward ──

    def forward(self, x, h_state=None, is_internal=False):
        """
        x:        [B*T, d_model]  token-level input
        h_state:  [1, d_model]     SelfModel recurrent state (or None)

        When h_state is provided, it adds a per-expert bias to the logits
        via a learned projection, conditioning expert selection on the
        current thinking state.
        """
        logits = F.linear(x, self.gate_weight.t(), self.gate_bias)
        if is_internal:
            logits += 2.0

        if h_state is not None:
            h = h_state
            if h.dim() == 1:
                h = h.unsqueeze(0)
            # h: [1, d_model] @ [d_model, n_experts] → [1, n_experts]
            # Broadcasts to [B*T, n_experts]
            logits = logits + h @ self.h_to_bias_weight

        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        # ── Update gating entropy and expert utilization ──
        if self.training:
            self._last_gating_entropy = self._gating_entropy(logits).mean().detach()
            util = torch.zeros(logits.size(-1), device=logits.device)
            n = top_k_indices.numel()
            if n > 0:
                util.scatter_add_(0, top_k_indices.view(-1),
                                  torch.ones(n, device=logits.device))
                util = util / (n / self.top_k)
            self.expert_util_ema.mul_(self._util_momentum).add_(
                util, alpha=1.0 - self._util_momentum
            )

            # ── Load-balancing auxiliary loss (Switch Transformer style) ──
            # aux = n_experts * Σ(mean_prob_i * mean_util_i)
            # Encourages uniform expert utilization; penalises collapse.
            probs = F.softmax(logits, dim=-1)
            mean_probs = probs.mean(dim=0)
            mean_util = util
            self._last_aux_loss = (
                logits.size(-1) * (mean_probs * mean_util).sum()
            )
        else:
            self._last_aux_loss = None

        return top_k_weights, top_k_indices, logits

    # ── Gating entropy ──

    def _gating_entropy(self, logits):
        """H[gating | x, h_t] = -Σ p_i log p_i (full distribution, before top-k)."""
        probs = F.softmax(logits, dim=-1)
        return -(probs * (probs.clamp(min=1e-8).log())).sum(dim=-1)

    def get_gating_stats(self):
        """
        Return internal gating signals for architecture self-modification.

        Returns:
            entropy:  average gating entropy from last forward (tensor scalar)
            utilization: expert_util_ema buffer (tensor)
        """
        return {
            'entropy': self._last_gating_entropy,
            'utilization': self.expert_util_ema.clone(),
        }

    # ── Sigma aggregation ──

    def aggregate_sigma(self, expert_sigmas, top_k_weights, top_k_indices):
        n_active = self.gate_weight.size(1)
        safe_indices = top_k_indices.clamp(0, n_active - 1)
        selected = expert_sigmas[safe_indices]
        weighted = (top_k_weights * selected).sum(dim=-1)
        return weighted.mean().item()

    def should_verify(self, sigma_aggregate, steps_since_last):
        if steps_since_last < self.verify_min_interval:
            return False, sigma_aggregate
        return sigma_aggregate > self.verify_threshold.item(), sigma_aggregate
