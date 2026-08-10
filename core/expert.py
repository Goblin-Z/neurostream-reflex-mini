import math
import uuid
import torch
import torch.nn as nn
import torch.nn.functional as F

from config.model_config import ReflexConfig
from core.rmsnorm import RMSNorm


def swish_derivative(x):
    """
    Derivative of Swish/SiLU: f(x) = x * sigmoid(x)
    f'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    """
    sig = torch.sigmoid(x)
    return sig + x * sig * (1.0 - sig)


class Expert(nn.Module):
    """
    MoE expert with SwiGLU FFN, uncertainty head, and local Hebbian update.

    SwiGLU: out = Swish(x @ W_gate) ⊙ (x @ W_up) @ W_down
    - 3 weight matrices instead of 2, but d_ff ≈ 2/3 of original
    - Better quality than GELU FFN (proven in LLaMA, PaLM)

    Each expert:
      - Has a baseline LR (lower = more stable)
      - Outputs both a hidden representation and a scalar uncertainty
      - Updates itself via Hebbian rule with SwiGLU gradient
      - Has an activation gate that prevents updates when hidden norm is large
      - Buffers: _input, _gate_pre, _gate, _up, _hidden, _output
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float,
                 baseline_lr: float = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.baseline_lr = baseline_lr

        # SwiGLU: 3 projections
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()  # Swish

        self.uncertainty_head = nn.Sequential(
            RMSNorm(d_ff),
            nn.Linear(d_ff, 32),
            nn.GELU(),
            nn.Linear(32, 8),
            nn.GELU(),
            nn.Linear(8, 1),
        )

        self.register_buffer('lr_bias', torch.tensor(0.0))
        self.register_buffer('uncertainty_ema', torch.tensor(0.5))
        self.register_buffer('activation_ema', torch.tensor(0.0))

        self.query_proj = nn.Linear(d_model, d_model, bias=False)

        # SwiGLU buffers for Hebbian update
        self._input: torch.Tensor = None
        self._gate_pre: torch.Tensor = None
        self._gate: torch.Tensor = None
        self._up: torch.Tensor = None
        self._hidden: torch.Tensor = None
        self._output: torch.Tensor = None

        # Hebbian momentum buffers (EMA of gradients for stable online learning)
        self._mom_w_down: torch.Tensor = None
        self._mom_w_gate: torch.Tensor = None
        self._mom_w_up: torch.Tensor = None

        self.id: str = uuid.uuid4().hex[:12]

    def forward(self, x, save_hebbian_buffers=True):
        """
        SwiGLU forward: out = SiLU(x @ W_gate) ⊙ (x @ W_up) @ W_down

        Returns: (out, sigma) where sigma is per-token uncertainty.
        """
        g_pre = self.w_gate(x)       # [*, d_ff]
        g = self.activation(g_pre)   # SiLU(g_pre)
        u = self.w_up(x)            # [*, d_ff]
        h = g * u                   # gated hidden [*, d_ff]

        if self.training and save_hebbian_buffers:
            self._input = x.detach().clone()
            self._gate_pre = g_pre.detach().clone()
            self._gate = g.detach().clone()
            self._up = u.detach().clone()
            self._hidden = h.detach().clone()

        h = self.dropout(h)
        out = self.w_down(h)        # [*, d_model]
        sigma = torch.sigmoid(self.uncertainty_head(h))
        self._output = out.detach() if not save_hebbian_buffers else out
        self.uncertainty_ema.mul_(0.9).add_(sigma.mean().detach(), alpha=0.1)
        self.activation_ema.mul_(0.9).add_(
            torch.norm(h, p=2, dim=-1).mean().detach(), alpha=0.1
        )
        return out, sigma

    def update_local(self, grad_y, lr, gamma=1.0):
        """
        Hebbian-style local update for SwiGLU FFN.

        grad_y: ∂loss/∂expert._output
        lr:     effective learning rate
        gamma:  modulation (sigma/verify_threshold * gate * focus_boost)

        SwiGLU: out = SiLU(x @ W_gate) ⊙ (x @ W_up) @ W_down
        Let g = SiLU(g_pre), u = x @ W_up, h = g ⊙ u
        out = h @ W_down

        Gradients:
          ∂L/∂W_down = grad_y^T @ h
          ∂L/∂h = grad_y @ W_down
          ∂L/∂g = ∂L/∂h * u       (chain through ⊙)
          ∂L/∂u = ∂L/∂h * g
          ∂L/∂g_pre = ∂L/∂g * SiLU'(g_pre)
          ∂L/∂W_gate = ∂L/∂g_pre^T @ x
          ∂L/∂W_up = ∂L/∂u^T @ x
        """
        if (grad_y is None or self._input is None
                or self._hidden is None or self._gate_pre is None
                or self._gate is None or self._up is None):
            return

        d_model = grad_y.size(-1)
        d_ff = self._hidden.size(-1)

        grad_y_2d = grad_y.reshape(-1, d_model)
        h_2d = self._hidden.reshape(-1, d_ff)
        g_pre_2d = self._gate_pre.reshape(-1, d_ff)
        g_2d = self._gate.reshape(-1, d_ff)
        u_2d = self._up.reshape(-1, d_ff)
        x_2d = self._input.reshape(-1, d_model)

        # Activation gate (dimension-normalized)
        norm_h = torch.norm(h_2d, p=2, dim=-1).mean() / (d_ff ** 0.5)
        gate = 1.0 / (1.0 + torch.exp(5.0 * (norm_h - 3.0)))
        update_scale = lr * gamma * gate

        # ∂L/∂W_down = grad_y^T @ h
        delta_w_down = grad_y_2d.t() @ h_2d
        if torch.isfinite(delta_w_down).all():
            self.w_down.weight.data -= update_scale * delta_w_down

        # ∂L/∂h = grad_y @ W_down
        grad_h = grad_y_2d @ self.w_down.weight.data  # [N, d_ff]

        # Chain through element-wise product: h = g ⊙ u
        grad_g = grad_h * u_2d   # ∂L/∂g
        grad_u = grad_h * g_2d   # ∂L/∂u

        # SiLU derivative
        grad_g_pre = grad_g * swish_derivative(g_pre_2d)

        # ∂L/∂W_gate = grad_g_pre^T @ x
        delta_w_gate = grad_g_pre.t() @ x_2d
        if torch.isfinite(delta_w_gate).all():
            self.w_gate.weight.data -= update_scale * delta_w_gate

        # ∂L/∂W_up = grad_u^T @ x
        delta_w_up = grad_u.t() @ x_2d
        if torch.isfinite(delta_w_up).all():
            self.w_up.weight.data -= update_scale * delta_w_up

    def generate_query(self, context_vector):
        with torch.no_grad():
            return self.query_proj(context_vector)

    def reset_weights(self):
        nn.init.xavier_uniform_(self.w_gate.weight)
        nn.init.xavier_uniform_(self.w_up.weight)
        nn.init.xavier_uniform_(self.w_down.weight)

    def clear_buffers(self):
        self._input = None
        self._gate_pre = None
        self._gate = None
        self._up = None
        self._hidden = None
        self._output = None

    def clear_momentum(self):
        """Clear Hebbian momentum buffers (e.g. after soft_reset)."""
        self._mom_w_down = None
        self._mom_w_gate = None
        self._mom_w_up = None

    @property
    def effective_lr(self):
        return self.baseline_lr * math.exp(max(-2.0, min(2.0, self.lr_bias.item())))

    @property
    def avg_uncertainty(self):
        return self.uncertainty_ema.item()
