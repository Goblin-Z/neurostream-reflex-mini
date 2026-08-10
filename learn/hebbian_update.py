import torch
import math


_MAX_UPDATE_SCALE = 1e-2
_MAX_DELTA_NORM = 1.0
_MOMENTUM = 0.9


def _apply_momentum(expert, attr_name, delta):
    """Apply momentum (EMA of gradients) to stabilize Hebbian updates."""
    mom = getattr(expert, attr_name, None)
    if mom is None:
        new_mom = delta.clone()
    else:
        new_mom = _MOMENTUM * mom + (1.0 - _MOMENTUM) * delta
    setattr(expert, attr_name, new_mom)
    return new_mom


def focal_update(grad_y, expert, lr, base_gamma=1.0, focus_boost=1.0):
    """
    Hebbian-style local update with SwiGLU gradient.

    grad_y:         ∂loss/∂expert._output
    expert:         Expert instance (must have SwiGLU buffers:
                    _input, _gate_pre, _gate, _up, _hidden)
    lr:             effective learning rate
    base_gamma:     base modulation (sigma/verify_threshold)
    focus_boost:    >1 for experts targeted by external feedback

    SwiGLU: out = SiLU(x @ W_gate) ⊙ (x @ W_up) @ W_down
    Let g = SiLU(g_pre), u = x @ W_up, h = g ⊙ u
    out = h @ W_down

    Gradients (manual backprop, no autograd):
      ∂L/∂W_down = grad_y^T @ h
      ∂L/∂h = grad_y @ W_down          (uses PRE-update W_down)
      ∂L/∂g = ∂L/∂h * u       (chain through ⊙)
      ∂L/∂u = ∂L/∂h * g
      ∂L/∂g_pre = ∂L/∂g * SiLU'(g_pre)
      ∂L/∂W_gate = ∂L/∂g_pre^T @ x
      ∂L/∂W_up = ∂L/∂u^T @ x

    The activation gate 1/(1+exp(5*(norm_h/sqrt(d_ff)-3))) prevents
    updates when per-dimension activation magnitude is pathologically large.
    """
    if (grad_y is None or expert._input is None
            or expert._hidden is None or expert._gate_pre is None
            or expert._gate is None or expert._up is None):
        return

    d_model = grad_y.size(-1)
    d_ff = expert._hidden.size(-1)

    grad_y_2d = grad_y.reshape(-1, d_model)
    h_2d = expert._hidden.reshape(-1, d_ff)
    g_pre_2d = expert._gate_pre.reshape(-1, d_ff)
    g_2d = expert._gate.reshape(-1, d_ff)
    u_2d = expert._up.reshape(-1, d_ff)
    x_2d = expert._input.reshape(-1, d_model)

    # Normalise norm by sqrt(d_ff) so the gate threshold is dimension-independent
    norm_h = torch.norm(h_2d, p=2, dim=-1).mean() / (d_ff ** 0.5)
    gate = 1.0 / (1.0 + torch.exp(5.0 * (norm_h - 3.0)))

    gamma = base_gamma * gate * focus_boost
    update_scale = lr * gamma
    update_scale = min(update_scale, _MAX_UPDATE_SCALE)

    # Save pre-update W_down for correct backprop to W_gate / W_up
    w_down_old = expert.w_down.weight.data.clone()

    # ∂L/∂W_down = grad_y^T @ h
    delta_w_down = grad_y_2d.t() @ h_2d
    if torch.isfinite(delta_w_down).all():
        delta_w_down = _apply_momentum(expert, '_mom_w_down', delta_w_down)
        d_norm = delta_w_down.norm()
        if d_norm > _MAX_DELTA_NORM:
            delta_w_down = delta_w_down * (_MAX_DELTA_NORM / d_norm)
        expert.w_down.weight.data -= update_scale * delta_w_down
        if expert.w_down.bias is not None:
            delta_b_down = grad_y_2d.sum(dim=0)
            expert.w_down.bias.data -= update_scale * delta_b_down

    # ∂L/∂h = grad_y @ W_down  (MUST use pre-update W_down)
    grad_h = grad_y_2d @ w_down_old  # [N, d_ff]

    # Chain through element-wise product: h = g ⊙ u
    grad_g = grad_h * u_2d   # ∂L/∂g
    grad_u = grad_h * g_2d   # ∂L/∂u

    # SiLU derivative: SiLU'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    sig = torch.sigmoid(g_pre_2d)
    silu_deriv = sig + g_pre_2d * sig * (1.0 - sig)
    grad_g_pre = grad_g * silu_deriv

    # ∂L/∂W_gate = grad_g_pre^T @ x
    delta_w_gate = grad_g_pre.t() @ x_2d
    if torch.isfinite(delta_w_gate).all():
        delta_w_gate = _apply_momentum(expert, '_mom_w_gate', delta_w_gate)
        d_norm = delta_w_gate.norm()
        if d_norm > _MAX_DELTA_NORM:
            delta_w_gate = delta_w_gate * (_MAX_DELTA_NORM / d_norm)
        expert.w_gate.weight.data -= update_scale * delta_w_gate
        if expert.w_gate.bias is not None:
            delta_b_gate = grad_g_pre.sum(dim=0)
            expert.w_gate.bias.data -= update_scale * delta_b_gate

    # ∂L/∂W_up = grad_u^T @ x
    delta_w_up = grad_u.t() @ x_2d
    if torch.isfinite(delta_w_up).all():
        delta_w_up = _apply_momentum(expert, '_mom_w_up', delta_w_up)
        d_norm = delta_w_up.norm()
        if d_norm > _MAX_DELTA_NORM:
            delta_w_up = delta_w_up * (_MAX_DELTA_NORM / d_norm)
        expert.w_up.weight.data -= update_scale * delta_w_up
        if expert.w_up.bias is not None:
            delta_b_up = grad_u.sum(dim=0)
            expert.w_up.bias.data -= update_scale * delta_b_up
