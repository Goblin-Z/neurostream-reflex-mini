import torch
import torch.nn as nn

from config.model_config import ReflexConfig
from core.rmsnorm import RMSNorm


class ReflexCritic(nn.Module):
    """
    Value estimator V(s) for Actor-Critic modulation.

    Architecture: d_model -> 256 -> 128 -> 64 -> 1 (4-layer MLP)

    Training signal: TD-error = r + gamma * V(s') - V(s)

    Z-score normalization uses fresh EMA statistics (not checkpoint-stale buffers)
    to avoid normalization drift across checkpoints.
    """

    def __init__(self, d_model: int, hidden_dim: int = 256):
        super().__init__()
        self.input_ln = RMSNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
        )

        self.register_buffer('v_mean', torch.tensor(0.0))
        self.register_buffer('v_std', torch.tensor(1.0))
        self._ema_alpha = 0.1
        self._ema_mean = None
        self._ema_sq = None
        self._count = 0

    def forward(self, state_vector):
        if state_vector.dim() == 1:
            state_vector = state_vector.unsqueeze(0)
        # 不强制 float：RMSNorm 内部按输入 dtype 计算（bf16 嫁接模式下
        # 强制 fp32 会导致后续 Linear(bf16) 与 fp32 输入 dtype 不匹配）
        x = self.input_ln(state_vector)
        return self.net(x)

    def get_normalized_v(self, state_vector):
        v_raw = self.forward(state_vector).item()
        self._count += 1
        alpha = self._ema_alpha
        if self._count == 1:
            self._ema_mean = v_raw
            self._ema_sq = v_raw * v_raw
        else:
            self._ema_mean = (1.0 - alpha) * self._ema_mean + alpha * v_raw
            self._ema_sq = (1.0 - alpha) * self._ema_sq + alpha * (v_raw * v_raw)
        if self._count < 10:
            return v_raw
        var = max(self._ema_sq - self._ema_mean ** 2, 1e-4)
        self.v_mean.fill_(self._ema_mean)
        self.v_std.fill_(var ** 0.5)
        return (v_raw - self.v_mean.item()) / self.v_std.item()


def compute_pseudo_reward(loss_int, lm_entropy=None, lyap_estimate=None,
                          sigma_aggregate=0.5):
    """
    Synthesize pseudo-reward from internal signals when external feedback
    is not available. Bootstraps the Critic before the verification loop closes.
    """
    r = 0.0
    r += max(0, 1.0 - loss_int) * 0.3
    if lm_entropy is not None and 0.5 < lm_entropy < 3.0:
        r += 0.1
    if lyap_estimate is not None:
        r -= abs(lyap_estimate) * 0.1
    r -= sigma_aggregate * 0.2
    return r
