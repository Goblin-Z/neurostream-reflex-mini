import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Faster than LayerNorm (no mean subtraction, no bias).
    Equivalent to LayerNorm without mean-centering.

    x_norm = x / sqrt(mean(x^2) + eps) * weight
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * norm * self.weight

    def extra_repr(self):
        return f'{self.weight.size(0)}, eps={self.eps}'
