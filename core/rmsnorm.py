import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Faster than LayerNorm (no mean subtraction, no bias).
    Equivalent to LayerNorm without mean-centering.

    x_norm = x / sqrt(mean(x^2) + eps) * weight

    NOTE: 统计量在 fp32 计算（与 Llama/Qwen3_5RMSNorm 一致）——
    bf16 直接累加在 64 层深网上会引入可观的累积误差，
    嫁接数值验证（verify_graft.py）依赖此精度对齐。
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        input_dtype = x.dtype
        x_f = x.to(torch.float32)
        norm = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_f * norm * self.weight).to(input_dtype)

    def extra_repr(self):
        return f'{self.weight.size(0)}, eps={self.eps}'
