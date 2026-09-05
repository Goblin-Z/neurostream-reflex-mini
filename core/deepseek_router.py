"""
core/deepseek_router.py — DeepSeek-V2 MoE 路由（MoEGate 语义移植）。

与官方行为一致（保证原权重前向数值等价）：
  - logits = x @ gate_weight^T（fp32），gate_weight 布局 [n_routed, d_model]
  - scores = softmax(logits)（全分布 softmax）
  - topk_method='greedy': 直接 top-k（Lite: top-6）
  - norm_topk_prob=False: top-k 权重不重新归一化（× routed_scaling_factor=1.0）
  - aux loss（DeepSeek seq_aux 公式，training 时）

Reflex 扩展（不影响原权重数值，零初始化起步）：
  - h_to_bias_weight: 内循环状态 h_t → 每专家偏置（状态门控，默认全零无影响）
  - is_internal 时 logits += 2.0（项目 Router 语义：内部前向倾向多样化）
  - 聚合 sigma（aggregate_sigma）与门控熵/利用率统计（项目接口兼容）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config.model_config import ReflexConfig


class DeepSeekRouter(nn.Module):
    def __init__(self, config: ReflexConfig, n_routed: int = None,
                 top_k: int = None):
        super().__init__()
        self.d_model = config.d_model
        self.n_routed = n_routed or getattr(config, 'n_routed_experts', 64)
        self.top_k = top_k or getattr(config, 'num_experts_per_tok', 6)
        self.norm_topk_prob = getattr(config, 'norm_topk_prob', False)
        self.routed_scaling_factor = getattr(config, 'routed_scaling_factor', 1.0)
        self.alpha = getattr(config, 'aux_loss_alpha', 0.001)
        self.seq_aux = getattr(config, 'seq_aux', True)
        self.scoring_func = getattr(config, 'scoring_func', 'softmax')
        self.topk_method = getattr(config, 'topk_method', 'greedy')
        self.n_group = getattr(config, 'n_group', 1)
        self.topk_group = getattr(config, 'topk_group', 1)

        # DeepSeek 布局: [n_routed, d_model]（加载直拷）
        self.gate_weight = nn.Parameter(
            torch.empty((self.n_routed, self.d_model)))
        nn.init.kaiming_uniform_(self.gate_weight, a=5 ** 0.5)

        # 状态门控（Reflex 扩展，零初始化 → 不影响原权重前向）
        self.h_to_bias_weight = nn.Parameter(
            torch.zeros(self.d_model, self.n_routed))

        # 门控熵/利用率统计（Reflex 接口兼容）
        self.register_buffer('activation_running_mean', torch.zeros(self.n_routed))
        self.register_buffer('activation_running_var', torch.ones(self.n_routed))
        self.ema_momentum = 0.99
        self.register_buffer('expert_util_ema', torch.ones(self.n_routed) / self.n_routed)
        self._last_gating_entropy = torch.tensor(0.0)
        self._last_aux_loss = None
        self._util_momentum = 0.99

    def forward(self, x, bsz=None, seq_len=None, h_state=None, is_internal=False):
        """
        x: [B*T, d_model]
        bsz/seq_len: 用于 seq_aux 公式（官方按序列统计；None 时退化为全局公式）
        h_state: [1, d_model]（状态门控，零初始化默认无影响）
        返回: (top_w [B*T, top_k], top_idx [B*T, top_k], logits [B*T, n_routed])
        """
        # 官方: fp32 计算门控分数
        logits = F.linear(x.float(), self.gate_weight.float(), None)
        if is_internal:
            logits = logits + 2.0
        if h_state is not None:
            h = h_state
            if h.dim() == 1:
                h = h.unsqueeze(0)
            # 统一 fp32 域：h（内循环状态，可能 fp32/bf16）与
            # h_to_bias_weight 都转 fp32 参与——混合 dtype（bf16 参数
            # × fp32 h）在 CE 在线训练 backward 时抛
            # "Found dtype Float but expected BFloat16"（v3.18 修复）
            logits = logits + h.to(logits.dtype) @ self.h_to_bias_weight.float()

        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'scoring_func={self.scoring_func}')

        # top-k 选择（greedy；Lite: n_group=1 无分组限制）
        if self.topk_method == 'greedy':
            topk_weight, topk_idx = torch.topk(scores, self.top_k, dim=-1, sorted=False)
        elif self.topk_method == 'group_limited_greedy':
            group_scores = scores.view(-1, self.n_group,
                                       self.n_routed // self.n_group).max(dim=-1).values
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1)
            score_mask = (group_mask.unsqueeze(-1)
                          .expand(-1, self.n_group, self.n_routed // self.n_group)
                          .reshape(-1, self.n_routed))
            tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)
            topk_weight, topk_idx = torch.topk(tmp_scores, self.top_k, dim=-1, sorted=False)
        else:
            raise NotImplementedError(f'topk_method={self.topk_method}')

        # norm_topk_prob=False（Lite）: 不重新归一化，仅乘 scaling factor
        if not self.norm_topk_prob:
            topk_weight = topk_weight * self.routed_scaling_factor

        # ── aux loss（DeepSeek 公式；training 时）──
        if self.training and self.alpha > 0.0:
            aux_topk = self.top_k
            topk_idx_flat = topk_idx.view(bsz, -1) if bsz is not None \
                else topk_idx.view(-1, self.top_k)
            if self.seq_aux and bsz is not None and seq_len is not None:
                # 官方 seq_aux：按序列统计后跨序列平均
                scores_for_seq_aux = scores.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed, device=scores.device)
                ce.scatter_add_(
                    1, topk_idx_flat,
                    torch.ones(bsz, seq_len * aux_topk, device=scores.device),
                ).div_(seq_len * aux_topk / self.n_routed)
                self._last_aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(
                    dim=1).mean() * self.alpha
            else:
                # 全局公式（bsz/seq 未知时退化；数值为近似，不影响前向）
                mask_ce = F.one_hot(topk_idx_flat.view(-1),
                                    num_classes=self.n_routed)
                ce = mask_ce.float().mean(0)
                pi = scores.mean(0)
                self._last_aux_loss = (ce * pi).sum() * self.n_routed * self.alpha
        else:
            self._last_aux_loss = None

        # ── 利用率/熵统计（Reflex 接口）──
        if self.training:
            util = torch.zeros(self.n_routed, device=scores.device)
            n = topk_idx.numel()
            if n > 0:
                util.scatter_add_(0, topk_idx.view(-1),
                                  torch.ones(n, device=scores.device))
                util = util / (n / self.top_k)
            self.expert_util_ema.mul_(self._util_momentum).add_(
                util, alpha=1.0 - self._util_momentum)
            self._last_gating_entropy = -(
                scores * (scores.clamp(min=1e-8).log())).sum(dim=-1).mean().detach()

        return topk_weight, topk_idx, logits

    # ── Reflex 接口兼容 ──

    def aggregate_sigma(self, expert_sigmas, top_k_weights, top_k_indices):
        """top-k 加权 sigma（expert_sigmas 按专家索引索引）。"""
        n_active = self.n_routed
        safe_indices = top_k_indices.clamp(0, n_active - 1)
        selected = expert_sigmas[safe_indices]
        return (top_k_weights * selected).sum(dim=-1).mean().item()

    def get_gating_stats(self):
        return {
            'entropy': self._last_gating_entropy,
            'utilization': self.expert_util_ema.clone(),
        }

    def should_verify(self, sigma_aggregate, steps_since_last):
        return sigma_aggregate > self.verify_threshold.item() if hasattr(
            self, 'verify_threshold') else False, sigma_aggregate
