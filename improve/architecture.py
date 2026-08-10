import torch
import torch.nn as nn
import math

from config.model_config import ReflexConfig
from core.model import ReflexMoELayer


class ArchitectureModifier:
    """
    Runtime architecture self-modification — v2: non-procedural triggers.

    v1 used hardcoded if/else on external metrics (reward_ema, loss_window).
    v2 uses model-internal signals from the Router:

        gating_entropy:      H[gating | x, h_t]
            High → Router is uncertain about which expert to use.
            Persistently high → the current expert set doesn't differentiate
            the Router's input states → need more experts or a split.

        expert_utilization:  how often each expert is selected in top-k.
            Near-zero → the expert is rarely used → should be replaced.
            Very high → the expert is overloaded → should be split.

        mean_entropy_across_layers:
            All low → the model is "too certain" everywhere → may need
            new capacity for new thinking patterns (add layer).

    All thresholds are either static hyperparameters or learnable parameters
    on the Router itself — no external metrics, no loss curve monitoring.
    """

    def __init__(self, config: ReflexConfig):
        self.config = config
        self._cooldown = 0
        self._stats = {'replace': 0, 'split': 0, 'add_layer': 0}

        # Entropy/utilization thresholds
        self.entropy_high = 1.5       # above this → uncertain → consider split
        self.entropy_low = 0.3        # below this → too certain → consider add
        self.util_low = 0.02           # below this → unused → consider replace
        self.util_prune = 0.005        # below this → unused for very long → prune
        self.util_high = 0.4           # above this → overloaded → consider split

        # Consecutive checks needed before action
        self._entropy_high_streak = 0
        self._entropy_low_streak = 0
        self._replace_candidates = {}  # expert_id → consecutive low-util count
        self._prune_candidates = {}    # expert_id → consecutive ultra-low-util count

        self._prev_avg_entropy = None

    def step(self, model, gating_stats_per_layer=None):
        """
        Evaluate architecture needs from Router-internal signals.

        Args:
            model: ReflexModel instance
            gating_stats_per_layer: list of dicts, one per layer, each with:
                - 'entropy': avg gating entropy (tensor scalar)
                - 'utilization': expert_util_ema (tensor [n_experts])

        Returns:
            None if no change, or one of:
                'replace' - expert replaced (router dims unchanged)
                'split'   - expert added (router dims changed)
                'prune'   - expert removed (router dims changed)
                'add_layer' - new layer appended (layer count changed)
        """
        if self._cooldown > 0:
            self._cooldown -= 1
            return None
        if not getattr(self.config, 'arch_self_mod_enabled', False):
            return None
        if gating_stats_per_layer is None or len(gating_stats_per_layer) == 0:
            return None

        # Compute mean entropy across all layers
        # (defensive: accept float or 0-dim tensor from get_gating_stats)
        avg_entropy = sum(float(s['entropy']) for s in gating_stats_per_layer)
        avg_entropy /= len(gating_stats_per_layer)

        # Track entropy streaks
        if avg_entropy > self.entropy_high:
            self._entropy_high_streak += 1
            self._entropy_low_streak = 0
        elif avg_entropy < self.entropy_low:
            self._entropy_low_streak += 1
            self._entropy_high_streak = 0
        else:
            self._entropy_high_streak = 0
            self._entropy_low_streak = 0

        # ── Decision 1: Expert replacement ──
        # Trigger: an expert has near-zero utilization persistently.
        replaced = self._check_replace(model, gating_stats_per_layer)
        if replaced:
            self._cooldown = 200
            return replaced  # 'replace' or 'prune'

        # ── Decision 2: Expert split ──
        # Trigger: a layer has persistently high entropy AND one expert is overloaded.
        split = self._check_split(model, gating_stats_per_layer, avg_entropy)
        if split:
            self._cooldown = 200
            return 'split'

        # ── Decision 3: Add layer ──
        # Trigger: ALL layers have low entropy -> model is uniformly too certain.
        if self._entropy_low_streak >= 20:  # 20 consecutive checks = ~4000 steps
            self._add_layer(model)
            self._cooldown = 500
            self._stats['add_layer'] += 1
            self._entropy_low_streak = 0
            return 'add_layer'

        self._prev_avg_entropy = avg_entropy
        return None

    def _check_replace(self, model, gating_stats_per_layer):
        """Replace or prune experts based on utilization.
        
        Three tiers:
          util < util_prune × 5+ checks → prune (remove entirely)
          util < util_low × 5+ checks   → replace (re-initialise)
          util >= util_low              → healthy, reset counter
        """
        for layer_idx, stats in enumerate(gating_stats_per_layer):
            util = stats['utilization']
            for i in range(len(util)):
                if i >= len(model.layers[layer_idx].all_experts):
                    continue
                expert = model.layers[layer_idx].all_experts[i]
                eid = expert.id
                util_val = util[i].item()

                if util_val < self.util_prune:
                    self._prune_candidates[eid] = \
                        self._prune_candidates.get(eid, 0) + 1
                    self._replace_candidates.pop(eid, None)
                    if self._prune_candidates[eid] >= 5:
                        if len(model.layers[layer_idx].all_experts) > 2:
                            self._prune_expert(model, layer_idx, i)
                            self._stats['replace'] += 1  # counts as replacement
                            self._prune_candidates.clear()
                            self._replace_candidates.clear()
                            return 'prune'
                elif util_val < self.util_low:
                    self._replace_candidates[eid] = \
                        self._replace_candidates.get(eid, 0) + 1
                    self._prune_candidates.pop(eid, None)
                    if self._replace_candidates[eid] >= 5:
                        self._replace_expert(model, layer_idx, i)
                        self._stats['replace'] += 1
                        self._replace_candidates.clear()
                        self._prune_candidates.clear()
                        return 'replace'
                else:
                    self._replace_candidates.pop(eid, None)
                    self._prune_candidates.pop(eid, None)
        return None

    def _check_split(self, model, gating_stats_per_layer, avg_entropy):
        """Split overloaded experts in high-entropy layers."""
        if self._entropy_high_streak < 5:  # need 5 consecutive high-entropy checks
            return False
        for layer_idx, stats in enumerate(gating_stats_per_layer):
            util = stats['utilization']
            for i in range(len(util)):
                if i < len(model.layers[layer_idx].all_experts):
                    util_val = util[i].item()
                    if util_val > self.util_high:
                        self._split_expert(model, layer_idx, i)
                        self._stats['split'] += 1
                        self._entropy_high_streak = 0
                        return True
        return False

    def _prune_expert(self, model, layer_idx, expert_idx):
        """
        Remove an expert entirely - it has near-zero utilization for many steps.
        Before removing, its baseline_lr was already effectively zero, so no
        knowledge is lost.  The Router's columns shrink accordingly.
        """
        model.layers[layer_idx].remove_expert_by_idx(expert_idx)

    def _replace_expert(self, model, layer_idx, expert_idx):
        """Copy the most-used expert's weights into the unused expert, add noise."""
        layer = model.layers[layer_idx]
        experts = layer.all_experts
        if len(experts) < 2:
            return
        util = model.layers[layer_idx].router.expert_util_ema
        best_idx = util.argmax().item()
        if best_idx >= len(experts):
            best_idx = 0
        worst = experts[expert_idx]
        best = experts[best_idx]
        with torch.no_grad():
            for pw, pb in zip(worst.parameters(), best.parameters()):
                pw.data = pb.data.clone() + 0.02 * torch.randn_like(pb.data)
        if hasattr(worst, 'uncertainty_ema'):
            worst.uncertainty_ema.fill_(0.5)

    def _split_expert(self, model, layer_idx, expert_idx):
        """Clone overloaded expert with doubled baseline_lr."""
        layer = model.layers[layer_idx]
        expert = layer.all_experts[expert_idx]
        new_e = layer.add_expert(baseline_lr=expert.baseline_lr * 2)
        with torch.no_grad():
            for pn, ps in zip(new_e.parameters(), expert.parameters()):
                pn.data = ps.data.clone() + 0.02 * torch.randn_like(ps.data)

    def _add_layer(self, model):
        """Copy last layer with small noise perturbation.

        C2 fix: 新层专家数对齐源层（源层可能被 split/prune 过，
        与 config 默认不一致）——否则各层 _learnable_sigmas 长度
        不齐（虽已不再 stack，但语义上应保持一致）。
        """
        last = model.layers[-1]
        new_layer = ReflexMoELayer(self.config)
        new_layer.attention.load_state_dict(last.attention.state_dict())
        new_layer.router.load_state_dict(last.router.state_dict())
        # 对齐专家数：移除新层多余专家（或补齐不足——从源层复制）
        n_src = len(last.all_experts)
        while len(new_layer.all_experts) > n_src:
            new_layer.remove_expert_by_idx(len(new_layer.all_experts) - 1)
        while len(new_layer.all_experts) < n_src:
            new_layer.add_expert(baseline_lr=1e-5)
        # 按索引复制专家权重（当前层数已与源层一致）
        for i in range(n_src):
            src = last.all_experts[i]
            new_layer.all_experts[i].load_state_dict(src.state_dict())
            for p in new_layer.all_experts[i].parameters():
                p.data += 0.01 * torch.randn_like(p.data)
        model.layers.append(new_layer)

    def get_stats(self):
        return dict(self._stats)
