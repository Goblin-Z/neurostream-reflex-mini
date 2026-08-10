import torch
import math

from config.model_config import ReflexConfig


class FluidExpertRoles:
    """
    Self-organizing stable/plastic boundary.

    In SEQ and current Reflex, expert roles (stable vs plastic) are
    hardcoded at initialization via the baseline_lr spectrum.

    This module allows experts to transition between roles based on
    their long-term uncertainty patterns:

      - An expert with persistently low uncertainty → promoted to stable
        (baseline_lr decreased, becomes harder to change)
      - An expert with persistently high uncertainty → promoted to plastic
        (baseline_lr increased, becomes easier to change)

    This creates a fluid, self-organizing knowledge structure where
    the stable/plastic boundary emerges from the model's experience
    rather than being fixed at birth.

    Transition rule:
      Every `evaluation_interval` steps, for each expert:
        - Compute uncertainty_ema trend (rising? falling? stable?)
        - If ema < threshold * 0.3 for N consecutive checks → demote to stable
        - If ema > threshold * 1.5 for N consecutive checks → promote to plastic
        - Adjustment: baseline_lr *= (1 ± step_size) with clamping
    """

    def __init__(self, config: ReflexConfig):
        self.config = config
        self.lr_min = 1e-8         # absolute floor
        self.lr_max = 1e-3         # absolute ceiling
        self.adjustment_step = 0.3  # multiply/divide by this each transition
        self.stable_threshold = config.verify_threshold * 0.3
        self.plastic_threshold = config.verify_threshold * 1.5
        self.consecutive_checks_needed = 3
        self._check_counters = {}  # expert_id → {'stable_streak': n, 'plastic_streak': n}

    def evaluate(self, model, current_step):
        """
        Evaluate all experts and adjust their baseline_lr if needed.

        Should be called periodically (e.g., every 100 internal steps).
        """
        adjustments = []

        for expert in model.get_all_experts():
            eid = expert.id
            if eid not in self._check_counters:
                self._check_counters[eid] = {
                    'stable_streak': 0,
                    'plastic_streak': 0,
                }

            counter = self._check_counters[eid]
            sigma = expert.avg_uncertainty

            if sigma < self.stable_threshold:
                counter['stable_streak'] += 1
                counter['plastic_streak'] = 0
            elif sigma > self.plastic_threshold:
                counter['plastic_streak'] += 1
                counter['stable_streak'] = 0
            else:
                counter['stable_streak'] = 0
                counter['plastic_streak'] = 0

            if counter['stable_streak'] >= self.consecutive_checks_needed:
                old_lr = expert.baseline_lr
                new_lr = max(self.lr_min, old_lr * (1 - self.adjustment_step))
                if new_lr != old_lr:
                    expert.baseline_lr = new_lr
                    adjustments.append({
                        'expert_id': eid,
                        'direction': 'stable',
                        'old_lr': old_lr,
                        'new_lr': new_lr,
                        'sigma': sigma,
                    })
                counter['stable_streak'] = 0

            elif counter['plastic_streak'] >= self.consecutive_checks_needed:
                old_lr = expert.baseline_lr
                new_lr = min(self.lr_max, old_lr * (1 + self.adjustment_step))
                if new_lr != old_lr:
                    expert.baseline_lr = new_lr
                    adjustments.append({
                        'expert_id': eid,
                        'direction': 'plastic',
                        'old_lr': old_lr,
                        'new_lr': new_lr,
                        'sigma': sigma,
                    })
                counter['plastic_streak'] = 0

        return adjustments

    def get_role_distribution(self, model):
        """Classify experts into stable/intermediate/plastic by current baseline_lr."""
        stable = []
        intermediate = []
        plastic = []
        for expert in model.get_all_experts():
            lr = expert.baseline_lr
            if lr <= 1e-7:
                stable.append(expert.id)
            elif lr >= 1e-5:
                plastic.append(expert.id)
            else:
                intermediate.append(expert.id)
        return {
            'stable': stable,
            'intermediate': intermediate,
            'plastic': plastic,
        }
