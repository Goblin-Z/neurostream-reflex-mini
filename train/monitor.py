"""
Gradient and stability monitor for NeuroStream-Reflex training.

Monitors:
  1. Gradient norms per layer and per expert (detect explosion/vanishing)
  2. Expert activation frequency (detect dead experts)
  3. Sigma variance (detect collapse to fixed point)
  4. Weight growth rate (detect runaway parameters)

Auto-recovery actions:
  - Reset dead experts (re-init weights)
  - Scale down learning rate on gradient explosion
  - Inject noise on sigma collapse
  - Log all events for analysis
"""
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class MonitorConfig:
    # Gradient monitoring
    grad_norm_window: int = 100          # moving average window
    grad_explosion_threshold: float = 10.0  # trigger if norm > this
    grad_vanishing_threshold: float = 1e-6  # trigger if norm < this
    grad_explosion_patience: int = 3       # consecutive explosions before action
    grad_vanishing_patience: int = 10      # consecutive vanishings before action

    # Expert monitoring
    expert_activation_window: int = 500    # how many steps to track
    dead_expert_threshold: float = 0.01    # activation rate below this = dead
    dead_expert_patience: int = 2000       # steps before resetting a dead expert

    # Sigma monitoring
    sigma_window: int = 200                # moving average window
    sigma_collapse_threshold: float = 0.01 # variance below this = collapse
    sigma_collapse_patience: int = 500     # steps before injecting noise

    # Weight monitoring
    weight_growth_threshold: float = 10.0  # max allowed weight magnitude
    weight_growth_patience: int = 1000

    # Recovery
    lr_scale_on_explosion: float = 0.5     # multiply LR by this on explosion
    noise_inject_on_collapse: float = 0.05 # noise magnitude on sigma collapse
    max_resets_per_expert: int = 3         # max times to reset an expert


class GradientMonitor:
    """
    Per-layer and per-expert gradient norm tracking.
    Detects explosion (>10.0) and vanishing (<1e-6).
    """

    def __init__(self, model, config: MonitorConfig = None):
        self.model = model
        self.cfg = config or MonitorConfig()
        self.layer_norms = {}        # layer_idx -> deque of norms
        self.expert_norms = {}       # (layer_idx, expert_idx) -> deque of norms
        self.explosion_count = 0
        self.vanishing_count = 0
        self.last_action_step = 0

    def step(self, loss, global_step: int):
        """Record gradient norms for all layers and experts."""
        norms = self._compute_grad_norms()
        for key, norm in norms.items():
            if key not in self.layer_norms:
                self.layer_norms[key] = deque(maxlen=self.cfg.grad_norm_window)
            self.layer_norms[key].append(norm)

        # Check for explosion
        if norms.get('total', 0) > self.cfg.grad_explosion_threshold:
            self.explosion_count += 1
        else:
            self.explosion_count = 0

        # Check for vanishing
        if norms.get('total', 0) < self.cfg.grad_vanishing_threshold:
            self.vanishing_count += 1
        else:
            self.vanishing_count = 0

        return norms

    def _compute_grad_norms(self) -> Dict[str, float]:
        norms = {}
        total_sq = 0.0
        for name, p in self.model.named_parameters():
            if p.grad is not None:
                n = p.grad.norm().item()
                total_sq += n * n
                if 'layers.' in name:
                    parts = name.split('.')
                    try:
                        li = int(parts[1])
                        key = f'layer_{li}'
                        norms[key] = norms.get(key, 0.0) + n * n
                    except (IndexError, ValueError):
                        pass
        norms['total'] = total_sq ** 0.5
        return norms

    def is_exploding(self) -> bool:
        return self.explosion_count >= self.cfg.grad_explosion_patience

    def is_vanishing(self) -> bool:
        return self.vanishing_count >= self.cfg.grad_vanishing_patience

    def get_layer_norms(self) -> Dict[str, float]:
        return {k: np.mean(v) for k, v in self.layer_norms.items()}


class ExpertActivityMonitor:
    """
    Tracks which experts are activated and how often.
    Detects dead experts (activation rate < 1% over 500 steps).
    """

    def __init__(self, model, config: MonitorConfig = None):
        self.model = model
        self.cfg = config or MonitorConfig()
        self.activation_counts = {}   # (layer_idx, expert_idx) -> deque of bools
        self.reset_counts = {}        # (layer_idx, expert_idx) -> int
        self.dead_experts = set()

    def step(self, global_step: int):
        """Record which experts were activated in this step."""
        for li, layer in enumerate(self.model.layers):
            for ei, expert in enumerate(layer.all_experts):
                key = (li, ei)
                if key not in self.activation_counts:
                    self.activation_counts[key] = deque(
                        maxlen=self.cfg.expert_activation_window
                    )
                # Expert is "active" if it has output (was routed to in THIS step)
                active = expert._output is not None
                self.activation_counts[key].append(active)
                # Reset for next step so stale activations don't persist.
                # Safe during training (Hebbian uses _output only in deployment,
                # where the monitor is not running).
                expert._output = None

    def get_dead_experts(self, global_step: int) -> List[Tuple[int, int]]:
        """Return list of (layer_idx, expert_idx) that are dead."""
        dead = []
        for key, history in self.activation_counts.items():
            if len(history) < self.cfg.expert_activation_window:
                continue
            rate = sum(history) / len(history)
            if rate < self.cfg.dead_expert_threshold:
                resets = self.reset_counts.get(key, 0)
                if resets < self.cfg.max_resets_per_expert:
                    dead.append(key)
        return dead

    def mark_reset(self, layer_idx: int, expert_idx: int):
        key = (layer_idx, expert_idx)
        self.reset_counts[key] = self.reset_counts.get(key, 0) + 1
        self.activation_counts[key].clear()

    def get_activation_rates(self) -> Dict[Tuple[int, int], float]:
        return {
            k: sum(v) / len(v) if v else 0.0
            for k, v in self.activation_counts.items()
        }


class SigmaMonitor:
    """
    Tracks sigma variance over time.
    Detects collapse (variance < 0.01 over 200 steps).
    """

    def __init__(self, config: MonitorConfig = None):
        self.cfg = config or MonitorConfig()
        self.sigma_history = deque(maxlen=self.cfg.sigma_window)
        self.collapse_count = 0

    def step(self, sigma: float):
        self.sigma_history.append(sigma)

    def is_collapsed(self) -> bool:
        if len(self.sigma_history) < self.cfg.sigma_window:
            return False
        var = np.var(self.sigma_history)
        if var < self.cfg.sigma_collapse_threshold:
            self.collapse_count += 1
            return self.collapse_count >= self.cfg.sigma_collapse_patience
        else:
            self.collapse_count = 0
            return False

    def get_variance(self) -> float:
        if len(self.sigma_history) < 2:
            return 0.0
        return np.var(self.sigma_history)

    def get_mean(self) -> float:
        if not self.sigma_history:
            return 0.5
        return np.mean(self.sigma_history)


class WeightMonitor:
    """
    Tracks weight magnitudes to detect runaway parameters.
    """

    def __init__(self, model, config: MonitorConfig = None):
        self.model = model
        self.cfg = config or MonitorConfig()
        self.max_magnitudes = {}  # name -> deque of max abs values
        self.growth_count = 0

    def step(self, global_step: int):
        for name, p in self.model.named_parameters():
            if p.dim() >= 2:
                max_abs = p.data.abs().max().item()
                if name not in self.max_magnitudes:
                    self.max_magnitudes[name] = deque(maxlen=100)
                self.max_magnitudes[name].append(max_abs)

    def get_runaway_params(self) -> List[str]:
        runaway = []
        for name, history in self.max_magnitudes.items():
            if len(history) < 50:
                continue
            if max(history) > self.cfg.weight_growth_threshold:
                runaway.append(name)
        return runaway


class ReflexMonitor:
    """
    Unified monitor for NeuroStream-Reflex training.
    Integrates gradient, expert, sigma, and weight monitoring.
    Provides auto-recovery actions.
    """

    def __init__(self, model, config: MonitorConfig = None):
        self.model = model
        self.cfg = config or MonitorConfig()
        self.gradient = GradientMonitor(model, config)
        self.expert_activity = ExpertActivityMonitor(model, config)
        self.sigma_mon = SigmaMonitor(config)
        self.weight_mon = WeightMonitor(model, config)
        self.events = []  # list of (step, type, message)

    def step(self, loss, sigma: float, global_step: int):
        """Run all monitors and return actions to take."""
        self.sigma_mon.step(sigma)
        self.weight_mon.step(global_step)

        # Gradient norms (loss is a plain tensor from chunked_loss;
        # gradient monitoring uses param.grad directly, not loss.requires_grad)
        grad_norms = self.gradient.step(loss, global_step)

        # Expert activity
        self.expert_activity.step(global_step)

        actions = []

        # Check gradient explosion
        if self.gradient.is_exploding():
            actions.append({
                'type': 'grad_explosion',
                'lr_scale': self.cfg.lr_scale_on_explosion,
                'total_norm': grad_norms.get('total', 0),
            })
            self._log_event(global_step, 'GRAD_EXPLOSION',
                            f"norm={grad_norms.get('total', 0):.2f}")

        # Check gradient vanishing
        if self.gradient.is_vanishing():
            actions.append({
                'type': 'grad_vanishing',
                'total_norm': grad_norms.get('total', 0),
            })
            self._log_event(global_step, 'GRAD_VANISHING',
                            f"norm={grad_norms.get('total', 0):.2e}")

        # Check dead experts
        dead = self.expert_activity.get_dead_experts(global_step)
        for li, ei in dead:
            actions.append({
                'type': 'reset_expert',
                'layer_idx': li,
                'expert_idx': ei,
            })
            self._log_event(global_step, 'DEAD_EXPERT',
                            f"L{li}E{ei} — resetting")
            self.expert_activity.mark_reset(li, ei)

        # Check sigma collapse
        if self.sigma_mon.is_collapsed():
            actions.append({
                'type': 'sigma_collapse',
                'noise': self.cfg.noise_inject_on_collapse,
                'variance': self.sigma_mon.get_variance(),
            })
            self._log_event(global_step, 'SIGMA_COLLAPSE',
                            f"var={self.sigma_mon.get_variance():.4f}")

        # Check runaway weights
        runaway = self.weight_mon.get_runaway_params()
        if runaway:
            actions.append({
                'type': 'runaway_weights',
                'params': runaway[:5],
            })
            self._log_event(global_step, 'RUNAWAY_W',
                            f"{runaway[:3]}")

        return actions, grad_norms

    def _log_event(self, step, etype, message):
        self.events.append((step, etype, message))
        if len(self.events) > 1000:
            self.events = self.events[-500:]

    def get_summary(self) -> Dict:
        """Return a summary of current monitoring state."""
        grad_norms = self.gradient.get_layer_norms()
        act_rates = self.expert_activity.get_activation_rates()
        sigma_var = self.sigma_mon.get_variance()
        sigma_mean = self.sigma_mon.get_mean()

        # Per-layer stats
        layer_stats = {}
        for li in range(len(self.model.layers)):
            rates = [act_rates.get((li, ei), 0) for ei in range(len(self.model.layers[li].all_experts))]
            layer_stats[f'L{li}'] = {
                'grad_norm': grad_norms.get(f'layer_{li}', 0),
                'avg_activation': np.mean(rates) if rates else 0,
                'dead': sum(1 for r in rates if r < self.cfg.dead_expert_threshold),
            }

        return {
            'sigma': {'mean': sigma_mean, 'variance': sigma_var},
            'gradient': {
                'total_norm': grad_norms.get('total', 0),
                'exploding': self.gradient.is_exploding(),
                'vanishing': self.gradient.is_vanishing(),
            },
            'experts': layer_stats,
            'events_last_100': len([e for e in self.events if e[0] > max(0, self.events[-1][0] - 100)]) if self.events else 0,
        }

    def apply_actions(self, actions, optimizer=None):
        """Apply recovery actions. Returns modified learning rate."""
        new_lr = None
        for action in actions:
            if action['type'] == 'grad_explosion' and optimizer is not None:
                for pg in optimizer.param_groups:
                    pg['lr'] *= action['lr_scale']
                new_lr = optimizer.param_groups[0]['lr']

            elif action['type'] == 'reset_expert':
                li, ei = action['layer_idx'], action['expert_idx']
                expert = self.model.layers[li].all_experts[ei]
                expert.reset_weights()
                print(f"  [monitor] Reset L{li}E{ei} (dead expert)")

            elif action['type'] == 'sigma_collapse':
                # Inject noise into endosphere with high sigma to trigger exploration
                noise = torch.randn(1, self.model.config.d_model) * action['noise']
                self.model.endosphere.push(noise.squeeze(0), sigma=0.8)
                print(f"  [monitor] Injected noise (sigma collapsed, var={action['variance']:.4f})")

        return new_lr
