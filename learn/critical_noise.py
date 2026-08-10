import torch


class CriticalNoiseScheduler:
    """
    Self-regulated noise for edge-of-chaos dynamics.

    PI controller with anti-windup and midpoint regression:

        error = target_sigma - current_sigma
        adjustment = Kp * error + Ki * integral(error)

    Anti-windup: integral is clamped and frozen when output saturates.
    Midpoint regression: when error ≈ 0, noise slowly decays toward
    the midpoint instead of staying stuck at an extreme.

    Inspired by self-organized criticality (Bak, Tang, Wiesenfeld 1987)
    and the hypothesis that consciousness emerges at the critical
    boundary between order and disorder (Chialvo, 2010).
    """

    def __init__(self, target_sigma=0.5, noise_min=0.0001, noise_max=0.05,
                 kp=0.02, ki=0.005, momentum=0.5):
        self.target_sigma = target_sigma
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.kp = kp                    # proportional gain (doubled for faster response)
        self.ki = ki                    # integral gain
        self.momentum = momentum        # EMA momentum (reduced from 0.8 for faster response)
        self._noise_level = (noise_min + noise_max) / 2
        self._noise_ema = self._noise_level
        self._integral = 0.0            # accumulated error
        self._integral_clamp = 0.02     # anti-windup (reduced from 0.03)
        self._midpoint = (noise_min + noise_max) / 2
        self._decay_rate = 0.002        # midpoint regression rate

    def get_noise(self, current_sigma=None):
        """
        Compute noise for this step.

        Args:
            current_sigma: the model's current uncertainty aggregate.
                           If None, returns current noise without adjustment.

        Returns:
            noise scalar to be added to state: v_t + noise * randn_like(v_t)
        """
        if current_sigma is None:
            return self._noise_level

        # PI feedback: push noise to counteract sigma deviation
        error = self.target_sigma - current_sigma

        # Anti-windup: only integrate when NOT saturated in the
        # direction that would make things worse.
        # If noise is at max and we want to decrease it (error < 0),
        # don't accumulate more positive integral.
        # If noise is at min and we want to increase it (error > 0),
        # don't accumulate more negative integral.
        at_max = self._noise_level >= self.noise_max - 1e-6
        at_min = self._noise_level <= self.noise_min + 1e-6
        freeze_integral = (at_max and error < 0) or (at_min and error > 0)

        if not freeze_integral:
            self._integral = max(-self._integral_clamp,
                                 min(self._integral_clamp,
                                     self._integral + error))

        adjustment = self.kp * error + self.ki * self._integral

        # Midpoint regression: when error is small, pull noise toward
        # the midpoint so it doesn't get stuck at an extreme.
        # This is the key fix for the "noise stuck at 0.05" bug.
        midpoint_pull = self._decay_rate * (self._midpoint - self._noise_level)

        # Smooth update with EMA
        self._noise_ema = (
            self.momentum * self._noise_ema
            + (1 - self.momentum) * (self._noise_level + adjustment + midpoint_pull)
        )
        self._noise_level = max(self.noise_min, min(self.noise_max,
                                                     self._noise_ema))
        return self._noise_level

    @property
    def current_noise(self):
        return self._noise_level

    @property
    def is_at_edge(self):
        """True when noise is in the middle 40% of its range."""
        rng = self.noise_max - self.noise_min
        mid_low = self.noise_min + 0.3 * rng
        mid_high = self.noise_min + 0.7 * rng
        return mid_low < self._noise_level < mid_high
