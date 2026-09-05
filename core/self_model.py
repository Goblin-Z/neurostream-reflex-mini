import torch
import torch.nn as nn
import torch.nn.functional as F

from core.rmsnorm import RMSNorm


class PriorExpert(nn.Module):
    """
    Single expert in the MoE Prior: produces (mean, logvar) for z.
    Each expert learns to model a different "mode" of world dynamics.
    """
    def __init__(self, d_model: int, z_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
        )
        self.mean = nn.Linear(hidden_dim, z_dim)
        self.logvar = nn.Linear(hidden_dim, z_dim)
        nn.init.zeros_(self.logvar.weight)
        nn.init.zeros_(self.logvar.bias)

    def forward(self, h: torch.Tensor):
        feat = self.net(h)
        return self.mean(feat), self.logvar(feat)


class PosteriorExpert(nn.Module):
    """
    Single expert in the MoE Posterior: produces (mean, logvar) for z
    conditioned on (h_t, o_t).
    """
    def __init__(self, d_model: int, z_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            RMSNorm(d_model * 2),
            nn.Linear(d_model * 2, hidden_dim),
            nn.GELU(),
        )
        self.mean = nn.Linear(hidden_dim, z_dim)
        self.logvar = nn.Linear(hidden_dim, z_dim)
        nn.init.zeros_(self.logvar.weight)
        nn.init.zeros_(self.logvar.bias)

    def forward(self, h: torch.Tensor, o: torch.Tensor):
        feat = self.net(torch.cat([h, o], dim=-1))
        return self.mean(feat), self.logvar(feat)


class MoEPrior(nn.Module):
    """
    Mixture-of-Experts Prior — Router(h_t) selects which experts imagine z.

    Each expert embodies a different "thinking mode":
      - Expert A: conservative imagination (low variance)
      - Expert B: exploratory imagination (high variance)
      - Expert C: quiet/coasting (near-zero delta)
    
    Router(h_t) softly weights these modes.  The output is a moment-matched
    Gaussian (weighted mean/variance), so the downstream KL[Q||P] remains
    a standard Gaussian KL with closed form.
    """

    def __init__(self, d_model: int, z_dim: int, hidden_dim: int, n_experts: int):
        super().__init__()
        self.n_experts = n_experts
        # Router: h_t → expert weights
        self.router = nn.Linear(d_model, n_experts, bias=False)
        nn.init.normal_(self.router.weight, std=0.02)
        self.experts = nn.ModuleList([
            PriorExpert(d_model, z_dim, hidden_dim) for _ in range(n_experts)
        ])

    def forward(self, h: torch.Tensor):
        """
        h: [B, d_model]
        Returns:
            mean:   [B, z_dim]  — moment-matched mean
            logvar: [B, z_dim]  — moment-matched log-variance
            weights: [B, n_experts] — Router gating weights
        """
        weights = F.softmax(self.router(h), dim=-1)  # [B, n_experts]

        means = []
        logvars = []
        for expert in self.experts:
            m, lv = expert(h)
            means.append(m)   # each [B, z_dim]
            logvars.append(lv)

        # Weighted combination (moment matching)
        mean = sum(w.unsqueeze(-1) * m for w, m in zip(weights.unbind(dim=-1), means))
        # 修复：矩匹配补全跨组件方差项（law of total variance）——
        # var = Σ w_i·(σ_i² + μ_i²) − μ²，否则混合方差被低估导致 KL 失真
        var = sum(
            w.unsqueeze(-1) * (torch.exp(lv.clamp(SelfModel.LOGVAR_MIN, SelfModel.LOGVAR_MAX)) + m ** 2)
            for w, lv, m in zip(weights.unbind(dim=-1), logvars, means)
        ) - mean ** 2
        logvar = torch.log(var.clamp(min=1e-6, max=SelfModel.VAR_MAX))

        return mean, logvar, weights


class MoEPosterior(nn.Module):
    """
    Mixture-of-Experts Posterior — Router(h_t, o_t) selects which experts
    correct the belief after observation.

    Different experts learn different "correction strategies":
      - Expert A: conservative correction (small change from prior)
      - Expert B: aggressive correction (large change from prior)
      - Expert C: focused correction on specific dimensions of z
    """

    def __init__(self, d_model: int, z_dim: int, hidden_dim: int, n_experts: int):
        super().__init__()
        self.n_experts = n_experts
        self.router = nn.Linear(d_model * 2, n_experts, bias=False)
        nn.init.normal_(self.router.weight, std=0.02)
        self.experts = nn.ModuleList([
            PosteriorExpert(d_model, z_dim, hidden_dim) for _ in range(n_experts)
        ])

    def forward(self, h: torch.Tensor, o: torch.Tensor):
        """
        h: [B, d_model]
        o: [B, d_model]
        Returns:
            mean:   [B, z_dim]
            logvar: [B, z_dim]
            weights: [B, n_experts]
        """
        weights = F.softmax(self.router(torch.cat([h, o], dim=-1)), dim=-1)

        means = []
        logvars = []
        for expert in self.experts:
            m, lv = expert(h, o)
            means.append(m)
            logvars.append(lv)

        mean = sum(w.unsqueeze(-1) * m for w, m in zip(weights.unbind(dim=-1), means))
        # 修复：矩匹配补全跨组件方差项（law of total variance）
        var = sum(
            w.unsqueeze(-1) * (torch.exp(lv.clamp(SelfModel.LOGVAR_MIN, SelfModel.LOGVAR_MAX)) + m ** 2)
            for w, lv, m in zip(weights.unbind(dim=-1), logvars, means)
        ) - mean ** 2
        logvar = torch.log(var.clamp(min=1e-6, max=SelfModel.VAR_MAX))

        return mean, logvar, weights


class SelfModel(nn.Module):
    """
    Stateful internal world model with MoE Prior/Posterior.

    Design:
      - GRU cell: h_t = GRU(h_{t-1}, z_{t-1}, action_{t-1})
      - MoE Prior: P(z_t | h_t) = Σ w_i * N(mean_i, var_i)
        -> Router(h_t) selects which thinking modes are active
      - MoE Posterior: Q(z_t | h_t, o_t) = Σ w_i * N(mean_i, var_i)
        -> Router(h_t, o_t) selects correction strategy
      - Decoder: o_pred = f(z_t, h_t)

    KL[Q||P] is the intrinsic curiosity signal: how much did observing o_t
    change my beliefs?
    """

    LOGVAR_MIN = -6.0
    LOGVAR_MAX = 4.0
    PRIOR_VAR_MIN = 1e-4
    VAR_MAX = 55.0

    def __init__(self, d_model: int, z_dim: int = 64, hidden_dim: int = None,
                 n_prior_experts: int = 3, n_post_experts: int = 3):
        super().__init__()
        self.d_model = d_model
        self.z_dim = z_dim
        hidden = hidden_dim or d_model
        self.n_prior_experts = n_prior_experts
        self.n_post_experts = n_post_experts

        self.gru = nn.GRUCell(d_model * 2 + z_dim, d_model)

        if n_prior_experts > 1:
            self.prior = MoEPrior(d_model, z_dim, hidden, n_prior_experts)
        else:
            self.prior = PriorExpert(d_model, z_dim, hidden)

        if n_post_experts > 1:
            self.posterior = MoEPosterior(d_model, z_dim, hidden, n_post_experts)
        else:
            self.posterior = PosteriorExpert(d_model, z_dim, hidden)

        self.decoder = nn.Sequential(
            RMSNorm(d_model + z_dim),
            nn.Linear(d_model + z_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        # Small random init (NOT zero) - zero init causes u_next=0 which kills
        # all transformer layer gradients during internal forward.
        nn.init.normal_(self.decoder[-1].weight, std=0.02)
        nn.init.zeros_(self.decoder[-1].bias)

        self.init_proj = nn.Linear(d_model, d_model)

        self._last_prior_weights = None
        self._last_post_weights = None

    def forward(self, h: torch.Tensor, z: torch.Tensor, action: torch.Tensor):
        # dtype 统一：模型可能整体为 bf16（嫁接模式），外部状态可能是 fp32
        dt = next(self.parameters()).dtype
        h = h.to(dtype=dt)
        z = z.to(dtype=dt)
        action = action.to(dtype=dt)
        gru_in = torch.cat([h, z, action], dim=-1)
        h_next = self.gru(gru_in, h)

        z_prior_mean, z_prior_logvar = self._prior(h_next)
        return h_next, z_prior_mean, z_prior_logvar

    def observe_and_correct(self, h: torch.Tensor, o: torch.Tensor):
        dt = next(self.parameters()).dtype
        h = h.to(dtype=dt)
        o = o.to(dtype=dt)
        z_post_mean, z_post_logvar = self._posterior(h, o)
        return z_post_mean, z_post_logvar

    def decode(self, z: torch.Tensor, h: torch.Tensor):
        dt = next(self.parameters()).dtype
        z = z.to(dtype=dt)
        h = h.to(dtype=dt)
        return self.decoder(torch.cat([z, h], dim=-1))

    def sample_z(self, mean: torch.Tensor, logvar: torch.Tensor, temperature=1.0):
        logvar = logvar.clamp(self.LOGVAR_MIN, self.LOGVAR_MAX)
        std = torch.exp(0.5 * logvar) * temperature
        std = std.clamp(max=3.0)
        eps = torch.randn_like(std)
        return mean + eps * std

    def kl_divergence(self, post_mean, post_logvar, prior_mean, prior_logvar):
        prior_logvar = prior_logvar.clamp(self.LOGVAR_MIN, self.LOGVAR_MAX)
        post_logvar = post_logvar.clamp(self.LOGVAR_MIN, self.LOGVAR_MAX)
        prior_var = torch.exp(prior_logvar).clamp(min=self.PRIOR_VAR_MIN)
        post_var = torch.exp(post_logvar)
        kl = 0.5 * (prior_logvar - post_logvar
                    + (post_var + (post_mean - prior_mean) ** 2) / prior_var
                    - 1.0)
        return kl.sum(dim=-1).mean()

    def init_state(self, seed=None, device=None):
        # dtype 与模型参数一致（bf16 嫁接模式下不能默认 fp32）
        dt = next(self.parameters()).dtype
        dev = device if device is not None else next(self.parameters()).device
        if seed is not None:
            h = self.init_proj(seed.to(device=dev, dtype=dt))
        else:
            h = torch.zeros(1, self.d_model, device=dev, dtype=dt)
        z = torch.zeros(1, self.z_dim, device=dev, dtype=dt)
        return h, z

    def get_mode_stats(self):
        """Return Prior/Posterior expert weights for interpretability."""
        return {
            'prior_weights': self._last_prior_weights,
            'post_weights': self._last_post_weights,
        }

    def _prior(self, h):
        if isinstance(self.prior, MoEPrior):
            mean, logvar, weights = self.prior(h)
            self._last_prior_weights = weights.detach()
            return mean, logvar.clamp(self.LOGVAR_MIN, self.LOGVAR_MAX)
        mean, logvar = self.prior(h)
        return mean, logvar.clamp(self.LOGVAR_MIN, self.LOGVAR_MAX)

    def _posterior(self, h, o):
        if isinstance(self.posterior, MoEPosterior):
            mean, logvar, weights = self.posterior(h, o)
            self._last_post_weights = weights.detach()
            return mean, logvar.clamp(self.LOGVAR_MIN, self.LOGVAR_MAX)
        mean, logvar = self.posterior(h, o)
        return mean, logvar.clamp(self.LOGVAR_MIN, self.LOGVAR_MAX)
