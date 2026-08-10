import torch
import torch.nn.functional as F


def _get_stable_param_names(model):
    """
    Get parameter names for stable experts.

    PyTorch's named_parameters() deduplicates shared modules, so
    stable_experts (which are views into all_experts) never appear
    under 'stable_experts.' names.  Instead, we identify stable
    experts by object identity and find their names under
    'all_experts.'.
    """
    stable_ids = set(id(e) for e in model.get_stable_experts())
    names = set()
    for name, param in model.named_parameters():
        # name looks like 'layers.0.all_experts.3.w_gate.weight'
        parts = name.split('.')
        if len(parts) >= 4 and parts[2] == 'all_experts':
            try:
                ei = int(parts[3])
                li = int(parts[1])
                expert = model.layers[li].all_experts[ei]
                if id(expert) in stable_ids:
                    names.add(name)
            except (IndexError, ValueError):
                pass
    return names


def estimate_fisher(model, embeddings, num_samples=10):
    """Estimate Fisher information diagonal for stable expert params."""
    stable_names = _get_stable_param_names(model)
    fisher = {name: torch.zeros_like(param)
              for name, param in model.named_parameters()
              if name in stable_names}

    model.eval()
    for _ in range(num_samples):
        model.zero_grad()
        logits = model.forward_embeddings(embeddings)
        log_probs = F.log_softmax(logits, dim=-1)
        sampled = torch.multinomial(
            torch.exp(log_probs.view(-1, log_probs.size(-1))), 1
        ).view(logits.size(0), logits.size(1))
        loss = F.nll_loss(
            log_probs.view(-1, log_probs.size(-1)),
            sampled.view(-1),
            reduction='mean',
        )
        loss.backward()
        for name, param in model.named_parameters():
            if name in fisher and param.grad is not None:
                fisher[name] += param.grad.data ** 2 / num_samples
    return fisher


def ewc_penalty(named_params, fisher, old_params, lambda_ewc=100.0, max_penalty=100.0):
    """
    Elastic Weight Consolidation penalty.

    The penalty is clamped to max_penalty to prevent it from blocking
    all learning when Fisher diagonals or parameter drift are large
    (e.g. right after a soft-reset of plastic experts).
    """
    penalty = 0.0
    for name, param in named_params:
        if name in fisher and name in old_params:
            diff = param - old_params[name]
            penalty = penalty + (fisher[name] * diff ** 2).sum()
    penalty = lambda_ewc * penalty
    if isinstance(penalty, torch.Tensor):
        penalty = penalty.clamp(max=max_penalty)
    return penalty


def get_stable_named_params(model):
    """Return list of (name, param) for stable expert parameters."""
    stable_names = _get_stable_param_names(model)
    return [(name, param) for name, param in model.named_parameters()
            if name in stable_names]
