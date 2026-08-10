import torch
import torch.nn.functional as F

from config.model_config import ReflexConfig


class ReflexRewardModel:
    """
    Multi-dimensional reward model for self-play quality filtering.

    Combines: self-consistency, perplexity, diversity, and length.
    """

    def __init__(self, config: ReflexConfig):
        self.w_consistency = getattr(config, 'reward_w_consistency', 0.4)
        self.w_perplexity = getattr(config, 'reward_w_perplexity', 0.3)
        self.w_diversity = getattr(config, 'reward_w_diversity', 0.2)
        self.w_length = getattr(config, 'reward_w_length', 0.1)
        self.min_len = getattr(config, 'reward_min_len', 10)
        self.max_len = getattr(config, 'reward_max_len', 200)

    def compute(self, logits=None, labels=None, response_text=None,
                response_len=None):
        r_total = 0.0
        details = {}

        if logits is not None and labels is not None:
            r_ppl = self._perplexity_reward(logits, labels)
            r_total += self.w_perplexity * r_ppl
            details['perplexity'] = r_ppl

        if response_text:
            r_div = self._diversity_reward(response_text)
            r_total += self.w_diversity * r_div
            details['diversity'] = r_div

        if response_len is not None:
            r_len = self._length_reward(response_len)
            r_total += self.w_length * r_len
            details['length'] = r_len

        details['total'] = r_total
        return r_total, details

    def compute_consistency(self, model, q_ids, a_ids, a_text):
        """Quick consistency-only reward for self-play filtering."""
        if a_ids is None or a_ids.size(-1) < 2:
            return 0.0
        device = next(model.parameters()).device
        full = torch.cat([q_ids.to(device), a_ids.to(device)], dim=-1)
        with torch.no_grad():
            logits = model(full)
        shift = full[:, 1:]
        loss = F.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
            shift.contiguous().view(-1),
        )
        return torch.exp(-loss).item()

    def _perplexity_reward(self, logits, labels):
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                               labels.view(-1))
        ppl = torch.exp(loss).item()
        return 1.0 / (1.0 + 0.1 * ppl)

    def _diversity_reward(self, text, n=3):
        tokens = text.split()
        if len(tokens) < n + 1:
            return 0.5
        ngrams = set()
        total = 0
        for i in range(len(tokens) - n + 1):
            ngrams.add(tuple(tokens[i:i + n]))
            total += 1
        return len(ngrams) / total if total > 0 else 0.5

    def _length_reward(self, response_len):
        if response_len < self.min_len:
            return response_len / self.min_len
        if response_len > self.max_len:
            return max(0.0, 1.0 - (response_len - self.max_len) / self.max_len)
        return 1.0
