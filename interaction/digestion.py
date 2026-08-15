import torch
import torch.nn.functional as F

from config.model_config import ReflexConfig
from learn.hebbian_update import focal_update


class DigestionQueue:
    """
    [未接入 - 预留模块] Async digestion queue for high-perplexity inputs.
    当前没有任何调用方（文档宣称的 Stage E 消化队列未接入内循环），
    保留作为后续"高困惑输入消化"设计的实现参考。

    When the external loop encounters input it's uncertain about
    (high perplexity or high sigma), it enqueues the input for
    async processing by the internal loop.

    The internal loop pops one item per step and does a standard
    LM training on it, updating plastic experts + router + lm_head.
    """

    def __init__(self, config: ReflexConfig):
        self.config = config
        self.queue = []

    def enqueue(self, input_ids, attention_mask=None, perplexity=0.0, sigma=0.0):
        self.queue.append({
            'input_ids': input_ids.detach().clone().cpu(),
            'attention_mask': attention_mask.detach().clone().cpu()
                               if attention_mask is not None else None,
            'perplexity': perplexity,
            'sigma': sigma,
        })

    def should_enqueue(self, logits, input_ids, sigma):
        if not torch.is_tensor(logits) or logits.dim() != 3:
            return False
        shift = input_ids[..., 1:]
        loss = F.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
            shift.contiguous().view(-1),
        ).item()
        ppl = loss
        ppl_threshold = getattr(self.config, 'digestion_perplexity_threshold', 5.0)
        sigma_threshold = getattr(self.config, 'digestion_sigma_threshold', 0.6)
        return ppl > ppl_threshold or sigma > sigma_threshold, ppl

    def digest_one(self, model):
        if not self.queue:
            return None

        item = self.queue.pop(0)
        input_ids = item.get('input_ids')
        if input_ids is None or input_ids.size(1) < 2:
            return None

        device = next(model.parameters()).device
        input_ids = input_ids.to(device)

        was_training = model.training
        try:
            model.train()
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                input_ids[:, 1:].contiguous().view(-1),
            )

            plastic_lr = self.config.feedback_lr
            params = []
            for layer in model.layers:
                for expert in layer.get_plastic_experts():
                    params.extend(expert.parameters())
                params.append(layer.router.gate_weight)
                params.append(layer.router.gate_bias)
            params.append(model.lm_head.weight)

            optimizer = torch.optim.Adam(params, lr=plastic_lr)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params,
                                           self.config.feedback_max_grad_norm)
            optimizer.step()

            return {'loss': loss.item(), 'input_len': input_ids.size(1)}
        finally:
            if not was_training:
                model.eval()

    def __len__(self):
        return len(self.queue)
