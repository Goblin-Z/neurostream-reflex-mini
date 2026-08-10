import torch
import torch.nn.functional as F

from config.model_config import ReflexConfig


class SelfPlayGenerator:
    """
    Curriculum-based self-play generation.

    The model finds its own confused spans, generates questions about them,
    answers them, and filters high-quality QA pairs for additional training.

    Curriculum: starts with easy confusions (barely above threshold) and
    gradually tackles harder ones as the model improves.
    """

    def __init__(self, config: ReflexConfig):
        self.config = config
        self.min_reward = getattr(config, 'self_play_min_reward', 0.5)
        self.max_pairs = getattr(config, 'self_play_max_pairs', 3)
        self.temperature = getattr(config, 'self_play_temperature', 0.7)
        self._difficulty = 0.0

    def adjust_difficulty(self, success_rate):
        """Raise difficulty if success > 70%, lower if < 30%."""
        if success_rate > 0.7:
            self._difficulty = min(1.0, self._difficulty + 0.05)
        elif success_rate < 0.3:
            self._difficulty = max(0.0, self._difficulty - 0.05)

    def generate(self, model, tokenizer, replay_item):
        if tokenizer is None:
            return []

        input_ids = replay_item.get('input_ids')
        if input_ids is None:
            return []

        device = next(model.parameters()).device
        if torch.is_tensor(input_ids):
            input_ids = input_ids.to(device)

        spans = self._extract_spans(model, input_ids)
        if not spans:
            return []

        qa_pairs = []
        for span in spans[:self.max_pairs]:
            text = span.get('text', '')
            if len(text) < 2:
                continue

            q_ids = self._gen_question(model, tokenizer, text, input_ids, device)
            if q_ids is None:
                continue

            a_ids, a_text = self._gen_answer(model, tokenizer, q_ids, device)
            if a_ids is None:
                continue

            reward = self._compute_reward(model, q_ids, a_ids, a_text)
            if reward > self.min_reward:
                qa_pairs.append({
                    'question': tokenizer.decode(q_ids[0], skip_special_tokens=True),
                    'answer': a_text,
                    'reward': reward,
                    'confused_span': text,
                })

        return qa_pairs

    def _extract_spans(self, model, input_ids):
        sigmas = getattr(model, '_last_token_sigmas', None)
        if sigmas is None:
            model.eval()
            with torch.no_grad():
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
                model(input_ids)
                sigmas = getattr(model, '_last_token_sigmas', None)
        if sigmas is None or sigmas.numel() == 0:
            return []

        spans = []
        used = set()
        threshold = self.config.verify_threshold
        difficulty_boost = self._difficulty * threshold

        for _ in range(self.max_pairs):
            valid = [(i, sigmas[0, i].item())
                     for i in range(sigmas.size(1)) if i not in used]
            if not valid:
                break
            anchor = max(valid, key=lambda x: x[1])[0]
            if sigmas[0, anchor].item() <= threshold + difficulty_boost:
                break

            left, right = anchor, anchor
            while (left > anchor - 3 and left > 0
                   and sigmas[0, left - 1].item() > threshold * 0.7):
                left -= 1
            while (right < anchor + 3 and right < sigmas.size(1) - 1
                   and sigmas[0, right + 1].item() > threshold * 0.7):
                right += 1

            tok = model._decode_tokenizer
            if tok is not None:
                ids = input_ids[0, left:right + 1].tolist()
                text = tok.decode(ids, skip_special_tokens=True).strip()
                if len(text) >= 2:
                    spans.append({
                        'text': text, 'positions': (left, right),
                        'sigma': sigmas[0, anchor].item(),
                    })
            used.update(range(left, right + 1))
        return spans

    def _gen_question(self, model, tokenizer, span_text, input_ids, device):
        full = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        prompt = f"关于'{full}'中的'{span_text}'，请用中文提问："
        try:
            p_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        except Exception:
            return None
        was = model.training
        model.eval()
        try:
            with torch.no_grad():
                out = model.generate(p_ids, max_new_tokens=500,
                                     temperature=self.temperature,
                                     top_k=40, top_p=0.9)
            q = tokenizer.decode(out[0][p_ids.size(1):], skip_special_tokens=True)
            q = q.strip()
            if len(q) < 3:
                return None
            return out[:, p_ids.size(1):]
        finally:
            if was:
                model.train()

    def _gen_answer(self, model, tokenizer, q_ids, device):
        was = model.training
        model.eval()
        try:
            with torch.no_grad():
                out = model.generate(q_ids.to(device), max_new_tokens=500,
                                     temperature=self.temperature,
                                     top_k=40, top_p=0.9)
            a = tokenizer.decode(out[0][q_ids.size(1):], skip_special_tokens=True)
            return out[:, q_ids.size(1):], a.strip()
        finally:
            if was:
                model.train()

    def _compute_reward(self, model, q_ids, a_ids, a_text):
        """Compute self-consistency reward."""
        from learn.reward import ReflexRewardModel
        r = ReflexRewardModel(self.config)
        return r.compute_consistency(model, q_ids, a_ids, a_text)


class SelfGeneratedBuffer:
    """Thread-safe buffer for self-play QA pairs with capacity-based eviction."""

    def __init__(self, capacity=1000):
        import threading
        self.buffer = []
        self.capacity = capacity
        self._lock = threading.RLock()

    def add(self, qa_pair):
        with self._lock:
            if len(self.buffer) >= self.capacity:
                worst = min(self.buffer, key=lambda e: e.get('reward', 0))
                self.buffer.remove(worst)
            self.buffer.append(qa_pair)

    def sample(self, n=4):
        import random
        with self._lock:
            if not self.buffer:
                return []
            return random.sample(self.buffer, min(n, len(self.buffer)))

    def __len__(self):
        with self._lock:
            return len(self.buffer)
