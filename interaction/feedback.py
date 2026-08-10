import torch
import torch.nn.functional as F
import math

from config.model_config import ReflexConfig


class StructuredFeedback:
    """
    Structured external validation pathway.

    When a user answers a verification question, their answer is NOT
    reduced to a scalar reward. Instead, it's processed as a structured
    learning signal:

    1. The answer text is encoded to an embedding
    2. The embedding is aligned with the confused region's embedding
    3. An alignment-weighted correction signal is computed
    4. This signal directly updates the confused experts via focal gamma

    This creates a genuine external-to-internal knowledge pathway,
    not just a scalar reward for meta-learning.
    """

    def __init__(self, config: ReflexConfig):
        self.config = config
        self.alignment_weight = config.feedback_alignment_weight
        self.strength = config.feedback_gamma_modulation_strength

    def process(self, pending, user_text, tokenizer, model):
        """
        Process user answer to a verification question.

        Args:
            pending: pending verification dict (expert_id, query_ids, etc.)
            user_text: user's answer text
            tokenizer: tokenizer for encoding
            model: the model

        Returns:
            feedback_context dict or None
        """
        reward = self._keyword_feedback(user_text)

        confused_span = pending.get('confused_text')
        query_ids = pending.get('query_ids')
        expert_id = pending.get('expert_id')

        if (confused_span is None or expert_id is None
                or user_text is None or len(user_text.strip()) < 2
                or tokenizer is None):
            return {'reward': reward, 'focal_boost': None}

        expert = model.get_expert_by_id(expert_id)
        if expert is None:
            return {'reward': reward, 'focal_boost': None}

        with torch.no_grad():
            answer_ids = tokenizer(user_text, return_tensors='pt')['input_ids']
            if answer_ids is None or answer_ids.size(1) < 1:
                return {'reward': reward, 'focal_boost': None}

            device = next(model.parameters()).device
            answer_ids = answer_ids.to(device)
            answer_emb = model.token_embedding(answer_ids).mean(dim=1)

            # Align against the model's actual question (confused_text),
            # NOT the user's previous input (query_ids).  The confused_text
            # is what the model asked about, so the user's answer should be
            # semantically compared to it.
            ref_ids = None
            if confused_span is not None:
                ref_ids = tokenizer(confused_span, return_tensors='pt')['input_ids']
            if ref_ids is None or ref_ids.size(1) < 1:
                ref_ids = query_ids
            if ref_ids is not None:
                r_ids = ref_ids.to(device)
                if r_ids.dim() == 1:
                    r_ids = r_ids.unsqueeze(0)
                ref_emb = model.token_embedding(r_ids).mean(dim=1)
            else:
                return {'reward': reward, 'focal_boost': None}

            alignment = F.cosine_similarity(answer_emb, ref_emb).item()
            alignment = max(-1.0, min(1.0, alignment))
            # Normalize alignment to [0, 1], 0 = unrelated, 1 = highly aligned
            alignment = (alignment + 1.0) / 2.0

            if alignment < 0.3:
                return {'reward': reward, 'focal_boost': None}

            expert_sigma = expert.avg_uncertainty
            threshold = self.config.verify_threshold
            excess = max(0.0, (expert_sigma - threshold) / max(threshold, 0.01))
            lr_min, lr_max = 1e-7, 1e-4
            log_lr = math.log(max(expert.baseline_lr, 1e-10))
            plasticity = (log_lr - math.log(lr_min)) / (math.log(lr_max) - math.log(lr_min))
            plasticity = max(0.0, min(1.0, plasticity))

            if reward < 0:
                focal_boost = 1.0 + self.strength * excess * plasticity * alignment
            elif reward > 0:
                focal_boost = 1.0 - self.strength * excess * plasticity * alignment * 0.5
            else:
                focal_boost = 1.0 + self.strength * 0.5 * excess * plasticity * alignment

            focal_boost = max(0.1, min(focal_boost, 5.0))

        return {
            'reward': reward,
            'focal_boost': focal_boost,
            'expert_id': expert_id,
            'answer_text': user_text,
        }

    def _keyword_feedback(self, text):
        if not text:
            return 0.0
        lower = text.lower().strip()
        # M2 fix: 负向词表【先】匹配，且负向词用更长短语——
        # 原实现正向单字"对/好"先匹配，导致"不对/不好"被判 +1.0
        # （负向回答被当成确认，focal 方向反转）
        negative = ['不对', '错了', '不是', '不好', '错误', '没对',
                    'no', 'wrong', 'incorrect', '不懂', '没明白',
                    '什么意思', '不知道', '错了']
        for pat in negative:
            if pat in lower:
                return -1.0
        positive = ['是的', '正确', '没错', 'yes', 'right', 'correct',
                    '好的', '可以', 'ok', '谢谢', '明白了', '懂了',
                    '对，', '对的', '对呀', '对！']
        for pat in positive:
            if pat in lower:
                return 1.0
        # 兜底：孤立单字"对/好"（前后非否定词）才判正
        import re
        if re.search(r'(?<![不没未无])对(?![不])', lower) or \
           re.search(r'(?<![不没未无])好(?![不])', lower):
            return 1.0
        return 0.0
