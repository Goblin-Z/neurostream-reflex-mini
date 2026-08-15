import time
import hashlib
import re


class ConfusionMap:
    """
    Concept-level confusion tracking.

    While token-level sigma tells the model "position 3 is uncertain",
    the ConfusionMap tracks "I've been confused about 量子力学 for 5
    consecutive turns" — a meta-cognitive signal.

    Each confused text span is grouped into a concept key. The map tracks:
      - How many times this concept has been confusing
      - Average sigma when this concept appears
      - How long ago it was last seen
      - Whether it has been resolved (asked about and answered)

    This enables the model to:
      1. Prioritize questions about long-standing confusions
      2. Recognize when a concept has been resolved
      3. Build a "knowledge gap" map over time
    """

    def __init__(self, resolution_threshold=3, group_jaccard=0.5):
        """
        Args:
            resolution_threshold: after this many consecutive low-sigma
                                  observations, a concept is marked resolved.
            group_jaccard: 字符 2-gram Jaccard 相似度阈值——同概念不同措辞
                           （"量子力学是什么" vs "什么是量子力学"）聚合为一组。
                           修复：原 MD5 精确哈希导致同概念永不累积（WIKI 4.17）。
        """
        self._map = {}  # concept_hash → ConceptEntry
        self._resolution_threshold = resolution_threshold
        self._group_jaccard = group_jaccard
        self._total_concepts = 0
        self._resolved_concepts = 0

    def record(self, confused_text, sigma, step=0):
        """
        Record a confusion observation.

        Args:
            confused_text: the text span the model is confused about
            sigma: the uncertainty value
            step: current internal step
        """
        if not confused_text or len(confused_text.strip()) < 1:
            return

        concept_hash = self._find_group(confused_text)
        if concept_hash is None:
            concept_hash = self._hash_concept(confused_text)
            self._map[concept_hash] = {
                'text': confused_text,
                'count': 0,
                'total_sigma': 0.0,
                'first_seen_step': step,
                'last_seen_step': step,
                'resolved': False,
                'low_sigma_streak': 0,
                'asked_about': False,
            }
            self._total_concepts += 1

        entry = self._map[concept_hash]
        entry['count'] += 1
        entry['total_sigma'] += sigma
        entry['last_seen_step'] = step
        entry['avg_sigma'] = entry['total_sigma'] / entry['count']

        if sigma < 0.3:
            entry['low_sigma_streak'] += 1
            if entry['low_sigma_streak'] >= self._resolution_threshold:
                if not entry['resolved']:
                    entry['resolved'] = True
                    self._resolved_concepts += 1
        else:
            entry['low_sigma_streak'] = 0
            entry['resolved'] = False

    def mark_asked(self, confused_text):
        """Mark a concept as having been asked about."""
        concept_hash = self._find_group(confused_text)
        if concept_hash is None:
            concept_hash = self._hash_concept(confused_text)
        if concept_hash in self._map:
            self._map[concept_hash]['asked_about'] = True

    def get_most_urgent(self, top_n=3):
        """
        Get the most urgent unresolved confusions.

        Urgency = count × avg_sigma × (1 + asked_penalty)

        Concepts that have been asked about but not resolved get
        reduced urgency (we already tried to resolve them).
        """
        candidates = [
            (h, e) for h, e in self._map.items()
            if not e['resolved'] and e['count'] >= 1
        ]
        if not candidates:
            return []

        def urgency(entry):
            asked_penalty = 0.3 if entry['asked_about'] else 1.0
            return (entry['count'] * entry['avg_sigma'] * asked_penalty)

        candidates.sort(key=lambda x: urgency(x[1]), reverse=True)
        return [
            {
                'text': e['text'],
                'count': e['count'],
                'avg_sigma': e['avg_sigma'],
                'asked': e['asked_about'],
                'urgency': urgency(e),
            }
            for _, e in candidates[:top_n]
        ]

    def get_stats(self):
        return {
            'total_concepts': self._total_concepts,
            'active': sum(1 for e in self._map.values() if not e['resolved']),
            'resolved': self._resolved_concepts,
            'asked_unresolved': sum(
                1 for e in self._map.values()
                if e['asked_about'] and not e['resolved']
            ),
        }

    def _hash_concept(self, text):
        """
        Hash a text span into a concept key.

        Uses a simple normalization (lowercase, strip whitespace/punctuation)
        and MD5 hash.  Semantic grouping is handled by _find_group (2-gram
        Jaccard), so the hash itself only needs to be deterministic.
        """
        normalized = re.sub(r'[^\u4e00-\u9fffA-Za-z0-9]', '', text.lower())[:50]
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()[:12]

    def _grams(self, text):
        """字符 2-gram 集合（归一化后）。"""
        n = re.sub(r'[^\u4e00-\u9fffA-Za-z0-9]', '', text.lower())[:60]
        return set(n[i:i + 2] for i in range(len(n) - 1))

    def _find_group(self, text):
        """
        语义分组：与已有概念的代表文本做 2-gram Jaccard 相似度，
        超过 _group_jaccard（默认 0.5）即归入该概念——同概念不同措辞聚合。
        返回已有 concept_hash 或 None（新建）。
        """
        g = self._grams(text)
        if not g:
            return None
        best_key, best_j = None, 0.0
        for h, e in self._map.items():
            eg = self._grams(e.get('text', ''))
            if not eg:
                continue
            j = len(g & eg) / max(1, len(g | eg))
            if j > best_j:
                best_j, best_key = j, h
        if best_key is not None and best_j >= self._group_jaccard:
            return best_key
        return None
