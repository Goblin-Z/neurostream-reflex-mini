import time
import hashlib


class ConfusionMap:
    """
    Concept-level confusion tracking.

    While token-level sigma tells the model "position 3 is uncertain",
    the ConfusionMap tracks "I've been confused about 量子力学 for 5
    consecutive turns" — a meta-cognitive signal.

    Each confused text span is hashed into a concept key. The map tracks:
      - How many times this concept has been confusing
      - Average sigma when this concept appears
      - How long ago it was last seen
      - Whether it has been resolved (asked about and answered)

    This enables the model to:
      1. Prioritize questions about long-standing confusions
      2. Recognize when a concept has been resolved
      3. Build a "knowledge gap" map over time
    """

    def __init__(self, resolution_threshold=3):
        """
        Args:
            resolution_threshold: after this many consecutive low-sigma
                                  observations, a concept is marked resolved.
        """
        self._map = {}  # concept_hash → ConceptEntry
        self._resolution_threshold = resolution_threshold
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

        concept_hash = self._hash_concept(confused_text)

        if concept_hash not in self._map:
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

        Uses a simple normalization (lowercase, strip whitespace)
        and MD5 hash. In a full implementation, this could use
        embedding similarity to group semantically related spans.
        """
        normalized = text.lower().strip()[:50]
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()[:12]
