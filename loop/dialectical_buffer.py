import torch
import threading
from collections import deque


class DialecticalBuffer:
    """
    Dialectical memory structure replacing flat deque.

    Instead of storing all states in a single FIFO queue, this buffer
    maintains three pools corresponding to the dialectical triad:

        THESIS (正题):     states with low sigma — the model's stable
                           understanding, what it "knows"
        ANTITHESIS (反题):  states with high sigma — contradictions,
                           confusion, unresolved tension
        SYNTHESIS (合题):   states that were antitheses but have been
                           resolved through learning — the integration
                           of thesis and antithesis

    The push() method classifies each incoming state by its sigma:

        sigma < threshold * 0.5         → thesis
        sigma > threshold * 1.5         → antithesis
        otherwise                       → synthesis (transition zone)

    When an antithesis state is later pushed with low sigma (resolved),
    it is promoted to synthesis — marking a completed dialectical cycle.

    The get_latest() method prioritizes antithesis states for the next
    thinking cycle, because that's where the system needs to work.
    """

    THESIS = 'thesis'
    ANTITHESIS = 'antithesis'
    SYNTHESIS = 'synthesis'

    def __init__(self, d_model: int, capacity: int = 1024,
                 sigma_threshold: float = 0.5):
        self.d_model = d_model
        self.capacity = capacity
        self.sigma_threshold = sigma_threshold

        cap_each = capacity // 3
        self._theses = deque(maxlen=cap_each)
        self._antitheses = deque(maxlen=cap_each)
        self._syntheses = deque(maxlen=cap_each)

        self._lock = threading.RLock()

    def push(self, vector: torch.Tensor, sigma: float = 0.5):
        """
        Classify and store a state based on its uncertainty.

        Dialectical semantics:
          THESIS     - stable knowledge (low sigma), or moderate states
                       below the antithesis threshold.
          ANTITHESIS - unresolved confusion (high sigma).
          SYNTHESIS  - a prior antithesis that has been *resolved* by a
                       later low-sigma state.

        Thresholds are symmetric around the edge-of-chaos point
        (sigma=0.5) so that the PI controller's natural oscillation
        (±0.05) creates a healthy mix of thesis and antithesis:
          sigma > threshold * 1.05  -> antithesis  (≈0.525)
          sigma < threshold * 0.45  -> thesis/synthesis (≈0.225)
          otherwise                 -> thesis (moderate)
        """
        vec = vector.detach().clone().cpu()

        with self._lock:
            if sigma > self.sigma_threshold * 1.05:
                # High uncertainty -> antithesis (unresolved tension)
                self._antitheses.append({
                    'vector': vec,
                    'sigma': sigma,
                })

            elif sigma < self.sigma_threshold * 0.45:
                # Low uncertainty -> check if this resolves a prior antithesis
                resolved = self._find_resolved_antithesis(vec)
                if resolved is not None:
                    # Promote to synthesis -- dialectical resolution!
                    self._syntheses.append({
                        'vector': vec,
                        'sigma': sigma,
                        'resolved_sigma': resolved['sigma'],
                    })
                else:
                    # New thesis
                    self._theses.append({
                        'vector': vec,
                        'sigma': sigma,
                    })

            else:
                # Moderate sigma -> thesis (stable enough, not confused
                # enough to be antithesis).  This is the model's everyday
                # working state.
                self._theses.append({
                    'vector': vec,
                    'sigma': sigma,
                })

    def get_latest(self):
        """
        Get the most recent state for the next thinking cycle.

        Priority: antithesis (unresolved) > synthesis (transition) > thesis (stable)

        The system should think most about what it's confused about.
        """
        with self._lock:
            if self._antitheses:
                return self._antitheses[-1]['vector']
            if self._syntheses:
                return self._syntheses[-1]['vector']
            if self._theses:
                return self._theses[-1]['vector']
        return None

    def get_latest_sigma(self):
        """Get the sigma of the most recent state."""
        with self._lock:
            if self._antitheses:
                return self._antitheses[-1]['sigma']
            if self._syntheses:
                return self._syntheses[-1]['sigma']
            if self._theses:
                return self._theses[-1]['sigma']
        return 0.5

    def sample_batch(self, batch_size: int, pool='mixed'):
        """
        Sample from a specific pool or mixed.

        pool options: 'thesis', 'antithesis', 'synthesis', 'mixed'
        """
        import random
        with self._lock:
            if pool == 'thesis':
                items = list(self._theses)
            elif pool == 'antithesis':
                items = list(self._antitheses)
            elif pool == 'synthesis':
                items = list(self._syntheses)
            else:
                items = (list(self._theses)
                         + list(self._antitheses)
                         + list(self._syntheses))

            n = len(items)
            if n == 0:
                return None
            actual = min(batch_size, n)
            indices = random.sample(range(n), actual)
            batch = [items[i]['vector'] for i in indices]
        return torch.stack(batch)

    def push_sequence(self, embeddings, input_ids):
        """Store sequence data for sequence-mode learning."""
        if not hasattr(self, '_seq_buffer'):
            self._seq_buffer = deque(maxlen=self.capacity // 2)
        with self._lock:
            self._seq_buffer.append({
                'embeddings': embeddings.detach().clone().cpu(),
                'input_ids': input_ids.detach().clone().cpu(),
            })

    def sample_sequence(self, window_size=32):
        import random
        with self._lock:
            if not hasattr(self, '_seq_buffer') or not self._seq_buffer:
                return None
            item = random.choice(list(self._seq_buffer))
        emb, ids = item['embeddings'], item['input_ids']
        if emb.dim() == 2 and emb.size(0) > window_size:
            start = random.randint(0, emb.size(0) - window_size)
            emb = emb[start:start + window_size]
            ids = ids[start:start + window_size]
        return emb.unsqueeze(0), ids.unsqueeze(0)

    @property
    def num_sequences(self):
        with self._lock:
            if not hasattr(self, '_seq_buffer'):
                return 0
            return len(self._seq_buffer)

    def get_dialectical_stats(self):
        """Return counts for each pool -- the dialectical balance."""
        with self._lock:
            return {
                'theses': len(self._theses),
                'antitheses': len(self._antitheses),
                'syntheses': len(self._syntheses),
            }

    def clear(self):
        with self._lock:
            self._theses.clear()
            self._antitheses.clear()
            self._syntheses.clear()
            if hasattr(self, '_seq_buffer'):
                self._seq_buffer.clear()

    def _find_resolved_antithesis(self, vec, threshold=0.7):
        """
        Check if a low-sigma vector resolves any pending antithesis.
        Uses cosine similarity so the model can recognise that a new
        stable state resolves a *related* prior confusion.
        """
        if not self._antitheses:
            return None

        vec_flat = vec.flatten().unsqueeze(0)
        best_sim = -1.0
        best_entry = None
        best_idx = -1

        for i, entry in enumerate(self._antitheses):
            ante_flat = entry['vector'].flatten().unsqueeze(0)
            sim = torch.nn.functional.cosine_similarity(
                vec_flat, ante_flat
            ).item()
            if sim > best_sim:
                best_sim = sim
                best_entry = entry
                best_idx = i

        if best_sim >= threshold:
            # Remove the resolved antithesis
            self._antitheses = deque(
                list(self._antitheses)[:best_idx]
                + list(self._antitheses)[best_idx + 1:],
                maxlen=self._antitheses.maxlen,
            )
            return best_entry

        return None

    def __len__(self):
        with self._lock:
            return (len(self._theses)
                    + len(self._antitheses)
                    + len(self._syntheses))
