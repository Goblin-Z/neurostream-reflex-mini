import torch
import random
import threading


class PriorityReplayBuffer:
    """
    Priority-weighted replay buffer.

    Samples with probability proportional to (novelty + |loss|).
    Thread-safe.
    """

    def __init__(self, capacity: int = 10000, d_model: int = 512):
        self.capacity = capacity
        self.d_model = d_model
        self.buffer = []
        self.priorities = []
        self._lock = threading.RLock()

    def add(self, embedding, loss, novelty, input_ids=None):
        item = {
            'embedding': embedding.detach().clone().cpu(),
            'input_ids': input_ids.detach().clone().cpu() if input_ids is not None else None,
            'loss': loss,
            'novelty': novelty,
        }
        priority = novelty + abs(loss)
        with self._lock:
            if len(self.buffer) >= self.capacity:
                min_idx = min(range(len(self.priorities)),
                              key=lambda i: self.priorities[i])
                self.buffer[min_idx] = item
                self.priorities[min_idx] = priority
            else:
                self.buffer.append(item)
                self.priorities.append(priority)

    def sample(self, batch_size=128):
        with self._lock:
            n = len(self.buffer)
            if n == 0:
                return None
            actual = min(batch_size, n)
            if actual >= n:
                indices = list(range(n))
            else:
                total = sum(max(p, 0.001) for p in self.priorities)
                if total <= 0:
                    indices = random.sample(range(n), actual)
                else:
                    weights = [max(p, 0.001) / total for p in self.priorities]
                    indices = random.choices(range(n), weights=weights, k=actual)
            batch = [self.buffer[i] for i in indices]
        stacked = torch.stack([b['embedding'] for b in batch])
        return {'embeddings': stacked, 'indices': indices, 'items': batch}

    def clear_old(self, keep_ratio=0.5):
        with self._lock:
            n = len(self.buffer)
            if n == 0:
                return
            keep = max(1, int(n * keep_ratio))
            idx = sorted(range(n), key=lambda i: self.priorities[i], reverse=True)[:keep]
            self.buffer = [self.buffer[i] for i in idx]
            self.priorities = [self.priorities[i] for i in idx]

    def __len__(self):
        with self._lock:
            return len(self.buffer)
