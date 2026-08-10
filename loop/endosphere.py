import torch
import threading
from collections import deque


class EndoSphereBuffer:
    """
    State buffer for the internal loop.

    Stores state vectors in a deque. Thread-safe access with RLock.
    """

    def __init__(self, d_model: int, capacity: int = 1024):
        self.d_model = d_model
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.seq_buffer = deque(maxlen=capacity // 2)
        self._lock = threading.RLock()

    def push(self, vector: torch.Tensor, sigma: float = 0.5):
        with self._lock:
            self.buffer.append(vector.detach().clone().cpu())

    def get_latest(self):
        with self._lock:
            if self.buffer:
                return self.buffer[-1]
        return None

    def sample_batch(self, batch_size: int):
        import random
        with self._lock:
            n = len(self.buffer)
            if n == 0:
                return None
            actual = min(batch_size, n)
            indices = random.sample(range(n), actual)
            batch = [self.buffer[i] for i in indices]
        return torch.stack(batch)

    def push_sequence(self, embeddings, input_ids):
        with self._lock:
            self.seq_buffer.append({
                'embeddings': embeddings.detach().clone().cpu(),
                'input_ids': input_ids.detach().clone().cpu(),
            })

    def sample_sequence(self, window_size=32):
        import random
        with self._lock:
            if not self.seq_buffer:
                return None
            item = random.choice(list(self.seq_buffer))
        emb, ids = item['embeddings'], item['input_ids']
        if emb.dim() == 2 and emb.size(0) > window_size:
            start = random.randint(0, emb.size(0) - window_size)
            emb = emb[start:start + window_size]
            ids = ids[start:start + window_size]
        return emb.unsqueeze(0), ids.unsqueeze(0)

    @property
    def num_sequences(self):
        with self._lock:
            return len(self.seq_buffer)

    def clear(self):
        with self._lock:
            self.buffer.clear()
            self.seq_buffer.clear()

    def __len__(self):
        with self._lock:
            return len(self.buffer)
