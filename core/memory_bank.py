"""
MemoryBank — 记忆系统 v4 核心组件（L1-L4）。

分层:
  - 语义槽 (memory_matrix, nn.Parameter)  : L2/L3 —— 可微长期记忆，
    状态经写入门更新，梯度随优化器演化；作为 AttnRes 的额外 source。
  - KV 缓存 (kvcache, 轮次 FIFO)         : L4 —— 对话逐 token 激活表示，
    注意力可直接 attend（内容记忆/复述能力）。

与现有模型的整合:
  - ReflexModel 注入 self.memory_bank
  - attention.py 从各层 _last_kv 收集轮次 KV
  - pipeline.py 轮次结束时调用 store_round_kv()
  - attn_res.py 读取语义槽作为跨块注意力 source
"""
import torch
import torch.nn as nn


class MemoryBank(nn.Module):
    def __init__(self, d_model: int, capacity: int = 128,
                 top_k: int = 8, write_lr: float = 0.05,
                 kv_rounds: int = 4, num_layers: int = 24,
                 n_heads: int = 10, head_dim: int = 64):
        super().__init__()
        self.d_model = d_model
        self.capacity = capacity
        self.top_k = top_k
        self.write_lr = write_lr
        self.kv_rounds = kv_rounds
        self.num_layers = num_layers
        self.n_heads = n_heads
        self.head_dim = head_dim

        # ── L2/L3: 可微语义槽 ──
        self.memory_matrix = nn.Parameter(
            torch.randn(capacity, d_model) * 0.02)
        self.write_gate = nn.Linear(d_model, 1, bias=False)
        nn.init.ones_(self.write_gate.weight)  # 初始写入门≈1
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self._pos = 0  # 环形写入指针

        # ── 自发固化: salience 系统（非参数，调度信号）──
        self.salience = torch.zeros(capacity)      # 每条语义槽的重要性累积
        self.cooldown = torch.zeros(capacity)      # 固化冷却步数
        self.salience_decay = 0.99                 # 新鲜度衰减
        self._salience_tau = 3.0                   # 成熟阈值（可被 config 覆盖）

        # ── L4: KV 缓存（轮次 FIFO）──
        self.kvcache = []  # list of {'k': [L, H, T, hd], 'v': [...], 'text': str, 'use': int}
        self._max_kv_tokens = 512       # RISK-1: 每轮 KV 最大 token 数（截断）
        self._max_kv_total = 1536       # RISK-1: 全部轮次总 KV token 上限

    # ── L2 语义槽：索引（候选筛选，决策在 AttnRes 注意力） ──

    def retrieve(self, query, top_k=None):
        """返回 top-k 相似记忆向量 [k, d]（仅索引，不做检索决策）。"""
        mem, _ = self.retrieve_with_index(query, top_k)
        return mem

    def retrieve_with_index(self, query, top_k=None):
        """返回 (记忆向量 [k, d], 槽位索引 [k])——注意力聚焦可回传索引。

        salience 同时做新鲜度衰减（每步由 AttnRes 回传权重累积）。
        """
        k = top_k or self.top_k
        k = min(k, self.capacity)
        with torch.no_grad():
            # query: [B, T, d] 或 [B, d] 或 [d] → 规约为 [1, d]
            q = query.detach()
            while q.dim() > 1:
                q = q.mean(dim=0)
            q = q.unsqueeze(0)                       # [1, d]
            sim = torch.cosine_similarity(q, self.memory_matrix, dim=-1)
            idx = sim.topk(k).indices
        return self.memory_matrix[idx], idx           # [k, d], [k]

    # ── L3 可微写入 ──

    def write(self, vector, lr=None):
        """语义槽写入（可微门控，方向 A）。vector: [d] 或 [1, d]。"""
        if vector.dim() == 1:
            vector = vector.unsqueeze(0)
        lr = lr or self.write_lr
        gate = torch.sigmoid(self.write_gate(vector)).squeeze()  # [1]
        with torch.no_grad():
            idx = self._pos % self.capacity
            self.memory_matrix.data[idx] *= (1.0 - lr * gate.item())
            self.memory_matrix.data[idx] += (
                lr * gate.item() * vector[0].detach())
            self._pos += 1

    # ── 自发固化: salience 累积与成熟检测 ──

    def accumulate_salience(self, indices, weights):
        """AttnRes 回传"被聚焦"的注意力权重 → 累积 salience。

        行为驱动：反复提及 → 反复检索 → 注意力累积 → salience 高。
        decay: 新鲜度衰减（最近使用为主）。
        """
        with torch.no_grad():
            self.salience *= self.salience_decay
            for i, w in zip(indices.tolist(), weights.tolist()):
                if 0 <= i < self.capacity:
                    self.salience[i] += w
            # 冷却递减
            self.cooldown = (self.cooldown - 1).clamp(min=0)

    def get_hot_memories(self, threshold=None, max_n=4):
        """返回 (索引 [n], 向量 [n, d])：salience 超阈值且不在冷却中的记忆。"""
        thr = threshold if threshold is not None else self._salience_tau
        hot = ((self.salience > thr) & (self.cooldown == 0))
        idx = hot.nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            return None
        idx = idx[:max_n]
        return idx, self.memory_matrix[idx]

    def mark_consolidated(self, indices, cooldown=50):
        """固化后重置 salience + 设置冷却（防连续触发）。"""
        with torch.no_grad():
            for i in indices.tolist():
                if 0 <= i < self.capacity:
                    self.salience[i] = 0.0
                    self.cooldown[i] = cooldown

    def retrieve_top_salience(self, k=1):
        """返回 salience 最高的记忆 (向量 [k, d], 索引 [k])。"""
        k = min(k, self.capacity)
        with torch.no_grad():
            idx = self.salience.topk(k).indices
        return self.memory_matrix[idx], idx

    # ── L4 KV 缓存 ──

    def store_round_kv(self):
        """从各层 attention 的 _last_kv 收集本轮 KV 并入队（FIFO）。"""
        from core.model import ReflexModel  # 延迟导入避免环
        # 通过 model 层收集由调用方完成（pipeline 直接调用各层）
        # 此处仅做占位——实际实现见 collect_round_kv
        pass

    def add_round_kv(self, layer_kvs, text=''):
        """layer_kvs: list of (k, v)，k/v 形状 [H, T, hd]（CPU fp16）。

        RISK-1 修复: 每轮 KV 截断到 _max_kv_tokens，防长对话爆显存/变慢。
        RISK-4 修复: 按使用频率加权淘汰（use_count 低的优先淘汰）。
        """
        ks = [kv[0] for kv in layer_kvs]
        vs = [kv[1] for kv in layer_kvs]
        T = ks[0].size(1)
        if T > self._max_kv_tokens:
            # 保留尾部（最近的 token 语义最新）
            ks = [k[:, -self._max_kv_tokens:, :] for k in ks]
            vs = [v[:, -self._max_kv_tokens:, :] for v in vs]
        self.kvcache.append(
            {'k': ks, 'v': vs, 'text': text, 'use': 0})
        # RISK-1b: 总 token 超限时淘汰最旧轮（近因优先）
        while (len(self.kvcache) > self.kv_rounds
               or sum(r['k'][0].size(1) for r in self.kvcache)
               > self._max_kv_total):
            self.kvcache.pop(0)

    def _evict_if_over(self):
        while len(self.kvcache) > self.kv_rounds:
            self.kvcache.pop(0)   # 近因优先：淘汰最旧轮（对话记忆自然行为）

    def get_kv(self, layer_idx):
        """返回该层所有历史 KV 的拼接（k/v 各 [H, T_sum, hd]）。

        use 计数: 每轮被检索纳入候选的次数（统计展示，不参与淘汰——
        对话记忆以近因为主，旧轮即便常被纳入也随新轮推进淡出）。
        """
        if not self.kvcache:
            return None
        ks, vs = [], []
        for round_ in self.kvcache:
            k, v = round_['k'][layer_idx], round_['v'][layer_idx]
            ks.append(k)
            vs.append(v)
            round_['use'] = round_.get('use', 0) + 1
        return (torch.cat(ks, dim=1), torch.cat(vs, dim=1))

    def clear_kv(self):
        self.kvcache = []

    # ── 状态 ──

    def get_stats(self):
        return {
            'slots_used': min(self._pos, self.capacity),
            'kv_rounds': len(self.kvcache),
        }
