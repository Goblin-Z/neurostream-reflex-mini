# NeuroStream-Reflex v2: 记忆系统融入学习设计方案

> 状态：设计完成，待实现
> 核心目标：学习即认知、学习即记忆
> 设计原则：一切顺其自然，不程序化调度，sigma 驱动一切

---

## 一、设计哲学

### 当前 v1 的问题

```
记忆系统                          学习系统
─────────                        ─────────
DialecticalBuffer (存储状态)      focal_update (更新权重)
ReplayBuffer (存储经验)           contemplator.backward()
ConfusionMap (存储困惑)           critic.train()
        ↓                              ↑
    仅用于 get_latest()            仅用当前步的 loss
    仅用于 consolidation(每50步)   不读取记忆
```

记忆和学习是**分离的** - 记忆存了东西但平时不用，学习只用当前数据不用记忆。

### v2 的目标

```
记忆系统 ←-> 学习系统 (双向耦合)
─────────     ─────────
记忆驱动学习:  sigma 低时自然回忆 -> 产生训练信号
学习更新记忆:  每次权重更新 -> 改变记忆的分类和检索
认知即记忆:    forward 过程中检索相关记忆 -> 条件化推理
```

### 理论基础

1. **互补学习系统（CLS）**：海马体快速编码（局部 Hebbian），新皮层缓慢巩固（全局回放）
2. **自发回忆**：人类在空闲时（低唤醒）会自发回忆过去经历，这促进记忆巩固
3. **认知失调**：持有矛盾观点时的不适感驱动学习 - 辩证张力即认知失调
4. **联想记忆**：遇到新情况时相关记忆自动浮现，影响当前判断

---

## 二、三层记忆-学习融合

### 层 1：记忆回放（Memory Replay）

#### 触发机制（非程序化）

**不使用固定步数触发**。回放由 sigma 自然驱动：

| sigma 区间 | 状态 | 回放行为 |
|-----------|------|----------|
| sigma < 0.35 | 空闲/稳定 | 自发回忆：从 DialecticalBuffer 采样 2-4 个过去状态进行回放 |
| 0.35 ≤ sigma ≤ 0.525 | 正常工作 | 不回放，专注当前状态 |
| sigma > 0.525 | 困惑/反题 | 定向回忆：检索与当前困惑最相似的 thesis（已有知识），尝试解决矛盾 |

**设计理由**：
- 低 sigma = 模型"空闲"，自然开始回忆（像人类发呆时脑海中浮现的记忆）
- 高 sigma = 模型"困惑"，定向检索相关记忆来帮助解决矛盾
- 中间区间 = 专注当前，不打扰

#### 回放过程

```python
def _memory_replay(self, sigma):
    """
    自发回忆（低 sigma）或定向回忆（高 sigma）。
    回放产生额外的训练信号，实现"学习即记忆"。
    """
    if sigma < 0.35:
        # 自发回忆：随机采样
        samples = self.model.endosphere.sample_batch(
            batch_size=3, pool='mixed'
        )
    elif sigma > 0.525:
        # 定向回忆：检索与当前状态最相似的 thesis
        current_state = self._v_t
        samples = self.model.endosphere.retrieve_similar(
            current_state, top_k=3, pool='thesis'
        )
    else:
        return  # 专注当前，不回放

    if samples is None:
        return

    # 回放：用当前模型重新 forward 过去的经验
    with self.model._lock:
        for mem_state in samples:
            # 当前模型对过去状态的预测
            pred = self.model.forward_internal(mem_state.unsqueeze(0).unsqueeze(0))
            # 蒸馏 loss：当前模型 vs 记忆时的模型输出
            # (记忆中存储了当时的模型输出作为 target)
            mem_target = mem_state  # 简化：用状态本身作为 target
            replay_loss = F.mse_loss(pred.squeeze(1), mem_target.unsqueeze(0))

            # 局部 Hebbian 更新（从记忆 loss）
            for layer_idx, layer, retain in self._gradient_mgr.iterate_layers(
                replay_loss, self.model.layers
            ):
                for expert, grad_y in self._gradient_mgr.compute_expert_gradients(
                    replay_loss, layer, retain
                ):
                    gamma = self._sigma_gamma() * 0.5  # 回放学习率减半
                    focal_update(grad_y, expert, expert.effective_lr, gamma)
```

#### 记忆存储增强

当前 DialecticalBuffer 只存储状态向量。v2 需要额外存储：

```python
# 在 push 时额外存储模型输出（作为回放的蒸馏 target）
entry = {
    'vector': vec,                    # 状态向量
    'sigma': sigma,                   # 不确定性
    'model_output': model_output,     # 当时的模型输出（用于回放蒸馏）
    'step': step,                     # 步数（用于衰减）
    'salience': coherence,            # 显著性（用于采样优先级）
}
```

#### 显著性衰减

记忆的显著性随时间衰减，但不做硬性删除：

```python
# 每步衰减
for entry in buffer:
    entry['salience'] *= 0.999  # 缓慢衰减

# 采样时按显著性加权
probabilities = softmax([e['salience'] for e in buffer])
samples = weighted_sample(buffer, probabilities, k=3)
```

**设计理由**：记忆不是"删除"的，而是"淡忘"的。低显著性的记忆很少被回忆，但偶尔也会浮现（就像人类久远的记忆）。

---

### 层 2：辩证张力驱动（Dialectical Tension as Loss）

#### 触发机制

当 DialecticalBuffer 产生新的 antithesis 时，自动计算辩证张力：

```python
# 在 Stage D（push to dialectical buffer）之后：
new_stats = self.dialectical_stats
if new_stats['antitheses'] > prev_stats.get('antitheses', 0):
    # 新 antithesis 产生 -> 计算辩证张力
    self._compute_and_apply_tension()
```

#### 张力计算

```python
def _compute_and_apply_tension(self):
    """
    辩证张力 = antithesis 与最近 thesis 之间的距离。
    张力驱动模型尝试解决矛盾 - 这就是"认知失调驱动学习"。
    """
    # 获取最新的 antithesis
    antithesis = self.model.endosphere._antitheses[-1]
    ante_vec = antithesis['vector']
    ante_sigma = antithesis['sigma']

    # 找到最相似的 thesis
    best_sim = -1.0
    best_thesis = None
    for thesis in self.model.endosphere._theses:
        sim = F.cosine_similarity(
            ante_vec.flatten().unsqueeze(0),
            thesis['vector'].flatten().unsqueeze(0)
        ).item()
        if sim > best_sim:
            best_sim = sim
            best_thesis = thesis

    if best_thesis is None or best_sim < 0.3:
        return  # 没有相关的 thesis，无法产生张力

    # 辩证张力 = 余弦距离 * sigma 差异
    tension = (1.0 - best_sim) * abs(ante_sigma - best_thesis['sigma'])

    # 生成张力 loss
    # 目标：将 antithesis 推向 thesis（解决矛盾）
    # loss = -tension * lambda（负号因为要最小化张力）
    tension_lambda = 0.5  # 适中值，后续可实验调整

    with self.model._lock:
        # 用 antithesis 状态做 forward
        pred = self.model.forward_internal(
            ante_vec.unsqueeze(0).unsqueeze(0)
        )
        # 目标：thesis 状态（模型应该学会将反题映射到正题）
        target = best_thesis['vector'].unsqueeze(0)

        tension_loss = tension_lambda * F.mse_loss(pred.squeeze(1), target)

        # 局部 Hebbian 更新（从张力 loss）
        for layer_idx, layer, retain in self._gradient_mgr.iterate_layers(
            tension_loss, self.model.layers
        ):
            for expert, grad_y in self._gradient_mgr.compute_expert_gradients(
                tension_loss, layer, retain
            ):
                gamma = self._sigma_gamma() * tension  # 张力越大，更新越大
                focal_update(grad_y, expert, expert.effective_lr, gamma)
```

#### 张力权重

```python
tension_lambda = 0.5  # 初始值
# 后续实验方向：
# - 0.3：温和，模型慢慢解决矛盾
# - 0.5：适中，平衡速度和稳定性
# - 0.7：激进，模型快速妥协（可能导致虚假合成）
# - 1.0：过强，可能破坏已学知识
```

---

### 层 3：记忆条件化推理（Memory-Conditioned Forward）

#### 触发机制

每次内循环前向传播前，检索相关记忆注入 hidden state：

```python
def _execute_step(self):
    # Stage A: 前向传播
    v_t = self._get_state()
    if v_t is None:
        v_t = self._init_state()

    # ── 记忆条件化 ──
    # 检索与当前状态最相关的记忆
    memory_context = self._retrieve_memory_context(v_t)
    if memory_context is not None:
        v_t = v_t + 0.1 * memory_context  # beta=0.1，小幅度注入

    # 后续流程不变
    noise = self._noise_scheduler.get_noise(sigma_for_noise)
    v_t = v_t + noise * torch.randn_like(v_t)
    u_next = self._imagine(v_t)
    ...
```

#### 记忆检索

```python
def _retrieve_memory_context(self, current_state, top_k=3):
    """
    检索与当前状态最相关的记忆，加权平均为上下文向量。
    像人类联想记忆：遇到新情况时相关记忆自动浮现。
    """
    all_memories = (
        list(self.model.endosphere._theses)[-50:] +
        list(self.model.endosphere._syntheses)[-20:] +
        list(self.model.endosphere._antitheses)[-10:]
    )

    if not all_memories:
        return None

    # 计算相似度
    similarities = []
    for mem in all_memories:
        sim = F.cosine_similarity(
            current_state.flatten().unsqueeze(0),
            mem['vector'].flatten().unsqueeze(0)
        ).item()
        # 相似度 * 显著性
        weight = sim * mem.get('salience', 0.5)
        similarities.append((weight, mem['vector']))

    # 取 top_k 最相关的记忆
    similarities.sort(key=lambda x: x[0], reverse=True)
    top = similarities[:top_k]

    if not top or top[0][0] < 0.2:
        return None  # 没有足够相关的记忆

    # 加权平均
    total_weight = sum(w for w, _ in top)
    if total_weight < 1e-6:
        return None

    context = sum(w * v for w, v in top) / total_weight
    return context.to(current_state.device)
```

#### 注入强度

```python
beta = 0.1  # 记忆注入强度
# 设计理由：
# - 0.1：记忆是"提示"而非"命令"，不主导推理
# - 太大（>0.3）：记忆覆盖当前状态，模型陷入回忆
# - 太小（<0.05）：记忆几乎无影响
```

---

## 三、整合后的内循环流程

```
Stage A:  前向传播
          v_t = get_state()
          memory_ctx = retrieve_memory_context(v_t)     ← 层3: 记忆条件化
          v_t = v_t + 0.1 * memory_ctx
          noise = PI_controller(sigma)
          v_t = v_t + noise * randn
          u_next = Contemplator(v_t)
          pred = forward_internal(u_next)
          loss = MSE(pred, target)

Stage B:  局部 Hebbian 更新 (from loss)
Stage B2: 全局反向传播 (from loss, Attention/Router/LMHead)    ← v1.1 已实现
Stage C:  Contemplator 更新
Stage C2: Critic 训练
Stage C3: 记忆回放                                                ← 层1: 自发/定向回忆
          if sigma < 0.35: 自发回忆 (随机采样 3 个记忆)
          if sigma > 0.525: 定向回忆 (检索相似 thesis)
          for each memory: re-forward + Hebbian update

Stage D:  推入辩证缓冲区
          if antithesis created:
              compute_tension(antithesis, nearest_thesis)        ← 层2: 辩证张力
              hebbian_update(from tension_loss, lambda=0.5)
              print("[DIALECTIC] Tension driving learning...")

Stage E:  Mini-consolidation (每 50 步)
Stage F:  Major-consolidation (每 500 步)
Stage G:  Fluid 专家角色评估 (每 100 步)
Stage H:  问题生成 (每 20 步)
```

---

## 四、DialecticalBuffer 增强需求

### 新增方法

```python
class DialecticalBuffer:
    # ... 现有方法保持不变 ...

    def retrieve_similar(self, query_vec, top_k=3, pool='thesis'):
        """
        检索与查询向量最相似的记忆。
        用于定向回忆（高 sigma 时检索相关 thesis 解决矛盾）。
        """
        with self._lock:
            if pool == 'thesis':
                items = list(self._theses)
            elif pool == 'antithesis':
                items = list(self._antitheses)
            elif pool == 'synthesis':
                items = list(self._syntheses)
            else:
                items = (list(self._theses) +
                         list(self._antitheses) +
                         list(self._syntheses))

            if not items:
                return None

            query_flat = query_vec.flatten().unsqueeze(0)
            scored = []
            for entry in items:
                sim = F.cosine_similarity(
                    query_flat,
                    entry['vector'].flatten().unsqueeze(0)
                ).item()
                scored.append((sim, entry))

            scored.sort(key=lambda x: x[0], reverse=True)
            top = scored[:top_k]

            if not top or top[0][0] < 0.2:
                return None

            return torch.stack([entry['vector'] for _, entry in top])

    def get_all_recent(self, thesis_n=50, synthesis_n=20, antithesis_n=10):
        """
        获取最近的记忆条目（用于记忆条件化检索）。
        """
        with self._lock:
            return (
                list(self._theses)[-thesis_n:] +
                list(self._syntheses)[-synthesis_n:] +
                list(self._antitheses)[-antithesis_n:]
            )

    def decay_salience(self, rate=0.999):
        """
        显著性衰减。不删除记忆，只让低显著性记忆更难被回忆。
        像人类记忆淡忘 - 久远的记忆偶尔浮现但越来越少。
        """
        with self._lock:
            for pool in [self._theses, self._antitheses, self._syntheses]:
                for entry in pool:
                    if 'salience' not in entry:
                        entry['salience'] = 0.5
                    entry['salience'] *= rate
```

### push 方法增强

```python
def push(self, vector, sigma=0.5, model_output=None, step=0, salience=None):
    """增强版：存储模型输出和显著性，用于回放蒸馏。"""
    vec = vector.detach().clone().cpu()

    if salience is None:
        salience = 1.0 - abs(sigma - 0.5) * 2  # 距离 0.5 越远越显著

    entry = {
        'vector': vec,
        'sigma': sigma,
        'model_output': model_output.detach().clone().cpu() if model_output is not None else None,
        'step': step,
        'salience': salience,
    }

    with self._lock:
        if sigma > self.sigma_threshold * 1.05:
            self._antitheses.append(entry)
        elif sigma < self.sigma_threshold * 0.45:
            resolved = self._find_resolved_antithesis(vec)
            if resolved is not None:
                self._syntheses.append(entry)
            else:
                self._theses.append(entry)
        else:
            self._theses.append(entry)
```

---

## 五、参数汇总

| 参数 | 值 | 说明 |
|------|-----|------|
| memory_replay_low_sigma | 0.35 | sigma 低于此值触发自发回忆 |
| memory_replay_high_sigma | 0.525 | sigma 高于此值触发定向回忆 |
| memory_replay_batch_size | 3 | 每次回放采样数量 |
| memory_replay_gamma_scale | 0.5 | 回放学习率缩放（比正常学习慢一半） |
| tension_lambda | 0.5 | 辩证张力权重 |
| tension_similarity_threshold | 0.3 | thesis-antithesis 相似度低于此值不产生张力 |
| memory_context_beta | 0.1 | 记忆条件化注入强度 |
| memory_context_top_k | 3 | 记忆条件化检索数量 |
| memory_context_similarity_threshold | 0.2 | 相似度低于此值不注入记忆 |
| salience_decay_rate | 0.999 | 显著性衰减率（每步） |

---

## 六、验证指标

| 指标 | 测量方法 | 期望结果 |
|------|----------|----------|
| 记忆回放频率 | 统计 sigma < 0.35 和 > 0.525 的步数比例 | 自发回忆约 20-30%，定向回忆约 10-20% |
| 辩证张力解决率 | antithesis 被提升为 synthesis 的比例 | > 30% |
| 记忆条件化影响 | 有/无记忆注入时的 loss 差异 | 有记忆时 loss 降低 5-15% |
| 学习效率 | 达到相同 loss 所需步数 | 比无记忆版本减少 20-40% |
| 知识保持 | 1000 步后旧任务的 loss 变化 | 增加不超过 10% |

---

## 七、实现顺序建议

1. **先实现层 2（辩证张力）**：最简单，只需在 Stage D 后加几行代码
2. **再实现层 3（记忆条件化）**：中等复杂度，需在 Stage A 前加检索逻辑
3. **最后实现层 1（记忆回放）**：最复杂，需增强 DialecticalBuffer 存储 + 回放逻辑

每层实现后独立验证，确认不影响现有功能再叠加下一层。

---

## 八、与 v1 的兼容性

- v2 的所有新增功能都通过 sigma 自然触发，不改变 v1 的固定步数调度
- v2 不删除任何 v1 功能，只增加新的学习信号来源
- v2 的参数都有默认值，不设置时等同于 v1 行为
- DialecticalBuffer 的增强向后兼容：旧条目没有 model_output/salience 字段时使用默认值

---

## 九、哲学总结

```
v1: 模型有记忆，但记忆不参与学习
    记忆是"仓库"，学习是"工人"，两者分离

v2: 记忆即学习，学习即记忆
    低 sigma 时自发回忆（闲暇时的思绪漫游）
    高 sigma 时定向回忆（困惑时的联想启发）
    矛盾产生张力驱动解决（认知失调促进学习）
    相关记忆条件化推理（联想记忆影响判断）

    记忆不再是仓库，而是思维的组成部分。
    学习不再是机械更新，而是认知的自然产物。
```
