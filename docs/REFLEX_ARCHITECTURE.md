# NeuroStream-Reflex 架构设计

---

## 核心设计原则

```
每个神经元都知道它需要知道什么，不需要建一张全脑计算图。
每个困惑都有一条明确的求正路径，不需要在黑暗中猜测。
```

### 原则 1：逐层梯度管理（替代 retain_graph 循环）

**问题现状**：SEQ 中 32 个 expert 共享一个计算图，用 `retain_graph=True` 挂住图循环 32 次再释放。

**Reflex 方案**：按层反向传播，每层 expert 循环完后立即释放该层计算图。

```
SEQ:     [forward all layers] → [compute loss] → [retain_graph loop 32 experts] → [free]
Reflex:  [forward all layers] → [compute loss] → [layer 3 grads → free] → [layer 2 grads → free] → ...
```

实现方式：`GradientManager` 按层倒序遍历，只保留前一层需要的图。

### 原则 2：结构化外部求证（替代 scalar reward）

**问题现状**：SEQ 中用户反馈是 `_parse_feedback(text)` 返回的标量 `{1.0, -1.0, 0.2}`，只用于调 query_proj 和阈值。

**Reflex 方案**：用户回答的文本作为结构化的纠偏信号，直接反传到产生困惑的 token 区域。

```
SEQ:     用户的 "错了" → +1/-1 标量 → 调 query_proj（不学知识）
Reflex:  用户的 "量子不是粒子而是场" → 编码 → 
         与困惑区域 embedding 对齐 → 
         局部 correction_loss → 
         直接更新产生困惑的 expert（真正学到知识）
```

### 原则 3：事件驱动循环（替代忙等待）

**问题现状**：SEQ 内循环 `time.sleep(0.001)` + 轮询，CPU 浪费。

**Reflex 方案**：`threading.Event()` 阻塞等待，暂停时零 CPU 消耗。

### 原则 4：持续巩固（替代 5000 步硬间隔）

**问题现状**：SEQ 每 5000 步才做一次睡眠巩固，其间 plastic 专家可能发生了灾难性遗忘。

**Reflex 方案**：指数衰减 + 迷你巩固 + 定期完整巩固，三级保护：

```
每步：plastic 权重 → exponential decay (λ=1e-6)
每50步：当前知识 → 蒸馏到 stable（mini）
每500步：Fisher EWC + 完整蒸馏（major）
每5000步：soft_reset + 清空缓冲区（full sleep）
```

### 原则 5：确定性求正反馈通路

**问题现状**：SEQ 中用户回答的文本虽然拼入 `full_ids` 参与了 next-token LM loss，但没有针对困惑区域做定向强化。

**Reflex 方案**：每条对话的回答同时走两条路：

```
用户回答文本 ──→ token-level alignment ──→ 困惑 expert 定向 correction loss
            ──→ 标准 LM loss（不变）
```

---

## 架构全景图

```
                        外部环境（用户/API/数据）
                                │
                                ▼
  ┌──────────────────── 外循环（InteractionLoop）──────────────────┐
  │  [输入] → encode → [有追问?] → structured_feedback →         │
  │  → generate response → [input_ids+question_ids+response]     │
  │  → forward_with_novelty → LM loss → focalized_plastic_update │
  │  → return response                                           │
  └───────────────┬──────────────────────────────────────────────┘
                  │ replay buffer ←┐
                  │                │
  ┌───────────────▼────────────────┴──── 内循环 ──────────────────┐
  │  Event-driven 后台线程：                                      │
  │  Stage A: v_t → noise → Contemplator → u_next → EMA blend   │
  │  Stage B: 逐层梯度管理 Hebbian 更新                            │
  │  Stage C: 逐层重前向 → Contemplator Adam 更新                  │
  │  Stage D: coherence check → push to replay + endosphere      │
  │  Stage E: 消化队列（每步）                                      │
  │  Stage F: 迷你巩固 + query_proj 预训练 (每50步)                │
  │  Stage G: 序列模式序列学习 (每20步)                              │
  │  Stage H: 课程自对弈 (每40步)                                   │
  │  Stage I: 持续指数衰减 (每步)                                    │
  │  Stage J: 主睡眠巩固 (每500步)                                  │
  │  Stage K: 架构自修改 (每200步)                                  │
  └──────────────────────────────────────────────────────────────┘
```

---

## 核心模块设计

### 1. GradientManager（loop/gradient_manager.py）

解决 SEQ `retain_graph=True` 的核心问题：

```python
# Reflex 的逐层梯度管理，不是 SEQ 的 32-expert 全保留
for layer_idx in range(num_layers - 1, -1, -1):
    layer = layers[layer_idx]
    retain = layer_idx > 0  # 只有前一层还需要图时才 retain
    for expert in layer.all_experts:
        if expert._output is not None:
            grad_y = autograd.grad(
                loss, expert._output,
                retain_graph=retain,    # 每层结束后释放
                ...
            )
            expert.update_local(grad_y, ...)
    # 离开该层 → 该层的图被释放（retain=False 时 autograd 自动释放）
```

对于 4 层 MoE（每层 8 expert），内存峰值从「32 个 expert 的图同时存活」降为「8 个 expert 的图同时存活」。

### 2. StructuredFeedback（interaction/feedback.py）

用户反馈从「标量 reward」升级为「结构化学习信号」：

```
用户回答： "量子不是粒子而是场"
                  │
           ┌──────┴──────┐
           ▼              ▼
    encode(answer)    encode(question)
           │              │
           ▼              ▼
    answer_emb ─── cosine_sim ─── question_emb
           │              │
           │    困惑区域 embedding（前向缓存）
           │              │
           ▼              ▼
    correction_signal = alignment × (answer_emb - confused_emb)
           │
           ▼
    直接作用于困惑 expert:
    focal_gamma = alignment × uncertainty_excess × plasticity
    expert.update_local(grad_y, effective_lr, focal_gamma)
```

不再需要 `retain_graph`——correction_signal 直接从 cached embedding 域计算，不依赖完整的 forward 计算图。

### 3. EventDrivenLoop（loop/internal_loop.py）

```python
class InternalLoop:
    def __init__(self):
        self._event = threading.Event()
        self._event.set()           # 初始为 running
        self._running_ = True
    
    def pause(self):
        self._event.clear()         # 阻塞 wait() 调用
    
    def resume(self):
        self._event.set()           # 解除阻塞
    
    def _run(self):
        while self._running_:
            self._event.wait()      # paused 时阻塞，零 CPU
            self._execute_step()
```

### 4. FocalizedHebbianUpdate（learn/hebbian_update.py）

与 SEQ 的 `update_local` 相同的数学形式，但增加了：
- 逐层图管理
- 可选的焦点调制（指定某些 expert 的 gamma 放大）
- 激活门控平滑（不变）

```python
def focal_update(expert, grad_y, lr, base_gamma, focus_boost=1.0):
    norm_h = torch.norm(expert._hidden, p=2).mean()
    gate = 1.0 / (1.0 + torch.exp(5.0 * (norm_h - 3.0)))
    gamma = base_gamma * gate * focus_boost
    # Hebbian: grad^T @ hidden（和 SEQ 一样，这是正确的局部梯度）
    delta_w2 = grad_y.T @ expert._hidden
    expert.fc2.weight.data -= lr * gamma * delta_w2
```

### 5. Dual Critic（learn/critic.py）

两个 Critic 跨不同时间尺度运行，解决 SEQ 中 Critic 训练不稳定的问题：

| Critic | 更新频率 | 用途 |
|--------|---------|------|
| Fast Critic | 每步 | TD-error → 瞬时 lr_bias 调制 |
| Slow Critic | 每 20 步 | 长期价值估计 → 睡眠优先级 |
| 组合 | —— | `effective_V = α × fast + (1-α) × slow` |

---

## SEQ 中保留的部分

这些模块的数学形式和接口在 Reflex 中直接保留（因为它们是正确的）：

| 模块 | 保留原因 |
|------|---------|
| `Expert.update_local` 的 `grad^T @ hidden` | 这是正确的局部梯度计算 |
| `uncertainty_head`（d_ff→32→8→1→sigmoid） | 不确定性建模正确，且简单有效 |
| `Router.top_k routing + aggregate_sigma` | MoE 路由逻辑没有 bug |
| `Contemplator（GRU + 门控残差）` | 想象网络架构合理 |
| `MetaLearner` 策略选择 | 无状态 bandit 元学习，简单有效 |
| `sleep_consolidation 中的 distillation + EWC` | 成熟的知识巩固机制 |
| `reward_model` 的多维度计算 | 一致性+困惑度+多样性+长度 |
| `ConfusionTracker` | 困惑历史追踪设计合理 |

---

## SEQ 中修改的部分

| 问题 | SEQ | Reflex |
|------|-----|--------|
| 计算图管理 | 32 expert 共享 retain_graph | 逐层 retain，每层后释放 |
| 反馈处理 | 标量 reward → query_proj | 结构化对齐 → 定向 expert 更新 |
| 循环暂停 | busy-wait 1ms | Event-driven 阻塞 |
| 睡眠间隔 | 固定 5000 步 | 持续指数衰减 + 3 级巩固 |
| query_id 训练 | 仅用于 query_proj | query_id 拼入 LM loss 的 CTX |
| 用户反馈的专家调制 | 仅 internal loop | 外循环 `focal_update` 直接使用 |
| 代码质量 | 死代码、未定义变量 | 每函数一个职责，无 dead path |

---

## 学习循环对比：SEQ vs Reflex

### SEQ（当前）

```
每轮内循环：
  Stage A: 前向（建全局图，所有层）
  Stage B: retain_graph=True, 32 expert 循环（图一直挂住）
  Stage C: backward → Contemplator.step（图释放）
  内存峰值：1 个计算图 + 32 个图的 grad 中间张量
```

### Reflex（新）

```
每轮内循环：
  Stage A: 前向（建全局图，所有层）
  Stage B: 逐层反向，每层 expert 循环后释放
  Stage C: 逐层重前向 → Contemplator.step
  内存峰值：1 个计算图 + 8 个图的 grad 中间张量（1/4）
```

---

## 外循环求证通路对比

### SEQ

```
用户输入 → process_text
  ├─ 有追问？→ _parse_feedback → scalar reward → update query_proj + threshold
  ├─ generate response
  ├─ cat(input, response) → LM loss → _apply_plastic_updates
  └─ 返回 response
```

用户回答的语义内容消失了——只剩一个标量 reward。

### Reflex

```
用户输入 → process_text
  ├─ 有追问？→ encode(answer) + token-level alignment
  │             ├─ reward (scalar, 保持兼容)
  │             ├─ alignment_signal → focalized_update on confused experts
  │             └─ correction_gamma = f(alignment, uncertainty, plasticity)
  ├─ generate response
  ├─ cat(question_ids, EOS, input, response, EOS) → LM loss
  │   └─ focalized_update (LM loss gradient × correction_gamma)
  └─ 返回 response
```

用户回答的语义**直接校正了产生困惑的 expert 权重**。

---

## 启动命令

```bash
# 交互模式
python -m neurostream_reflex.run

# SFT 精调（从 checkpoint 恢复）
python -m neurostream_reflex.scripts.sft --resume reflex_checkpoint.pt --steps 5000

# 纯自监督运行（内循环 + 自对弈）
python -m neurostream_reflex.run --no-input
```

---

*NeuroStream-Reflex 设计 v1.0 — 2026-07-07*
