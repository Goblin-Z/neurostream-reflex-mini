# NeuroStream-Reflex V2-Mini 深度质量审计 (第二轮)

> **生成日期**：2026-07-24
> **审计范围**：全部模块，基于 Mini V3 (0.769B, d640, 24L, 6E) 配置
> **审计方法**：逐模块审查代码实现 vs 设计意图，寻找会导致训练结果差的隐患
> **基准哲学**：单轮收敛、多轮防爆炸、系统级维持混沌边缘意识流

---

## 0. 核心发现概览

本轮审计发现 **3 个 P0 级隐患**（会导致训练结果差）和 **7 个 P1 级隐患**（影响质量上限），以及 **4 个 P2 级优化点**。

最关键的 3 个 P0：

| # | 问题 | 影响 |
|---|------|------|
| P0-A | 反馈通路中 `_train_on_full` 的 CE loss 梯度方向与困惑专家不对齐 | 求正通路学到的不是"用户纠正的知识"而是"对话序列的下一个 token" |
| P0-B | Sigma 聚合被 `.item()` 截断，梯度无法回流到 uncertainty_head | 不确定性信号不学习，sigma 质量不提升 |
| P0-C | 内循环 Stage D 中 replay buffer 的 priority = novelty + abs(loss)，abs(loss) 依赖 KL 项 | KL 主导 priority 导致高困惑状态被过度采样，低困惑被忽略，蒸馏偏向困惑状态 |

---

## 1. P0-A: 求正通路的 CE loss 与困惑专家不对齐

### 问题

`pipeline.py:185-200` 的 `_train_on_full` 计算的是整个对话序列的 next-token CE loss：

```python
logits = self.model(full_ids)
shift = full_ids[:, 1:]
loss = F.cross_entropy(logits[:, :-1], shift)
```

这个 loss 的梯度通过 `autograd.grad(loss, expert._output)` 传播到所有激活的专家。但**困惑专家的目标是"纠正特定困惑"，而 CE loss 是"预测对话序列"**。这两个目标不对齐。

### 具体影响链

```
用户回答 "量子不是粒子而是场"
  ↓
反馈通路: confused_text = "量子是粒子吗？" (模型的问题)
  focal_boost = 1.0 + strength * excess * plasticity * alignment
  ↓
_train_on_full: CE loss on [question + user_answer + model_response + EOS]
  → 梯度方向: "让模型在 user_answer 的下一个 token 预测更准确"
  → 但目标应该是: "让困惑专家学会 '量子是场而非粒子'"
  ↓
focal_update(grad_y, expert, lr, gamma, focus_boost=boost)
  → grad_y 来自 CE loss，不是来自困惑纠正信号
  → focal_boost 放大了一个"错误方向"的梯度
  → 专家学会了"更好地预测对话序列"而非"纠正知识错误"
```

### 为什么这是 P0

`REFLEX_ARCHITECTURE.md` 原则 2 和 5 的核心设计是：

```
用户回答的语义直接校正产生困惑的 expert 权重
```

但当前实现中，correction signal 只存在于 focal_boost 的标量乘数中，而**梯度方向本身来自 CE loss**，与用户纠正的语义内容无关。这意味着：

- 用户说 "量子不是粒子而是场" → focal_boost=2.0
- CE loss 的梯度方向是 "在 '量子' 之后预测 '不是' 更准确"
- 困惑专家被 2× 强化地学习 "预测对话序列"，而非 "量子是场"

**求正通路的核心目标被 CE loss 的梯度方向稀释了。**

### 改进方向

在 `_train_on_full` 中，除了 CE loss，还应加入一个**对齐损失**：

```python
# 额外的对齐损失: 让困惑专家的输出与用户答案的 embedding 对齐
if feedback_ctx is not None and feedback_ctx.get('focal_boost') is not None:
    # 获取用户答案的 embedding
    answer_ids = tokenizer(feedback_ctx['answer_text'], ...)
    answer_emb = model.token_embedding(answer_ids).mean(dim=1)
    # 获取困惑专家在 last forward 中的输出
    target_expert = model.get_expert_by_id(feedback_ctx['expert_id'])
    if target_expert is not None and target_expert._output is not None:
        expert_out = target_expert._output.mean(dim=1)  # [B, d_model]
        # 对齐损失: 专家输出应该与用户答案的语义方向一致
        align_loss = 1.0 - F.cosine_similarity(expert_out, answer_emb).mean()
        loss = loss + align_weight * align_loss
```

这样 CE loss 负责 "预测对话序列"，align_loss 负责 "纠正知识错误"，focal_boost 放大的是**正确方向**的梯度。

### 可行性

中。需要在 `_train_on_full` 中访问 tokenizer 和 feedback_ctx['answer_text']，并确保 target_expert._output 在 forward 后被捕获。需要修改 `feedback.py` 返回 answer_text，并在 `pipeline.py` 的 `_process_feedback` 中传入。

---

## 2. P0-B: Sigma 梯度截断 — uncertainty_head 不学习

### 问题

`model.py:81-84` 中，sigma 被 `.detach()` 截断：

```python
expert_sigmas[i] = expert_sigma.mean().detach()
per_token_sigma[rows] += (weight.squeeze(-1) * expert_sigma.squeeze(-1)).detach()
```

然后 `router.aggregate_sigma` 中 `.item()` 再次截断（`router.py:187`）：

```python
weighted = (top_k_weights * selected).sum(dim=-1)
return weighted.mean().item()  # Python float, no gradient
```

### 影响链

```
forward -> expert.sigma = sigmoid(uncertainty_head(h))
  ↓
sigma 用于:
  1. router.should_verify() -> 决定是否提问
  2. router.aggregate_sigma() -> 决定 gamma 调制强度
  3. interaction.update_sigma() -> 决定 can_ask
  4. feedback.py 的 excess 计算 -> 决定 focal_boost 强度
  ↓
但 sigma 的梯度全部被 .detach() 和 .item() 截断
  → uncertainty_head 收不到任何梯度
  → 不确定性估计永远不学习
  → sigma 的质量取决于 random init，永远不变
```

### 为什么这是 P0

sigma 是整个系统的核心信号：
- 决定什么时候提问（can_ask）
- 决定 Hebbian 更新的强度（gamma 调制）
- 决定 focal_boost 的强度（excess 计算）
- 决定辩证缓冲的分类（thesis/antithesis）

如果 sigma 不学习，所有这些决策都基于一个**不会改善**的随机初始估计。随着模型训练，模型的知识在变，但 sigma 估计永远不变 —— 这导致系统决策质量不提升。

### 改进方向

保留 detach 用于标量统计（不影响前向计算图），但**额外提供一个可微分的 sigma 通路**用于 uncertainty_head 的学习：

```python
# 在 ReflexMoELayer.forward 中:
# 保留 detached sigma 用于统计
expert_sigmas[i] = expert_sigma.mean().detach()

# 同时保留可微分的 sigma 用于 uncertainty_head 学习
if self.training:
    self._learnable_sigmas[i] = expert_sigma.mean()  # no detach
```

然后在内部循环或外部循环中，添加一个 **sigma 校准损失**：

```python
# sigma 校准: 让 sigma 接近模型实际的预测不确定性
# 用困惑度作为 target: 高 CE loss -> 高 sigma, 低 CE loss -> 低 sigma
target_sigma = torch.tanh(ce_loss)  # 归一化到 [0, 1]
sigma_calibration_loss = F.mse_loss(
    learnable_sigma_agg, target_sigma.detach()
)
loss = loss + sigma_cal_weight * sigma_calibration_loss
```

### 可行性

中。需要在 `ReflexMoELayer` 中增加 `_learnable_sigmas` 缓冲区，在 `forward` 中保留可微分 sigma，并在训练 loop 中添加 sigma 校准损失。

---

## 3. P0-C: Replay priority 被 KL 主导，蒸馏偏向困惑状态

### 问题

`internal_loop.py:347-354` 的 Stage D：

```python
loss_val = self._loss_int.item()
coherence = 1.0 - math.tanh(loss_val)
novelty = 1.0 - coherence  # = tanh(loss_val)
self._replay_buffer.add(
    self._u_next_input.squeeze(0).detach(),
    loss=loss_val,
    novelty=novelty,
)
```

`replay.py:28` 的 priority：

```python
priority = novelty + abs(loss)
```

而 `loss_val` 包含 `curiosity_beta * KL`（`internal_loop.py:653-655`）：

```python
total = (self._imagination_lambda * loss_imagination
         + self._curiosity_beta * loss_curiosity)
```

### 影响链

```
KL 散度 (curiosity) 是 loss_int 的主要组成部分:
  loss_int = imagination_lambda * MSE + curiosity_beta * KL
  其中 imagination_lambda=1.0, curiosity_beta=0.1
  当 KL 较大时 (惊讶大), loss_int 较大
  ↓
coherence = 1 - tanh(loss_int) → 低 coherence (困惑状态)
novelty = 1 - coherence = tanh(loss_int) → 高 novelty
priority = novelty + abs(loss) → 高 priority
  ↓
replay buffer 采样时:
  高 priority = 高困惑状态被过度采样
  低 priority = 低困惑 (稳定知识) 状态被忽略
  ↓
mini/major 蒸馏时:
  采样到的大多是困惑状态
  稳定知识状态很少被蒸馏到 stable 专家
  → 长期知识巩固偏向困惑，忽略稳定
```

### 为什么这是 P0

巩固机制（consolidation）的目的是将 **plastic 学习到的知识蒸馏到 stable 专家**。如果 replay buffer 偏向困惑状态，stable 专家学到的大多是困惑而非稳定知识。这会导致：

- stable 专家被"污染"（学到的不是稳定知识而是困惑）
- 长期记忆质量下降
- 模型的"自同一性"被侵蚀

这与核心哲学 "stable 专家是长期记忆库" 直接矛盾。

### 改进方向

解耦 novelty 和 loss 对 priority 的贡献，用 KL 单独作为困惑信号：

```python
# internal_loop.py Stage D:
loss_val = self._loss_int.item()
# 用 KL 单独衡量困惑（而非 total loss 中的 KL 部分）
kl_val = self._kl_value if self._kl_value > 0 else 0.0
coherence = 1.0 - math.tanh(loss_val)
# priority = 基础新颖度 + KL 困惑度 (分开加权)
novelty = 1.0 - coherence  # 低 coherence = 高困惑 = 高新颖度
# 但 priority 应该用 loss_val (总体表现) 而非 KL 来决定采样优先级
priority = coherence + 0.5 * (1.0 - coherence)  # coherence 高 = 应该被采样
self._replay_buffer.add(
    self._u_next_input.squeeze(0).detach(),
    loss=loss_val,
    novelty=priority,  # 高 coherence = 高 priority = 优先蒸馏
)
```

这样高 coherence（低困惑）的状态被优先采样，stable 专家学到的是稳定知识。困惑状态仍然有非零 priority（0.5 权重），但不会被过度采样。

### 可行性

高。只需修改 Stage D 的 priority 计算逻辑。

---

## 4. P1 级隐患

### P1-1: 内部循环的 v_t 来源不稳定 — 辩证缓冲的 get_latest 优先 antithesis

`dialectical_buffer.py:105-120` 的 `get_latest()` 优先返回 antithesis 状态：

```python
if self._antitheses:
    return self._antitheses[-1]['vector']  # 优先困惑状态
if self._syntheses:
    return self._syntheses[-1]['vector']
if self._theses:
    return self._theses[-1]['vector']
```

这意味着每次内循环的初始状态 v_t 倾向于困惑状态，导致：
- SelfModel 被喂入困惑状态 -> 想象基于困惑 -> 意识流偏向困惑
- 内循环 loss 反映困惑状态 -> Hebbian 更新偏向困惑方向
- 系统级偏向混沌而非有序

这与 "混沌边缘" 哲学（有序与混沌之间的平衡）有偏差。系统应该有 **周期性**：有时处理困惑（antithesis），有时处理稳定（thesis），有时处理中间态。

**改进方向**：在 `get_latest()` 中加入周期性轮换，而非总是优先 antithesis：

```python
def get_latest(self):
    # 轮换: 70% antithesis, 20% synthesis, 10% thesis
    # 保持混沌边缘振荡而非偏向混沌
    import random
    r = random.random()
    with self._lock:
        if r < 0.7 and self._antitheses:
            return self._antitheses[-1]['vector']
        elif r < 0.9 and self._syntheses:
            return self._syntheses[-1]['vector']
        elif self._theses:
            return self._theses[-1]['vector']
        elif self._antitheses:
            return self._antitheses[-1]['vector']
        elif self._syntheses:
            return self._syntheses[-1]['vector']
    return None
```

### P1-2: FluidRoles 的 sigma 阈值与 verify_threshold 耦合

`fluid_roles.py:39-40`:

```python
self.stable_threshold = config.verify_threshold * 0.3
self.plastic_threshold = config.verify_threshold * 1.5
```

当 verify_threshold=0.5 时：
- stable_threshold = 0.15
- plastic_threshold = 0.75

但专家的 `avg_uncertainty` 是 sigmoid 输出，范围 [0, 1]，均值通常在 0.3-0.7 之间。0.75 的 plastic_threshold 意味着很少专家会被 promote 到 plastic。这导致 stable:plastic 边界过于刚性，"fluid roles" 名不副实。

**改进方向**：将阈值与 sigma 的历史分布挂钩，而非固定于 verify_threshold：

```python
# 用 sigma 的 EMA 分位数代替固定阈值
self.stable_threshold = self._sigma_p25  # 25th percentile
self.plastic_threshold = self._sigma_p75  # 75th percentile
```

### P1-3: 内循环的 `_get_state` 依赖 model._h_state，但 external forward 不更新它

`internal_loop.py:235-237`:

```python
v_t = self._get_state()
if v_t is None:
    v_t = self._init_state()
```

`_get_state` 返回 `model._h_state`（如果有）或 `endosphere.get_latest()`。但 `model._h_state` 只在内循环的 SelfModel 更新时设置（`internal_loop.py:273`）。在外部 forward（用户交互）中，`model._h_state` 不更新。

这意味着：用户交互后，内循环的下一个 v_t 仍然是之前的 SelfModel 状态或 endosphere 中的旧状态，而不是用户交互产生的新状态。意识流没有跟上外部交互的节奏。

**改进方向**：在外部 forward 中也更新 `model._h_state`（用最后一层的 hidden state）：

```python
# model.py forward 末尾:
x = self.ln_f(x)
# 用最后一层 hidden state 更新内循环状态
self._h_state = x.mean(dim=1).detach()  # [B, d_model] -> [1, d_model] for single
self._last_layer_outputs = {'hidden_states': x}
```

### P1-4: 内部循环的 pred_emb 用于 Hebbian 更新，但 pred_emb 是 forward_internal 的输出，不是困惑区域的输出

`internal_loop.py:283-285`:

```python
pred_emb = self.model.forward_internal(
    u_next_input, h_state=h_next
)
loss_int = self._compute_loss(pred_emb, v_t)
```

`forward_internal` 用 `u_next_input`（SelfModel 想象的 embedding）作为输入，经过全部 24 层。然后 `autograd.grad(loss_int, expert._output)` 计算所有激活专家的梯度。

但困惑专家的 "困惑" 是基于**用户输入**产生的，而 `pred_emb` 是基于**SelfModel 想象**产生的。这两个输入不同：
- 用户输入导致困惑 -> 需要纠正
- SelfModel 想象导致梯度 -> 不是困惑的方向

Hebbian 更新的梯度来自 SelfModel 想象的 forward，而非用户困惑的 forward。这意味着 Hebbian 更新的方向是 "让 SelfModel 的想象更准确"，而非 "纠正用户的困惑"。

这与 P0-A 的 CE loss 不对齐问题类似，但更深层：**内循环的梯度源与用户困惑源不共享同一个前向计算图**。

**改进方向**：在 `_compute_loss` 中，将用户输入的 forward 与 SelfModel 想象的 forward 结合，让梯度同时流向两个方向：

```python
# 结合用户输入的 forward (困惑源) 和 SelfModel 的 forward (想象源)
if hasattr(self.model, '_last_user_emb'):
    user_emb = self.model._last_user_emb
    # 让 pred_emb 接近 user_emb (让想象扎根于用户困惑)
    grounded_loss = F.mse_loss(pred_emb, user_emb.detach())
    loss_int = loss_int + grounded_weight * grounded_loss
```

### P1-5: InteractionManager 的 emergent cooldown 依赖 sigma 下降，但 sigma 在部署初期不稳定

`manager.py:77-92` 的 `can_ask`：

```python
return (self._state == self.IDLE
        and self._questions_this_session < self._max_questions_per_session
        and self._current_model_sigma > self._sigma_threshold)
```

部署初期，SelfModel/Critic 从随机初始化启动（我们已通过 Phase 2.5 修复），sigma 的估计不准确。这意味着：
- 初期 sigma 可能一直低于阈值 -> 从不提问
- 或一直高于阈值 -> 频繁提问（超过 5 次/会话限制）

即使有 Phase 2.5，Critic 的 pseudo_reward 仍然是粗略估计，sigma 的校准需要时间。

**改进方向**：在部署的前 N 步，使用一个 **保守的硬编码 cooldown**（如每 50 步最多问 1 次），等 sigma 稳定后切换到 emergent 模式：

```python
@property
def can_ask(self):
    with self._lock:
        # Warm-up period: use conservative cooldown
        if self._total_steps < 500:
            return (self._state == self.IDLE
                    and self._questions_this_session < self._max_questions_per_session
                    and self._last_question_step is None
                    or (self._total_steps - self._last_question_step) > 50)
        # Emergent: sigma-driven
        return (self._state == self.IDLE
                and self._questions_this_session < self._max_questions_per_session
                and self._current_model_sigma > self._sigma_threshold)
```

### P1-6: 训练数据的文本被 `len(text) > 200` 过滤，但 Wikipedia 有很多短条目

`prepare_data.py` 的 Wikipedia 提取：

```python
if len(text) > 100:
    yield {"text": text}
```

SkyPile/CWT/FineWeb 的过滤是 `len(text) > 200`。Wikipedia 的过滤是 `len(text) > 100`。Wikipedia 有很多短条目（stub articles），这些被过滤掉了，但它们可能包含有用的定义性知识。

**改进方向**：对 Wikipedia 使用更低的阈值（如 50）并添加 title prefix 帮助模型理解短文本的上下文：

```python
if len(body) > 50:
    text = f"{title}\n\n{body}"
    yield {"text": text}
```

### P1-7: Hebbian 更新中 bias 更新的 update_scale 未 clamp

`hebbian_update.py:61-66` (bias 更新部分):

```python
if expert.w_down.bias is not None:
    delta_b_down = grad_y_2d.sum(dim=0)
    expert.w_down.bias.data -= update_scale * delta_b_down
```

`delta_b_down = grad_y_2d.sum(dim=0)` 没有 norm clamp 也没有 momentum。bias 更新可能比 weight 更新更大（因为 sum(dim=0) 聚合了所有 token 的梯度）。

**改进方向**：对 bias 更新也加 momentum 和 clamp：

```python
if expert.w_down.bias is not None:
    delta_b_down = grad_y_2d.sum(dim=0)
    delta_b_down = _apply_momentum(expert, '_mom_b_down', delta_b_down)
    d_norm = delta_b_down.norm()
    if d_norm > _MAX_DELTA_NORM:
        delta_b_down = delta_b_down * (_MAX_DELTA_NORM / d_norm)
    expert.w_down.bias.data -= update_scale * delta_b_down
```

但需要为 bias 也添加 `_mom_b_down/_mom_b_gate/_mom_b_up` momentum buffer。

---

## 5. P2 级优化点

### P2-1: 内部循环中 `_loss_history` 无锁读写

`internal_loop.py:343-344` 在锁外 append `_loss_history`，而 `global_drift` 等属性在锁内读取。虽然 CPython GIL 保护了 list.append 的原子性，但 `_loss_history[-5000:]` 的切片赋值不是原子的，可能与其他线程的 append 冲突。

**改进**：将 `_loss_history` 的 append 也移入 `model._lock` 内。

### P2-2: `_generate_emergent_question` 的 generate 在 `model._lock` 内运行，阻塞外循环

`internal_loop.py:545-553` 的 `generate` 在 `model._lock` 内运行，这意味着当内循环生成问题时，外循环（用户交互）被阻塞。用户可能会感觉到延迟。

**改进**：将问题生成移到锁外，只将结果提交到 `InteractionManager`。

### P2-3: ConfusionMap 的 MD5 hash 不支持语义分组

`confusion_map.py:128-137` 用 MD5 哈希文本（前 50 字符小写）。语义上相近但措辞不同的困惑（如 "量子力学是什么" vs "什么是量子力学"）被分到不同的 key，无法聚合。

**改进**：用 embedding 余弦相似度做近重复检测，而非精确字符串匹配。但计算成本高，适合 P2。

### P2-4: `feedback.py` 的 alignment 阈值 0.3 是硬编码

`feedback.py:89`:

```python
if alignment < 0.3:
    return {'reward': reward, 'focal_boost': None}
```

0.3 的阈值没有经过验证。如果用户答案与困惑文本语义相关但措辞不同（如 "量子不是粒子而是场" vs "量子是粒子吗？"），cosine similarity 可能低于 0.3（因为它们不包含相同的 token）。这导致很多有效的用户反馈被丢弃。

**改进**：降低阈值到 0.2 或用余弦相似度的分位数而非固定值。

---

## 6. 修复优先级

```
Phase 1 (P0 - 影响训练结果):
  P0-A: 求正通路 CE loss 不对齐 -> 添加对齐损失
  P0-B: Sigma 梯度截断 -> 添加 sigma 校准损失
  P0-C: Replay priority 被 KL 主导 -> 解耦 priority 计算

Phase 2 (P1 - 影响质量上限):
  P1-1: get_latest 偏向 antithesis -> 周期性轮换
  P1-3: model._h_state 在外部 forward 中不更新 -> 同步更新
  P1-4: 内循环梯度源与用户困惑源不共享 -> grounded loss
  P1-7: bias 更新无 momentum/clamp -> 添加 momentum+clamp

Phase 3 (P2 - 优化):
  P1-2: FluidRoles 阈值刚性 -> 分位数阈值
  P1-5: emergent cooldown 不稳定 -> warmup 期保守模式
  P2-4: alignment 阈值 0.3 -> 降到 0.2
  P2-1: _loss_history 锁外读写 -> 移入锁内
```

---

*本文档作为第二轮修复的依据。每个 Phase 完成后应回归测试 127 项验证。*
