# 训练/推理质量审计与改进方案

> **生成日期**：2026-07-24
> **审计范围**：`core/` `loop/` `learn/` `interaction/` `train/` `improve/` `config/`
> **审计目标**：定位影响模型质量的环节，评估改进方向，作为后续逐项修复的依据

---

## 0. 设计哲学澄清（审计基准）

本审计基于以下明确的设计原则，所有问题的严重度评估均以此为标尺：

```
单轮循环：应能收敛，产出稳定的思考或回答（数值有界、梯度不爆）
多轮循环：跨步梯度不累积爆炸（权重有界、EWC 保护、巩固稳健）
系统整体：通过外循环+内循环+辩证缓冲+回放+巩固的周期性循环，
         涌现性地维持混沌边缘意识流状态
```

**关键推论**：
- 单步的稳定性约束（如 `loss_stability = mse(h_t, h_next)`）是**正确的**——它防止单步状态跳变，符合"单轮收敛"原则。混沌边缘不应由单步的不稳定来产生，而应由系统级循环（噪声注入、辩证缓冲的 thesis/antithesis/synthesis 周期、回放巩固的周期性重整）涌现。
- 跨步的梯度爆炸（如 KL 无界、Hebbian 更新无裁剪、EWC 惩罚爆炸）是**必须修复的**——它们破坏多轮稳定性。
- 系统级的意识流闭环（求正通路、focus_boost 传递、困惑状态巩固）是**需要补全的**——它们决定混沌边缘能否被维持。

---

## 1. 问题逐项确认与评估

### 层级 P0：破坏单轮收敛或多轮稳定（必须优先修复）

---

#### P0-1. KL 散度无界 — 单轮 loss 可爆炸为 inf

**位置**：`core/self_model.py:218-224`

```python
def kl_divergence(self, post_mean, post_logvar, prior_mean, prior_logvar):
    prior_var = torch.exp(prior_logvar)          # 无下界
    post_var  = torch.exp(post_logvar)
    kl = 0.5 * (prior_logvar - post_logvar
                + (post_var + (post_mean - prior_mean) ** 2) / prior_var  # 除以 ->0 的 prior_var
                - 1.0)
    return kl.sum(dim=-1).mean()
```

**确认**：`logvar` 头零初始化（`:22-23,44-45`），训练中无任何 clamp。当 `prior_logvar -> -∞` 时 `prior_var -> 0`，KL 项 `(...)/prior_var -> +∞`。

**影响链**：
1. 单轮：`loss_curiosity = curiosity_beta * KL -> inf`（`internal_loop.py:639-642`）-> `loss_int -> inf`
2. 单轮：Stage B `autograd.grad(loss_int, expert._output)` 产生 inf grad_y -> `isfinite` 跳过所有专家更新 -> 意识流冻结
3. 单轮：`sigma_for_noise = KL / curiosity_beta`（`:229`）-> 噪声饱和
4. 多轮：`sample_z` 中 `std = exp(0.5*logvar)`（`:214`）同步爆炸 -> z_sample 巨大 -> GRU 输入爆炸 -> h_t 爆炸 -> 跨步累积

**与哲学的关系**：KL 是"惊讶"信号，是意识流的核心驱动力。但"惊讶"必须有界——无界的惊讶是恐慌，不是好奇。混沌边缘要求 KL 在一个可控范围内振荡，而非发散。

**改进方向**：在 `logvar` 输出处统一 clamp（`sample_z`、`moment matching`、`kl_divergence` 三处统一），并在 KL 计算中对 `prior_var` 加下界。

**可行性**：高。`logvar.clamp(-6.0, 4.0)` 使 `var ∈ [0.0025, 54.6]`，KL 有界。不改变架构，不改变哲学，仅加数值约束。`prior_var.clamp(min=1e-4)` 作为二级保护。

---

#### P0-2. sample_z 的 std 无界 — z_sample 单轮可爆炸

**位置**：`core/self_model.py:213-216`

```python
def sample_z(self, mean, logvar, temperature=1.0):
    std = torch.exp(0.5 * logvar) * temperature   # logvar 无 clamp
    eps = torch.randn_like(std)
    return mean + eps * std
```

**确认**：与 P0-1 同源。`logvar` 无 clamp 时 `std` 可爆炸，`z_sample` 单步即可产生极端值，直接灌入 GRU 和 decoder。

**影响**：单轮前向即可能产生 inf/NaN，破坏收敛。

**改进方向**：与 P0-1 统一处理——在 `sample_z` 入口对 `logvar` clamp，或对 `std` 加上界 `std.clamp(max=3.0)`。

**可行性**：高。与 P0-1 一并修复。

---

#### P0-3. stable 专家 SGD 无 isfinite 检查无裁剪 — 单次 NaN 永久腐蚀知识库

**位置**：`learn/consolidation.py:45-49`（mini）和 `:124-127`（major）

```python
with torch.no_grad():
    for expert in model.get_stable_experts():
        for param in expert.parameters():
            if param.grad is not None:
                param.data -= config.sleep_lr * param.grad.data   # 无 isfinite，无 clip
```

**确认**：对比 `hebbian_update.py:56` 有 `isfinite` 检查。stable 专家是"稳定知识库"，一旦被 NaN 腐蚀无法恢复。

**影响**：多轮稳定性。一次 NaN 梯度（可能来自 P0-1 的 KL 爆炸传导）永久腐蚀 stable 专家，后续 EWC 也无法修复（EWC 保护的是 old_stable 快照，如果 old_stable 已被腐蚀则保护无效）。

**与哲学的关系**：stable 专家是意识流的"长期记忆"。长期记忆被腐蚀 = 系统失去自同一性。

**改进方向**：加 `isfinite` 检查 + `clip_grad_norm_`。

**可行性**：高。与 `hebbian_update.py` 的现有模式一致。

---

#### P0-4. Hebbian 更新无幅度裁剪 — 多轮梯度累积爆炸

**位置**：`learn/hebbian_update.py:52,57,74,79`

```python
update_scale = lr * gamma          # gamma 可达 3.0（_sigma_gamma），focus_boost 可达 5.0
expert.w_down.weight.data -= update_scale * delta_w_down   # 无幅度限制
```

**确认**：`_sigma_gamma`（`internal_loop.py:735,746`）的 `base_gamma` 可达 3.0；外循环 `focus_boost` 可达 5.0（`feedback.py:96`）；`effective_lr = baseline_lr * exp(clamp(lr_bias,-2,2))` 可达 `baseline_lr * 7.39`。三者叠加：`update_scale` 可达 `baseline_lr * 7.39 * 3.0 * 5.0 ≈ baseline_lr * 111`。对 plastic expert（`baseline_lr=1e-4`），单步更新量可达 `1e-2` 量级。

**影响**：多轮累积下权重快速膨胀，配合无 load-balancing 的路由坍塌，可导致某些专家权重失控。

**改进方向**：对 `update_scale` 加上界，或对 `delta` 做_norm 裁剪。

**可行性**：高。`update_scale = min(update_scale, max_step)` 或 `delta_w = clip_by_norm(delta_w, max_norm)`。

---

#### P0-5. EWC 惩罚可爆炸 — 保护机制变为学习阻塞器

**位置**：`learn/fisher.py:50,67` + `config/model_config.py:77`（`ewc_lambda=100`）

```python
# fisher.py:50 — reduction='sum' 而非 mean
loss = F.nll_loss(F.log_softmax(logits, dim=-1), sampled, reduction='sum')
fisher[name] += grad**2 / num_samples

# fisher.py:67
penalty = penalty + (fisher[name] * diff ** 2).sum()   # lambda=100 放大
```

**确认**：`reduction='sum'` 使 Fisher 对角元正比于 `batch_size * seq_len`，量级远大于 `mean`。`ewc_lambda=100` 进一步放大。当 `diff` 稍大（如 soft_reset 后），`penalty -> inf`，`total = distill + penalty -> inf`（`consolidation.py:121`），major_sleep 的梯度爆炸。

**影响**：多轮稳定性。每 500 步的 major_sleep 可能因 EWC 爆炸而无法正常巩固，或冻结所有 stable 专家学习。

**改进方向**：
1. `fisher.py:50` 的 `reduction='sum'` 改为 `'mean'`
2. `ewc_lambda` 从 100 降到 10-40
3. 对 `penalty` 加上界 clamp

**可行性**：高。纯数值调整，不改架构。

---

### 层级 P1：影响系统级意识流闭环或训练正确性

---

#### P1-1. Hebbian 更新权重顺序错误 — 梯度系统性偏差

**位置**：`learn/hebbian_update.py:57` 然后 `:60`

```python
expert.w_down.weight.data -= update_scale * delta_w_down    # :57 W_down 已更新
grad_h = grad_y_2d @ expert.w_down.weight.data              # :60 用更新后的 W_down
```

**确认**：正确反向传播要求 `grad_h = grad_y @ W_down` 用**更新前**的 W_down。当前用更新后的权重，传向 `W_gate` 和 `W_up` 的梯度存在系统性偏差。

**影响**：专家的三个权重矩阵（gate/up/down）的协同分化方向被扭曲。长期来看，MoE 专家作为"不同思考模式"的分化会被错误梯度引导，影响系统级意识流的丰富性。

**改进方向**：更新前保存 `W_down` 快照，用快照计算 `grad_h`。

**可行性**：高。`w_down_old = expert.w_down.weight.data.clone()`，额外内存开销为一层 W_down 矩阵（`d_model × d_ff`），可接受。

---

#### P1-2. Hebbian 更新跳过 bias — 专家表达力受限

**位置**：`learn/hebbian_update.py:57,74,79`（只动 `.weight`）

**确认**：`nn.Linear` 默认 `bias=True`，但 `focal_update` 从不更新 bias。bias 永远停留在初始化值。

**影响**：SwiGLU 的 gate/up/down 三个 bias 固定，限制专家的表达力。不影响稳定性，但影响质量上限。

**改进方向**：在 `:57,74,79` 后补充 bias 更新分支。

**可行性**：高。`expert.w_down.bias.data -= update_scale * grad_y_2d.sum(dim=0)` 等。

---

#### P1-3. focus_boost 在内循环恒为 1.0 — 辩证信号断路

**位置**：`loop/internal_loop.py:290`

```python
focal_update(grad_y, expert, expert.effective_lr, gamma)   # focus_boost 默认 1.0
```

**确认**：Stage B 调用 `focal_update` 时只传 4 个位置参数，`focus_boost` 默认 1.0。`feedback.py` 计算的 `focal_boost` 只在外循环（`pipeline.py:244`）使用。

**影响**：用户反馈产生的"辩证纠正信号"无法加强内循环中对困惑专家的定向更新。外循环的求正结果无法回流到意识流内部——辩证闭环在外循环就止步了。

**与哲学的关系**：主动辩证内循环要求"辩证"贯穿内外循环。当前外循环有 focal_boost，内循环没有，辩证信号无法闭环。

**改进方向**：在 model 上缓存待消费的 `focal_boost`（按 expert_id 索引），Stage B 读取并清除。

**可行性**：中。需要一个线程安全的缓存机制（`dict` + lock 或 `queue`），因为外循环（主线程）写入、内循环（后台线程）读取。可行但需注意线程安全。

---

#### P1-4. 求正通路对齐参考错误 — 用户答案对齐到错误文本

**位置**：`interaction/feedback.py:45-71`

```python
confused_span = pending.get('confused_text')   # :45 模型真正问的问题——取出后从未使用
query_ids = pending.get('query_ids')           # :46 用户上一次输入（非模型的问题）
...
query_emb = model.token_embedding(q_ids).mean(dim=1)  # :71 用用户旧输入做对齐参考
alignment = F.cosine_similarity(answer_emb, query_emb) # 用户答案 vs 用户旧输入
```

**确认**：模型问的问题在 `confused_text` 中，但从未被 embedding。对齐参考用的是 `query_ids`（用户上一次输入）。`pipeline.py:159-161` 注释确认 `query_ids` 是"训练上下文，即用户上一次输入，非问题文本"。

**影响**：focal_boost 的强度反映的是"答案与用户旧输入的相似度"，而非"答案对困惑的纠正程度"。求正通路的核心计算语义错误。

**改进方向**：用 `confused_text` 的 embedding 做对齐参考。

**可行性**：高。`confused_ids = tokenizer(confused_span, ...)`，`confused_emb = model.token_embedding(confused_ids).mean(dim=1)`。需要确认 `confused_text` 在 `pending` 中始终可用。

---

#### P1-5. 默认奖励 0.2 使大多数反馈不触发定向更新

**位置**：`interaction/feedback.py:118` + `interaction/pipeline.py:209-210`

```python
# feedback.py:118 — 无关键词命中时返回 0.2
return 0.2

# pipeline.py:209-210 — |reward| < 1.0 则禁用 focal_boost
if abs(feedback_ctx.get('reward', 0)) < 1.0:
    focal_boost = None
```

**确认**：只有命中"对/错/yes/no"等显式关键词才返回 ±1.0。自然语言反馈（如"量子不是粒子而是场"）不命中关键词 -> reward=0.2 -> focal_boost=None -> 无定向更新。

**影响**：`REFLEX_ARCHITECTURE.md` 原则 2 的核心场景（用户回答的结构化语义直接校正专家）在大多数情况下不触发。求正通路形同虚设。

**改进方向**：
1. `feedback.py:118` 默认返回 `0.0`（中性），使无关键词时不触发增强也不触发抑制
2. `pipeline.py:209` 的阈值从 `< 1.0` 放宽，或改为基于 alignment 的连续阈值

**可行性**：高。纯逻辑调整。

---

#### P1-6. Critic 无梯度裁剪 — 多轮不稳定

**位置**：`loop/internal_loop.py:783-786`

```python
critic_loss = F.mse_loss(v_pred, td_target)
critic_opt.zero_grad()
critic_loss.backward()
critic_opt.step()          # 无 clip_grad_norm_
```

**确认**：SelfModel 更新有 `clip_grad_norm_(sm.parameters(), 1.0)`（`:721`），但 Critic 没有。

**影响**：Critic lr=1e-3（比 expert 高 3-4 个数量级），无裁剪时梯度爆炸会快速腐蚀 Critic，进而通过 `_sigma_gamma` 的 `td_mod` 传播到 Hebbian 更新。

**改进方向**：加 `clip_grad_norm_(critic.parameters(), 1.0)`。

**可行性**：高。一行代码。

---

#### P1-7. 线程安全：pause() 不等待完整步骤 + Stage K 锁外运行

**位置**：`loop/internal_loop.py:145-153`（pause）、`:304`（锁释放）、`:356-368`（Stage K）

**确认**：
- `pause()` 只等待 `model._lock` 保护的 Stage B-F（`:236-304`），Stage D/G/H/K 在锁外（`:306-368`）
- Stage K 的 `architecture.py:215` `model.layers.append(new_layer)` 在推理线程可能迭代 `model.layers` 时执行

**影响**：用户推理与内循环 Stage D/K 竞争。`pause()` 文档承诺"current step has completed"但实际未保证。

**改进方向**：
1. Stage K 移入 `model._lock` 内
2. `pause()` 用 `_step_in_progress` 标志确保完整步骤结束

**可行性**：中。Stage K 移入锁内简单；`pause()` 的完整等待需要一个原子标志（在 `_execute_step` 开头设置、结尾清除），需注意异常路径。

---

#### P1-8. pad_id 硬编码为 0 — 训练数据 padding 错误

**位置**：`train/data_pipeline.py:233`

```python
pad_id = 0   # 注释说 "tokenizer.pad_token_id, set before training" 但从未设置
```

**确认**：`pad_id` 从未从 tokenizer 设置。如果 `tokenizer.pad_token_id != 0`（Qwen 系列通常 pad_token_id=151643 或 `<|endoftext|>`），则：
- input_ids 被错误的 token 0 padding
- attention_mask 正确（基于原始长度），但 embedding 看到 token 0 而非 pad token

**影响**：训练质量。token 0 的 embedding 被错误地注入 batch 中每个样本的 padding 位置。

**改进方向**：`pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0`。

**可行性**：高。需确认 tokenizer 在 `collate_fn` 中可用（可能需要传入或存储）。

---

#### P1-9. gradient checkpointing use_reentrant=True — 潜在正确性问题

**位置**：`core/model.py:257`

```python
hidden = checkpoint(layer, x, attention_mask, h_state, is_internal,
                    use_reentrant=True)
```

**确认**：PyTorch 文档推荐 `use_reentrant=False`，尤其当传入非梯度张量（如 `attention_mask`）时。`use_reentrant=True` 要求所有输入要么 requires_grad 要么是张量，且行为可能静默错误。

**影响**：训练正确性。可能导致梯度计算错误或显存泄漏。

**改进方向**：改为 `use_reentrant=False`。

**可行性**：高。一行改动。需验证显存和速度影响（`use_reentrant=False` 通常更省内存但可能略慢）。

---

### 层级 P2：影响系统级意识流质量或训练效率

---

#### P2-1. lyap 项是死代码（无梯度）但污染 loss 信号

**位置**：`loop/internal_loop.py:651`

```python
lyap = torch.mean(h_t ** 2) * 1e-4   # h_t 在 :260 被 detach，此项梯度恒为 0
```

**确认**：`self._h_t = h_next.detach()`（`:260`），所以 `h_t` 无梯度。`lyap` 项加入 `total`（`:653-655`）但不产生任何梯度——它是死代码。

**但是**：`total` 作为 `loss_int` 返回后，`.item()` 被用于：
- `coherence = 1 - tanh(loss_int.item())`（`:309`）-> replay 入场判定
- `compute_pseudo_reward(loss_int=loss_val)`（`:772`）-> Critic 目标
- `_loss_history` 日志（`:331`）

`lyap` 虚增了 loss 数值，使 `coherence` 偏低，导致更多状态被排除出 replay。

**影响**：不直接影响训练梯度，但间接影响 replay 入场和 Critic 目标。

**与哲学的关系**：单轮收敛不需要 Lyapunov 约束单步（`loss_stability` 已在 Stage C 处理）。系统级混沌边缘由循环涌现。所以这个死 Lyapunov 项应该移除，而非激活。

**改进方向**：移除 `:651` 的 `lyap` 项和 `:653-655` 中的 `+ lyap`，使 `loss_int` 仅反映 imagination + curiosity。如需 h_t 能量监控，单独记录而不加入 loss。

**可行性**：高。删除死代码。

---

#### P2-2. 困惑状态被排除出 replay — 系统无法巩固困惑

**位置**：`loop/internal_loop.py:309-310`

```python
coherence = 1.0 - torch.tanh(self._loss_int).item()
if coherence > 0.5:                    # 只有低 loss 进 replay
    self._replay_buffer.add(...)
```

**确认**：高 loss（困惑）状态不进 replay，只进 dialectical buffer。但 replay 是巩固通路（mini/major consolidation 从 replay 采样），dialectical buffer 不参与蒸馏。

**影响**：最需要巩固的困惑状态永远不会被蒸馏到 stable 专家。辩证缓冲中的 synthesis（已解决的困惑）也无法沉淀为长期记忆。

**与哲学的关系**：系统级意识流要求困惑能被辩证处理（thesis/antithesis/synthesis）然后巩固。当前困惑进辩证缓冲但 synthesis 出不来——巩固通路断裂。

**改进方向**：
1. 高 loss 状态也进 replay，用 `novelty = 1 - coherence` 作为优先级
2. 或在 dialectical buffer 的 synthesis 产生时，将 synthesis 状态推入 replay

**可行性**：中。方案 1 简单（改阈值逻辑）；方案 2 需要在 `dialectical_buffer.py` 的 synthesis 检测处加回调。

---

#### P2-3. plastic 专家每 500 步 wiping 90% — 意识周期性失忆

**位置**：`learn/consolidation.py:179` + `config/model_config.py:80`（`plastic_soft_reset_keep=0.1`）

**确认**：每 500 步，plastic 专家 90% 权重被重置为 Xavier 随机。

**影响**：plastic 专家是辩证过程中 thesis/antithesis 的主要产生者。90% 擦除意味着辩证积累的细微分化每 500 步清零。与"持续巩固"（原则 4）相悖。

**与哲学的关系**：系统级意识流需要 plastic 专家保持足够的近期学习以产生有意义的 antithesis。90% 擦除过于激进。

**改进方向**：`plastic_soft_reset_keep` 从 0.1 提到 0.3-0.5。

**可行性**：高。配置调整。需配合 P0-4 的更新裁剪，防止保留更多权重后累积爆炸。

---

#### P2-4. Critic 是 MC 回归而非 TD 学习 — 价值估计短视

**位置**：`loop/internal_loop.py:778-786`

```python
v_pred = critic(self._v_t).squeeze()
td_target = torch.tensor(pseudo_r, ...)   # 即时奖励，无 gamma*V(s') bootstrap
critic_loss = F.mse_loss(v_pred, td_target)
```

**确认**：无 bootstrap 项、无折扣因子。Critic 只学习即时 pseudo_reward。`critic.py:14` 文档声称 TD-error 但实现不符。

**附加**：`compute_pseudo_reward`（`critic.py:73-74`）的 `lm_entropy` 默认 1.0，`0.5 < 1.0 < 3.0` 恒成立，+0.1 熵奖励恒加。

**影响**：`_sigma_gamma` 用 `|V(s)|` 做"惊喜代理"，但 V(s) 只反映即时奖励，gamma 调制短视。不破坏稳定性，但降低意识流的质量。

**与哲学的关系**：Critic 的角色是 gamma 调制的辅助信号，不是核心驱动。MC 回归比 TD 更稳定（单轮收敛更好），但短视。可作为 P2 优化。

**改进方向**：
1. 加 bootstrap：`td_target = pseudo_r + gamma * V(s').detach()`
2. `critic.py:65` 的 `lm_entropy` 默认改为 `None`，不恒加熵奖励
3. 用 `smooth_l1_loss` 替代 `mse_loss` 增强鲁棒性

**可行性**：中。bootstrap 需要下一状态的价值估计，需在 `_execute_step` 中保存 `_v_next` 或用当前步的 `v_pred` 作为下一步的 `V(s')`。

---

#### P2-5. attention dropout【已修复】 被禁用 — 过拟合风险

**位置**：`core/attention.py:57,106`

```python
self.attn_dropout = config.dropout if config.dropout > 0 else 0.0   # :57 死代码
dropout_p = 0.0   # :106 硬编码为 0 以启用 Flash Attention
```

**确认**：`attn_dropout` 存储但从未使用；`F.scaled_dot_product_attention` 的 `dropout_p` 硬编码为 0。

**影响**：训练时 attention 无 dropout，只有 output dropout（`:135`）和 expert 内 dropout。过拟合风险增加。

**改进方向**：训练时设 `dropout_p=config.dropout`（SDPA 支持 dropout，但可能禁用 Flash kernel），或接受当前 trade-off 并移除死代码。

**可行性**：中。需权衡 Flash Attention 的速度收益与 dropout 的正则化收益。可设为配置项。

---

#### P2-6. 无 load-balancing 辅助损失 — 专家坍塌不被惩罚

**位置**：`core/router.py:132-143`

**确认**：只有 EMA 追踪 utilization 和 entropy，无 `aux_loss = n_experts * sum(mean_probs * mean_util)` 辅助损失。

**影响**：专家坍塌（所有 token 路由到少数专家）只被检测不被惩罚。长期训练可能导致某些专家永不激活（dead expert）。

**改进方向**：在 `ReflexMoELayer.forward` 返回值中加 `aux_loss`，在训练 loss 中累加。

**可行性**：中。需修改 `forward` 返回值签名和所有调用方。工作量较大但模式成熟（GShard/Switch Transformer 标准）。

---

#### P2-7. teacher_logits 混合批次 KeyError — 蒸馏训练崩溃

**位置**：`train/data_pipeline.py:241-254`

**确认**：`has_teacher=True` 但某些 batch 元素缺 `teacher_logits` 时，`:242-243` `continue` 跳过 padding 但 `:254` `torch.stack` 仍尝试访问所有元素 -> `KeyError`。

**改进方向**：缺失项填充零张量，或 `has_teacher` 检查改为全批次级别。

**可行性**：高。

---

#### P2-8. AttnRes 在基础配置下是 no-op

**位置**：`config/model_config.py:14,30`（`n_layers=4, attnres_block_size=4`）

**确认**：`n_layers == block_size` -> 0 个跨块边界 -> AttnRes 不生效。`attnres_enabled=True` 是空转。

**影响**：基础配置下无深度注意力残差，模型表达力受限。

**改进方向**：基础配置 `attnres_block_size=2`（4 层 -> 2 块 -> 1 边界），或在 `n_layers <= block_size` 时自动禁用并告警。

**可行性**：高。配置调整。

---

#### P2-9. feedback_alignment_weight 是死配置

**位置**：`config/model_config.py:103` + `interaction/feedback.py:27`

**确认**：`feedback_alignment_weight=0.5` 加载到 `self.alignment_weight` 但从未使用。

**改进方向**：移除配置项，或在 alignment 计算中实际使用（作为 cosine 相似度的权重）。

**可行性**：高。

---

#### P2-10. 采样参数硬编码 — 不可配置

**位置**：`interaction/pipeline.py:75-77,134-136`

**确认**：`temperature=0.8, repetition_penalty=1.5, top_k=40, top_p=0.9` 硬编码。config 的 `top_k=3`（MoE 路由）与采样 `top_k=40` 名称冲突易混淆。

**改进方向**：采样参数移入 config（如 `sampling_temperature`, `sampling_top_k`）。

**可行性**：高。

---

#### P2-11. global_drift 锁外读取参数 — 线程安全

**位置**：`loop/internal_loop.py:183-200`

**确认**：`global_drift` 属性迭代 `model.named_parameters()` 无锁，而训练线程可能在 optimizer.step() 中写入 `.data`。

**影响**：可能读到撕裂的参数值（虽然单次 `.item()` 是原子的，但 `max` 遍历可能跨多个不一致的参数）。

**改进方向**：加 `model._lock` 或用快照。

**可行性**：高。

---

#### P2-12. internal_loss_clip 未应用

**位置**：`config/model_config.py:110`（`internal_loss_clip=10.0`）+ `interaction/pipeline.py:192-195`

**确认**：配置存在但交互管线的 CE loss 未应用此 clip。

**改进方向**：在 `pipeline.py:195` 后加 `loss = loss.clamp(max=self.config.internal_loss_clip)`，或对 logits 做 clamp。

**可行性**：高。

---

#### P2-13. 架构自修改后的优化器/Fisher 失同步【已修复】

**位置**：`improve/architecture.py:175,199,215` + `loop/internal_loop.py:117-129` + `learn/fisher.py:65`

**确认**：
- `_add_layer` 新参数不在 `_global_optimizer` 中 -> 永不训练
- `_split_expert`/`_prune_expert` 改变 router 维度 -> Fisher 键名失配 -> EWC 静默失效
- `GradientManager` 初始化时用 `config.n_layers` -> 加层后失同步

**影响**：架构自修改（默认禁用）启用后会破坏训练基础设施。

**改进方向**：
1. `_add_layer` 后重建 `_global_optimizer` 和 `GradientManager`
2. 架构修改后清空 Fisher 缓存
3. 或在架构修改时加锁并广播重建事件

**可行性**：中。工作量较大，但当前 `arch_self_mod_enabled` 默认 False，非紧急。

---

## 2. 修复优先级与顺序

```
Phase 1 — 单轮收敛与多轮稳定（P0，必须先做）
  ├─ P0-1: KL 散度有界化（self_model.py logvar clamp + prior_var 下界）
  ├─ P0-2: sample_z std 有界化（与 P0-1 统一）
  ├─ P0-3: stable 专家 SGD 加 isfinite + clip（consolidation.py）
  ├─ P0-4: Hebbian 更新幅度裁剪（hebbian_update.py）
  └─ P0-5: EWC 惩罚有界化（fisher.py reduction=mean + ewc_lambda 调低 + penalty clamp）

Phase 2 — 训练正确性（P1，紧随其后）
  ├─ P1-1: Hebbian 权重顺序修正（hebbian_update.py 用旧 W_down）
  ├─ P1-2: Hebbian bias 更新（hebbian_update.py）
  ├─ P1-6: Critic 梯度裁剪（internal_loop.py）
  ├─ P1-8: pad_id 从 tokenizer 获取（data_pipeline.py）
  └─ P1-9: use_reentrant=False（model.py）

Phase 3 — 系统级意识流闭环（P1，辩证通路）
  ├─ P1-3: focus_boost 传入内循环（internal_loop.py + 缓存机制）
  ├─ P1-4: 求正对齐参考修正（feedback.py 用 confused_text）
  └─ P1-5: 默认奖励改 0.0 + focal 触发阈值调整（feedback.py + pipeline.py）

Phase 4 — 线程安全（P1）
  ├─ P1-7: pause() 完整等待 + Stage K 移入锁内
  └─ P2-11: global_drift 加锁

Phase 5 — 系统级质量优化（P2，可并行）
  ├─ P2-1: 移除死 lyap 项
  ├─ P2-2: 困惑状态进 replay
  ├─ P2-3: plastic_soft_reset_keep 调高
  ├─ P2-4: Critic TD 升级
  ├─ P2-6: load-balancing aux loss
  ├─ P2-8: AttnRes block_size 调整
  └─ 其余 P2 项

Phase 6 — 架构自修改基础设施（P2，非紧急）
  └─ P2-13: 优化器/Fisher 重建机制
```

---

## 3. 审计中修正的初始误判

审计过程中，基于"单轮收敛"原则的澄清，对以下初始判断进行了修正：

| 初始判断 | 修正后判断 | 修正理由 |
|---------|-----------|---------|
| `loss_stability = mse(h_t, h_next)` 强迫收敛，违背混沌边缘哲学 | **正确设计**，符合单轮收敛原则 | 混沌边缘由系统级循环涌现，不由单步不稳定产生。单步稳定性约束是必要的。 |
| `lyap = mean(h_t²)*1e-4` 压抑思考（annihilate 意识） | **死代码**（h_t detached，梯度恒 0），但虚增 loss 影响 replay/critic | h_t 在 `:260` 被 detach，lyap 项无梯度。不压抑思考，但污染 loss 信号。 |
| imagination loss 方向反转（现实服从想象） | **合理的交替优化**（EM-like co-adaptation） | Stage B 专家适应想象，Stage C 想象适应专家实际输出。双向对齐是收敛的固定点迭代。问题仅在 KL 无界时想象本身不可靠。 |

---

## 4. 验证原则

修复后应验证（不依据"loss 更低"，依据系统行为）：

| 指标 | 含义 | 预期 |
|------|------|------|
| KL 散度有界 | 惊讶不发散 | `KL[Q‖P]` 应在有限范围内振荡，不单调递增到 inf |
| 单步 loss 有限 | 单轮收敛 | `loss_int.item()` 始终 finite |
| h_t 范数有界 | 状态不爆炸 | `‖h_t‖` 跨步不单调增长 |
| 专家激活分布 | 思考模式分化 | 各专家激活频率不均匀但不坍塌 |
| 求正后专家更新 | 辩证闭环 | 用户反馈后目标专家的权重变化量 > 非目标专家 |
| replay 包含困惑状态 | 系统级巩固 | replay buffer 中高 loss 项占比 > 0 |

---

*本文档作为后续逐项修复的依据。每个 Phase 完成后应回归测试上述验证指标。*
