# NeuroStream-Reflex：设计哲学与实现忠实度整合分析报告

> **生成日期**：2026-08
> **分析方法**：通读 `docs/` 全部 12 份设计/审计文档 + 全部 60+ 源码文件 + 实证核查（加载 8.53GB 真实 checkpoint `sft_kd_150k_final.pt` 解剖参数、实跑内循环 6 步、实跑生成测试）
> **评估立场**：以项目自身设计哲学（主动辨证内循环意识流 / 双循环自指 / sigma 主动求证 / 混沌边缘 / 部署即学习）为标尺，评估"实现是否忠实于哲学"，而非以标准 LLM 训练范式评判。
> **前置声明**：本文是对此前一轮体检报告的整合与修正。若干"缺陷"判定在完整理解设计哲学后被修正为"设计意图"或"实现断裂"，修正清单见 §4.4。

---

## 1. 核心设计哲学解读（五支柱）

### 1.1 主动辨证内循环意识流（路线 B：思维空间）

`REFLEX_INNER_LOOP_v2.md §0` 明确区分两条路线：

| | 路线 A：预测纠正机 | 路线 B：思维空间（本项目） |
|--|--|--|
| 驱动力 | 最小化预测误差 | 最大化内部动力学相干性 |
| 终止条件 | MSE → 0 → 停止思考 | KL 自然振荡 → 永不停止 |
| 失败模式 | 真空中的完美预测器 | 混沌发散（受约束） |

**核心立场：思考不是预测错误的工具，思考是系统维持自身内部动力学丰富性的方式。预测是思考的副产物，不是目的。** KL 散度在此被重新解释为"惊讶/信息增益"（Schmidhuber 好奇心），而非"我猜错了多少"。

### 1.2 内外双循环 = 同一个自指循环的两个尺度

`MEMORY_SYSTEM_v4_DESIGN.md §七·八（内外循环一致性）` 明确：外循环（对话）与内循环（意识流）是**同一个"记忆-计算"自指循环在不同尺度上的运行**——共享状态与记忆，计算产物回流记忆，记忆又驱动下一步计算：

```
外部输入 → 外循环: forward(h_state + mem_kv) → 生成（带内循环"想法"）
        → 对话注入 h_t（L1）+ KV 入队（L4）+ 语义槽（L3）
        → 内循环: 读 h_t/记忆 → SelfModel 演化 → Hebbian → 新 h_t
        → 下次外循环: 用最新 h_state + 记忆生成 ← 状态流通
```

自指循环基础单元：**状态(h_t) + 记忆(mem_kv/语义槽) + 输入 → 计算 → 新状态 + 新记忆**——每一步计算改变状态，状态决定下一步计算。

### 1.3 不依赖外界输入的主动求证（sigma 触发器 + 涌现冷却）

- 模型困惑时主动提问，**不依赖外部输入**——sigma 是模型自身的内部状态信号（专家 `uncertainty_head` 的输出），不是外部规则；
- `InteractionManager` 的冷却不是定时器：**回答 → 学习 → sigma 降 → 停止提问**（涌现冷却）；新困惑 → sigma 升 → 再问；
- 疑问未解决（sigma 持续高）→ 可以追问——真正的求知循环。

### 1.4 混沌边缘的系统工程（单步收敛 / 多轮稳定 / 系统级涌现）

`TRAINING_QUALITY_AUDIT.md §0（设计哲学澄清）` 给出审计基准与关键推论：

```
单轮循环：应能收敛（数值有界、梯度不爆）
多轮循环：跨步梯度不累积爆炸（权重有界、EWC 保护）
系统整体：通过周期循环（噪声注入 + 辩证缓冲 + 回放巩固）
         涌现性地维持混沌边缘意识流状态
```

**关键推论**：单步稳定性约束（`loss_stability = mse(h_t, h_next)`）是**正确的**——混沌边缘不应由单步的不稳定产生，而应由系统级循环涌现。不塌陷靠：噪声 PI 控制器 + 辩证缓冲反题优先 + KL 驱动噪声；不发散靠：KL/std 有界化 + Hebbian 双 clip + EWC 惩罚有界化 + 梯度裁剪。

### 1.5 部署即学习（权重修改是特性，不是 bug）

`README.zh-CN.md` + `WIKI.md §8`：模型在部署过程中持续学习——每次对话都是训练步骤。**部署阶段训练权重会被修改**是设计意图，通道包括：

```
Hebbian 局部更新（每步，非全局 BP）→ 专家权重
在线训练 CE + sigma 校准 + alignment（每轮对话）→ 专家 + uncertainty_head
mini/major consolidation（每 50/500 步）→ stable 专家 + 全局参数（attention/router/lm_head/AttnRes）
记忆固化（salience 驱动）→ 语义槽蒸馏进 stable
架构自修改（每 200 步）→ 专家分裂/裁剪/加层
```

`WIKI.md §8` 哲学总结：**内部循环不"看数据"，它"想"**——SelfModel 预测自己的下一步状态，Hebbian 强化被路由的专家，用户输入只是状态空间的扰动。

---

## 2. 自指循环的完整运行模型（整合理解）

### 2.1 内循环微循环（`loop/internal_loop.py`，A-K 阶段）

```
Stage A: v_t（= h_t 或辩证缓冲最新）→ KL 驱动噪声 → SelfModel 想象(z_prior)
         → u_next = decode(z, h_next) → EMA 混合 → forward_internal → loss_int
Stage B:  逐层 autograd.grad → focal_update（Hebbian，gamma=σ/阈值×gate×focus_boost）
Stage B2: plastic 渐进衰减（×0.999999/步）
Stage C:  SelfModel 三目标（imagination + KL + stability）→ AdamW
Stage C2: Critic 延迟一步 TD(0)（伪奖励 = f(loss_int, sigma)）
Stage D:  replay（priority=novelty+|loss|）+ 辩证缓冲分类（thesis/antithesis/synthesis）
          + 语义槽写入 + salience 固化检查
Stage E/F: 每 50/500 步 mini_distill / major_sleep（蒸馏+EWC+全局回放+soft-reset）
Stage G/H: 每 100 步 FluidRoles；每 20 步主动求证检查（sigma 驱动，无外部输入）
Stage K:  每 200 步架构自修改
```

### 2.2 外循环宏循环（`interaction/pipeline.py`）

```
用户输入 → [AWAITING_ANSWER? 反馈路径 | 新查询路径]
  → 生成（携带 h_state + mem_kv，Router 状态门控 + 记忆注意力）
  → 在线训练 _train_on_full（CE 只监督回复段 + sigma 校准 + alignment）
  → 对话注入 h_t（L1）+ KV 入队（L4）+ 语义槽写入（L3）
  → 反馈路径：StructuredFeedback（关键词 reward + 语义对齐 → focal_boost）
    → 困惑专家定向强化（外循环）+ push_focal_boost（内循环 Stage B 消费）
```

### 2.3 四个耦合界面（双循环在此闭合）

| 界面 | 外循环 → 内循环 | 内循环 → 外循环 |
|------|----------------|----------------|
| 状态 | 对话注入 `model._h_state`（L1） | 生成时读 `_h_state`（Router 状态门控 + SelfModel 演化） |
| 记忆 | KV 入队 / 语义槽写入 / 对话 push endosphere | `mem_kv` 拼接 + AttnRes 语义槽 source |
| 权重 | 在线训练 / focal_boost 推送 | Hebbian / consolidation / 架构自修改 |
| 提问 | 用户回答 → 反馈 → 学习 → sigma↓ | sigma↑ → 主动提问（无外部输入触发） |

### 2.4 涌现冷却因果链（验证"主动求证"是否成立的总纲）

```
困惑（sigma 高）→ 主动提问 → 用户回答 → focal 强化 + 在线训练
→ 学习（权重/记忆变化）→ sigma 下降 → 冷却（不再问同一问题）
→ 新困惑（sigma 再升）→ 再问
```

这条链的**触发端**不依赖外部输入（内循环自查），**学习端**依赖外部反馈（这正是"部署即学习"与"主动求证"的合流点）。**链是否成立取决于 sigma 对学习的响应性**——这是后续评估的关键判据（§5）。

---

## 3. 实现忠实度矩阵

### A 类：高忠实（机制真实实现，与哲学一致）

| 机制 | 设计意图 | 实现证据 | 判定 |
|------|---------|---------|------|
| 内循环 A-K 阶段 | 意识流持续运行 | `internal_loop.py` 全部阶段就位；实跑 6 步无异常，loss≈1.01 | ✅ |
| SelfModel v2（GRU+Prior/Posterior+Decoder） | 有状态世界模型 | `core/self_model.py` 完整实现 MoE Prior/Posterior 矩匹配 | ✅ |
| KL 驱动噪声 | v2：惊讶调节探索 | `internal_loop.py:254-258` 用 `kl/β` 驱动 PI 噪声控制器 | ✅ |
| 辩证缓冲 | thesis/antithesis/synthesis 三分 | `dialectical_buffer.py` 分类 + 反题优先 + 余弦合成检测 | ✅ |
| 涌现冷却（无硬定时器） | 冷却由 sigma 涌现 | `manager.py` 无冷却计时器，`can_ask` 纯 sigma 门控 + 会话上限 | ✅ |
| Hebbian（SwiGLU 局部梯度） | 非全局 BP 的局部学习 | `hebbian_update.py` 数学正确（w_down_old 快照 + 动量 + 双 clip） | ✅ |
| 固化解耦 | plastic→consolidation→stable | `consolidation.py` 三级机制 + EWC（object identity 适配） | ✅ |
| 防发散工程 | 混沌不爆炸 | KL clamp / std clamp(max 3.0) / EWC clamp / Hebbian clip / 梯度裁剪（P0 系列已修复） | ✅ |
| L1 对话注入 | 对话进入意识流 | `pipeline._inject_dialog_memory`（归一化 + alpha 混合） | ✅ |
| L4 KV 内容记忆 | 能"复述原话" | `attention._last_kv` 缓存 + `memory_bank.add_round_kv` + mem_kv 拼接 | ✅ |
| salience 自发固化 | 记忆"成熟即固化"（行为驱动） | `memory_bank.accumulate_salience` + `internal_loop._check_memory_consolidation` | ✅ |
| focal_boost 跨线程闭环 | 辩证信号贯穿内外循环 | `model.push/pop_focal_boost`（锁保护的生产-消费） | ✅ |
| 求正对齐 + alignment loss | 用户语义直接校正困惑专家 | `feedback.py`（confused_text 对齐）+ `pipeline._compute_alignment_loss` | ✅ |
| sigma 校准（在线） | 让 sigma 学会反映 CE 不确定度 | `pipeline._train_on_full` calibration loss（权重 0.05） | ✅（弱，见 B 类） |

### B 类：部分忠实（机制存在，但存在张力或起点不足）

| 机制 | 张力/缺口 | 证据 |
|------|----------|------|
| sigma 信号信息量 | 校准权重 0.05、每轮一次、且 checkpoint 中 uncertainty_head 未经监督训练 → 实测 per-token sigma std=0.018（近乎常数） | 实测 + `train_reflex.py:318-322` 冻结列表 |
| KL 的双重身份 | v2 文档既称"KL=惊讶，追求信息增益"又将其作为 loss 项最小化；实现忠实执行了"loss 最小化 + KL 驱动噪声"两半，"最大化惊讶"无实现载体 | `internal_loop.py:777-793` vs `REFLEX_INNER_LOOP_v2.md §2.2` |
| Hebbian 更新量级 | 单步 ~1e-6（实测 6 步 6.08e-6），与"实时学习"期望之间存在"慢"与"持续积累"的张力；且无监督方向 | 实测 + `WIKI.md §4.13` |
| h_to_bias_weight（状态门控） | 训练期冻结+主路径不传状态 → checkpoint 全零（实测 absmax=0.0）；**但部署在线训练/内循环确有梯度路径，作者已在 v4 §七·八 记录"随对话使用逐步生效"**——属"设计中的渐进生效"，当前 checkpoint 尚未生效 | 实测 + `MEMORY_SYSTEM_v4 §七·八` |
| memory_bank（L2/L3） | 设计方向 A（随优化器可微演化）成立：AttnRes 记忆 source 对选中槽行可微，全局回放白名单含 memory_bank；**但 checkpoint 训练版本无 memory_bank（版本漂移）→ 加载后 4 个参数 missing、随机起点** | 实测 missing=4 + `core/attn_res.py:98-103` |
| 忙循环 | "持续思考"哲学 ✓，但实现无节流（无 sleep，`internal_steps_per_cycle` 未用），实测 2.17 steps/s 持续占用；且 `REFLEX_ARCHITECTURE.md` 原则 3 宣称"事件驱动替代忙等待" | 实测 + `internal_loop.py:221-235` |
| 架构自修改 | 哲学上"架构随学习演化"，但默认开启、无对照实验、replace 复制最常用专家+噪声会改写已训练权重 | `config: arch_self_mod_enabled=True` + `architecture.py:187-203` |
| AttnRes post_norm 部署覆写 | 部署脚本将 checkpoint 学到的 post_norm 尺度（实测 ±0.076）强制覆写为 0.1，与训练行为不一致（为"记忆微调前提"的工程权衡） | `run_mini.py:67-71` + `train_memory.py:79-81` |

### C 类：断裂/未闭环（哲学要求，但链路未闭合）

| 断裂 | 哲学影响 | 证据 |
|------|---------|------|
| 涌现冷却因果链不成立（当前状态） | sigma 无信息量 → "学习→sigma 下降→冷却"无法涌现；提问时机本质随机 | 实测 sigma std=0.018；`manager.can_ask` 纯 sigma 门控 |
| confusion_map 失效 | "概念级困惑追踪"是元认知雏形，但 hash key 用生成的 question 文本 → 同概念两次文本不同 → count 永不累积 | `WIKI.md §4.17` 已记录 + `confusion_map.py:128-137` |
| replay priority 被 KL 主导（P0-C） | 蒸馏偏向困惑状态 → stable 专家被"困惑"污染，违背"stable=长期记忆" | `internal_loop.py:369-376`（审计 v2 已指出，未修） |
| 内循环梯度与外部困惑不对齐（P1-4） | Hebbian 梯度源是 SelfModel 想象，非用户困惑 → "辩证纠正"未真正作用于困惑区域 | `TRAINING_QUALITY_AUDIT_v2.md §P1-4`（未修） |
| Stage E/G/H 死代码 | 文档宣称的消化队列/序列学习/自对弈不存在（`digestion.py`/`self_play.py`/`reward.py` 无调用方） | grep 实证 |
| "惊讶最大化"无载体 | 若哲学目标是好奇心最大化，KL 应进 Critic 伪奖励（intrinsic reward）而非仅作 loss 正则 | `internal_loop.py` 现有实现 |
| SelfModel/Critic 未预热即部署 | 意识流从随机初始化起步（当前 checkpoint phase=sft，warmup 未跑）→ 想象内容无语义 | 实测权重分布=默认初始化 + `WIKI.md §6`（作者已记录） |

---

## 4. 实证核查结果（2026-08 实跑）

### 4.1 checkpoint 解剖（`sft_kd_150k_final.pt`，8.53GB）

| 项目 | 实测值 | 解读 |
|------|--------|------|
| 阶段 | phase=sft, step=2999 | 对应 `run_pipeline.sh` 阶段 1（SFT 3000 步），KD/精修/warmup 未含 |
| `h_to_bias_weight` | 全零（24 层 absmax=0.0） | 训练期设计性冻结；部署期将逐步在线学习 |
| `verify_threshold` 参数 | 0.5（未学） | 死参数（manager 用 config 常量） |
| `self_model.*` / `critic.*` | 权重分布 = 默认初始化（±0.0395） | 未跑 warmup，随机起步 |
| `memory_bank.*` | 4 个参数 missing | 版本漂移，部署随机起点 |
| embedding | 范围 ±0.9，已充分训练 | 预训练真实执行过 |
| AttnRes post_norm | ±0.076 | 训练学到的尺度，部署时被强制覆写为 0.1 |

### 4.2 内循环实跑（小配置 4L/512d，CPU，6 步）

```
2.17 steps/s（忙循环，无节流）
loss_int ≈ 1.01 | sigma ≈ 0.456 | KL ≈ 0.472 | h_state norm ≈ 3.2
0 错误
plastic 专家权重最大变化 6.08e-6（≈1e-6/步）
Hebbian gate 实测 ≈ 1.0（激活门不是瓶颈，瓶颈是自指 loss 的梯度量级）
```

**解读**：内循环动力学"活着"——KL 有界且非零（惊讶信号存在）、h_t 持续演化、无崩溃。符合"混沌边缘不塌陷不发散"的短期观测。但"学到了什么"无法从这 6 步判断——需要长跑 + 行为仪表（§6.4）。

### 4.3 生成实测（SFT 2999 步 checkpoint）

| 问题 | 结果 |
|------|------|
| 中国的首都是 | 半对（提到"北京"），但循环论证、语言混乱 |
| 解释光合作用 | 跑题胡编（"紫外线照射"） |
| 1+1 等于几？ | 完全胡言（"是的，不是！2.0\*1=3.0\*\*5..."） |

**解读**：这是**外循环（SFT）基础能力不足**——3000 步 × 10 万条模板化教师数据，过拟合 + 多样性缺失；不是内循环哲学的问题。`WIKI.md §5` 实测同样确认"硬标签 SFT 真实 loss 1.81，能答北京（3/5），KD 反而退化"。

### 4.4 对上一轮体检报告的修正清单（诚实记录）

| # | 上轮判定 | 修正后判定 | 修正依据 |
|---|---------|-----------|---------|
| 1 | "h_to_bias_weight 永远零，状态门控从未生效" | 训练期零是**设计**（v4 §七·八 已记录）；部署在线训练/内循环确有梯度路径，会随使用逐步生效；当前 checkpoint 尚未生效 | `pipeline._train_on_full` 传 h_state；作者验证记录 |
| 2 | "memory_bank 无梯度路径" | 梯度路径存在（AttnRes 记忆 source 对选中槽行可微，方向 A 成立）；问题是 checkpoint 版本不含 memory_bank（版本漂移） | `attn_res.py:98-103` + `internal_loop.py` 全局白名单 |
| 3 | "自指闭环无外部锚点 = 缺陷" | 这是**路线 B 的设计意图**；`TRAINING_QUALITY_AUDIT.md §3` 已判定 imagination loss 为"合理的交替优化（EM-like co-adaptation）"；评估标准应是系统行为指标 | 审计文档 §3 修正表 |
| 4 | "KL 被最小化，与好奇心宣称相反" | 文档自身存在张力（KL 既是 loss 项又是惊讶信号）；实现忠实执行了 loss 正则 + KL 驱动噪声两半；"最大化惊讶"确无实现载体——这是**待决策的哲学点**，非单方面缺陷 | `REFLEX_INNER_LOOP_v2.md §2.2` |
| 5 | "Hebbian 在线学习可忽略，'部署即学习'无效" | 机制**真实运行**（权重确实被修改——这正是"部署阶段权重会被修改"的设计）；单步量级小是特性与张力；缺的是"学习有效性的可归因验证" | 实测 + WIKI §6 部署测试（500+ 步稳定） |
| 6 | "主动求证触发随机" | 触发机制**不依赖外部输入**（设计 ✓，`_try_generate_verification` 每 20 步自查 + seed 兜底）；问题是 sigma 在 checkpoint 中无信息量 → "困惑"语义未落地，这是涌现冷却前提缺失 | 实测 std=0.018 |

---

## 5. 哲学框架内的动力学评估：内循环会出现什么问题

### 5.1 防发散工程：完备（低风险）

KL 有界化（logvar clamp [-6,4] + prior_var 下界）、`sample_z` std clamp、EWC 惩罚 clamp、Hebbian `_MAX_UPDATE_SCALE/_MAX_DELTA_NORM`、SelfModel/Critic 梯度裁剪、plastic 渐进衰减——`TRAINING_QUALITY_AUDIT.md` P0 系列全部落地。**数值爆炸路径基本封死**（实跑 6 步亦无异常）。

### 5.2 防塌陷机制：存在，但缺仪表验证（中风险）

防塌陷依赖：噪声 PI 控制器（目标 σ=0.5 维持振荡）+ KL 驱动噪声 + 辩证缓冲反题优先 + sigmoid sigma 本身有界。理论闭环成立，但**"是否真的在混沌边缘"没有观测仪表**：
- KL 是否长期不归零？（实跑 0.47 ✓ 短期）
- h_t 变化率是否持续 > 0？（v2 §4 验证原则）
- 专家激活分布是否分化而非坍缩？
- 有/无内循环的对照实验（可归因贡献）？

**风险路径**：如果 sigma 长期无信息量（常数），噪声 PI 控制器的误差输入失真 → 噪声退化为固定值注入 → "混沌边缘"退化为"随机漫游"。这是最需要警惕的退化模式。

### 5.3 涌现冷却回路的断点（核心风险）

涌现冷却要求：学习 → sigma 降。但：
1. checkpoint 中 sigma 无信息量（校准不足）；
2. Hebbian 学习量级 ~1e-6/步，对 sigma（uncertainty_head 参数）无直接作用（sigma 只被 calibration loss 0.05 权重缓慢校正）；
3. → "回答 → 学习 → sigma 降 → 冷却"在**当前 checkpoint 状态**无法涌现。提问要么不触发（sigma<0.5），要么随机触发（噪声越过阈值）。

**这不是哲学问题，是实现的校准/预热缺口**——修复方向见 §6.2。

### 5.4 部署学习的风险（权重被修改的代价）

机制齐全（§1.5），但：
- **无监督方向**：Hebbian 梯度来自自指 loss，语义方向无保障；在线 CE 训练只监督回复段，尚可；
- **soft-reset 抵消**：每 500 步 plastic 保留 50%，与"持续积累"张力；
- **无验证门禁**：没有"修改后能力不退化"的检查；
- **架构自修改默认开**：replace/split 直接改写已训练权重，无对照实验。

---

## 6. 结论与行动清单（哲学框架内）

### 6.1 保留（哲学内核，不动）

1. 路线 B：内循环不预测世界，维持自持动力学——保留 SelfModel 结构；
2. 双循环自指：h_state/mem_kv/语义槽共享——结构与数据流已就绪；
3. sigma 驱动的主动求证（不依赖外部输入触发）——机制保留，补校准；
4. 混沌边缘系统工程——防发散完备，补防塌陷仪表；
5. 部署即学习——权重修改是特性，补有效性验证。

### 6.2 修复断裂（按优先级）

**P0（涌现冷却的前提）**
1. **sigma 校准强化**：把 calibration loss 权重从 0.05 提高（或单独阶段训练 uncertainty_head 对齐 CE）；在 `train_consciousness_warmup` 中加入 sigma 校准目标；部署初期用保守冷却（审计 v2 P1-5 方案）防随机提问。
2. **warmup 必跑**：SelfModel/Critic 随机起步违背"意识流有内容"——`run_pipeline.sh` 补 Step 6（`run_all.sh` 已有），且产出 `warmup_final.pt` 后 KD 从它继续（run_all 已如此设计）。
3. **h_to_bias_weight 生效验证**：部署在线训练已含梯度路径，加一条 stats 显示其最大变化量（`global_drift` 已有类似机制），确认"状态门控随使用逐步生效"可观测。

**P1（辩证闭环）**
4. 修 P0-C：replay priority 解耦（按审计 v2 方案）；
5. 修 P1-4：内循环梯度接地（用户输入 embedding 作为 grounded 项加入 loss_int）；
6. 修 confusion_map：hash key 改 embedding 相似度分组；
7. 忙循环限速：`internal_steps_per_cycle` 生效（每步 sleep 或按步数限流），`_step_count` 递增即节流点；
8. 架构自修改默认关闭（`--no-arch-self-mod`），对照实验证明收益后再开。

**P2（一致性）**
9. memory_bank 版本对齐：重训或迁移时确保 checkpoint 含 memory_bank 参数；
10. 删除部署时 AttnRes post_norm 强制覆写（或将其并入 warmup 训练目标，使训练-部署一致）；
11. 文档-代码对齐：删/标注死代码（digestion/self_play/reward、`generate_multi_turn`/`self_instruct_questions` 等），更新 WIKI 已知问题表（confusion_map 失效、矩匹配不完整等仍未修但表已清零）。

### 6.3 哲学决策点：KL 的身份

二选一并让实现一致：
- **选项 1（稳定正则）**：KL 继续作 loss 项（现状），把文档中的"最大化惊讶"表述修正为"维持惊讶的有界振荡"——噪声 PI 控制器负责探索；
- **选项 2（好奇心奖励）**：KL 进入 Critic 伪奖励（`compute_pseudo_reward` 加 KL 项），loss 只保留 imagination+stability——实现"惊讶驱动学习"的宣称。

建议选选项 1（0.77B 容量下更稳），但必须修正文档表述，消除"宣称最大化、实现最小化"的张力。

### 6.4 验证仪表体系（按 v2 §4 + 审计 §4，全部可落地）

| 仪表 | 测量 | 预期 |
|------|------|------|
| KL 不归零 | 千步级 KL 轨迹 | 有界非零振荡 |
| h_t 不固定 | ‖h_t - h_{t-1}‖ 变化率 | 持续 > 0 |
| 专家分化 | 路由激活分布熵 | 不均匀但不坍缩 |
| 内循环可归因 | 有/无内循环生成对照（`--no-internal-loop`） | 可归因差异 |
| sigma 响应学习 | 反馈后 sigma 是否下降（涌现冷却曲线） | 下降 |
| 部署学习有效 | 同任务前后回答质量/困惑度变化 | 不退化 |
| 记忆利用 | 轮 5 复述轮 1（train_memory 验证） | 能复述 |

### 6.5 训练/数据建议（服务哲学目标）

1. **外循环基础能力是内循环有意义的前提**：当前 SFT 3000 步 × 10 万条模板数据不足——用 150 万条真实数据（Firefly/BELLE/ShareGPT/COIG）+ 教师生成数据配比重训，SFT 2~4 万步（上一轮报告 §5.2 的配比方案仍然有效）；
2. **数据中注入"澄清式对话"**（模糊指令→反问→澄清→回答）：这是把"主动求证"从 sigma 机制迁移为**可学习行为**的最短路径，与 sigma 机制互补；
3. **记忆微调（`train_memory.py`）**：长程引用多轮数据（6-8 轮 + 跨轮引用）补齐 L4 利用，这是 v4 §七·五 已设计的轻量步骤；
4. **warmup 纳入一键流水线**（run_pipeline.sh 补 Step 6）。

---

## 7. 对核心设计主张的逐条确认

| 主张 | 确认结论 |
|------|---------|
| 主动辨证内循环意识流是核心 | ✅ 确认。实现忠实（A-K 阶段齐全，路线 B 数学自洽）；"辨证"闭环有断裂（P0-C/P1-4/confusion_map），可修 |
| 内外双循环共用一套自指循环 | ✅ 确认。v4 §七·八 的结构与数据流全部就绪（h_state/mem_kv/语义槽/对话注入/在线训练带状态）；状态门控随部署使用渐进生效（作者已记录） |
| 不依赖外界输入的主动求证输出 | ✅ 确认机制真实：内循环每 20 步自查（无外部输入），seed 兜底保证无对话也能生成问题，经管道以 `[QUESTION]` 输出；缺口在 sigma 信息量（校准/预热），修复路径明确 |
| 混沌边缘：不塌陷也不发散 | ✅ 防发散工程完备（有界化全落地）；防塌陷机制存在但缺长跑仪表——建议按 §6.4 建立观测后给出实证结论 |
| 部署阶段实时学习（权重被修改是特性） | ✅ 确认：Hebbian/consolidation/全局回放/在线训练/记忆固化/架构自修改全部真实修改权重（实测 Hebbian 每步 ~1e-6 累积 + 每 50/500 步巩固批次修改）；需要补的是"有效性可归因验证"，而非关闭学习 |

---

## 8. 一句话总结

**哲学内核是自洽的，且比上一轮报告所认知的更完整地落地了**：五支柱（意识流/双循环/sigma 求证/混沌边缘/部署即学习）在代码中都有真实机制，防发散工程完备，自指设计是路线 B 的刻意选择而非缺陷。当前真正的问题集中在三处**可修复的断裂**：① sigma 校准/预热不足导致"涌现冷却"因果链在现有 checkpoint 上不成立；② 辩证闭环两处未修（replay 偏置、梯度不接地）与 confusion_map 失效；③ 部署学习缺验证仪表与对照实验。修复路径都已给出，且全部在哲学框架内——不改变任何设计主张，只让实现重新忠实于哲学。

---

*本文档是对 `TRAINING_QUALITY_AUDIT.md` / `TRAINING_QUALITY_AUDIT_v2.md` / `WIKI.md` 已知问题表的整合与补充，可作为下一轮修复（P0-C、P1-4、sigma 校准、warmup 入流水线、验证仪表）的依据。*
