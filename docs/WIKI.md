# NeuroStream-Reflex V2 Mini — 项目 WIKI

> 综合文档：项目概览 + 完整代码解析 + 训练流水线 + 部署 + 已知问题。
> 代码解析基于全部 28 个 Python 文件逐行分析（行号以当前文件为准）。最后更新：2026-08-07

---

## 1. 项目概览

0.77B 参数中文神经形态语言模型，核心设计是"外部训练 + 内部意识流"双循环：

- **外循环（训练）**：标准三阶段——预训练 → SFT → 知识蒸馏，产出能对话/回答的 base 能力
- **内循环（部署）**：SelfModel 状态演化 + Hebbian 在线学习 + Critic 价值 + 辩证缓冲，让模型在交互中持续"思考"

```
用户输入 → [Pipeline] → 文本/状态 → [InternalLoop 后台线程]
                                  ├─ SelfModel: h_t/z_t 演化（意识流）
                                  ├─ Hebbian: 专家权重在线强化
                                  ├─ Critic: 价值估计（惊讶度调制）
                                  ├─ 辩证缓冲: thesis/antithesis/synthesis
                                  └─ Consolidation: 每 50/500 步固化
        ← 回答/验证提问 ←
```

**内部循环不处理文本/代码数据**——它处理抽象状态向量（640 维 embedding），通过 SelfModel 预测自己的下一步状态（imagination）、KL 探测惊讶（curiosity）、Hebbian 强化被路由的专家。用户提问只是给状态空间一个初始扰动。

---

## 2. 模型配置（ReflexMiniConfig，~0.77B）

| 参数 | 值 | 说明 |
|------|-----|------|
| d_model | 640 | 隐藏维度（5×128，GPU 对齐） |
| d_ff | 2048 | FFN 中间维度 |
| n_layers | 24 | 层数 |
| n_heads / n_kv_heads | 10 / 2 | GQA 5:1 |
| vocab_size | 151936 | Qwen2.5 词表 |
| max_seq_len | 2048 | 训练 1024 / 推理 2048 |
| 专家 | 4 stable + 2 plastic | MoE，top-2 路由 |
| 专家 LR 谱 | stable 1e-7 ×4, plastic (1e-5,1e-4) | baseline_lr 决定塑性 |
| AttnRes | block_size=6, rank=160 | 3 个块边界 |
| verify_threshold | 0.5 | 主动求证 sigma 阈值 |
| plastic_reg_strength | 1e-6 | 塑性权重渐进衰减 |
| self_model_z_dim | 128 | 潜变量维度 |
| critic_hidden_dim | 320 | |
| 参数量 | 769M | |

---

## 3. 代码地图

```
config/model_config.py    三档配置（ReflexConfig/Medium/Mini）
core/
  model.py                ReflexModel（24 层 MoE + AttnRes + SelfModel + Critic + generate）
  expert.py               MoE 专家（SwiGLU + 可学习 sigma + Hebbian 缓冲）
  router.py               top-k 路由 + 负载均衡 + sigma 聚合 + 状态门控
  attention.py            GQA + RoPE + QK-Norm + Flash
  attn_res.py             块间 Delta 注意力残差（AttnResStack）
  self_model.py           世界模型（GRU + MoE Prior/Posterior + Decoder）
  rope.py / rmsnorm.py    旋转位置编码 / 归一化
loop/
  internal_loop.py        内部循环（Stage A-K 主逻辑，911 行）
  gradient_manager.py     逐层 autograd.grad 管理
  dialectical_buffer.py   辩证缓冲（thesis/antithesis/synthesis）
  endosphere.py           扁平状态缓冲（被辩证缓冲替代）
learn/
  hebbian_update.py       focal_update（Hebbian 权重更新 + 动量 + 双 clip）
  consolidation.py        mini_distill / major_sleep（每 50/500 步固化）
  critic.py               ReflexCritic + pseudo-reward
  fisher.py               EWC 弹性权重固化
  confusion_map.py        困惑追踪（问题文本做 key，已知失效）
  critical_noise.py       临界噪声调度
  fluid_roles.py          专家角色评估（baseline_lr 调整）
  replay.py / reward.py  回放 / 奖励
interaction/
  pipeline.py             交互管线（提问处理 + 在线训练 + 专家 Hebbian）
  manager.py              InteractionManager（验证提问状态机）
  feedback.py             结构化反馈（关键词 + 对齐 → focal boost）
  digestion.py            DigestionQueue（无调用方）
improve/
  architecture.py         架构自修改（prune/replace/split/add_layer）
  self_play.py            自对弈
train/
  train_reflex.py         训练主脚本（pretrain/SFT/warmup/distill + LR 调度）
  data_pipeline.py        数据管线（流式 + chat template + 稀疏 KD + teacher 生成）
  monitor.py              梯度/专家/sigma/权重监控与自愈
scripts/
  generate_qa.py          教师自问自答生成（qa/multi-turn/self-instruct）
  filter_qa.py            数据清洗（重复/噪声/去重）
  dedup_sft.py            SFT 数据去重
  prepare_data.py         云端数据组装（四源 pretrain + 四源 SFT）
  init_from_qwen.py       Qwen embedding 初始化
  run_pipeline.sh         一键训练流水线（幂等）
run_mini.py               部署入口（内循环全开）
chat_sft.py               checkpoint 快速测试
```

---

## 4. 完整代码解析

### 4.1 core/rmsnorm.py — RMSNorm

- `RMSNorm(d_model, eps=1e-6)`：weight 全 1 初始化，`x/√(mean(x²)+eps)*w`，无 bias
- 形状 `[..., d]` 不变；PostDA-Norm 靠调用方 `weight.fill_(1e-3)` 注入缩放语义

### 4.2 core/rope.py — RoPE

- **rotate-half** 风格（LLaMA 式前后两半互旋，非 GPT-NeoX 交错）
- `inv_freq = 1/θ^(arange(0,hd,2)/hd)`；缓存 `_cached_cos/sin` **只增不减**
- 位置恒从 0 开始（**无 KV-cache 增量生成**，长序列全量重算）
- `apply_rope` 死代码（attention 直接调 self.rope）
- **异常**：docstring 声称 "NTK-aware extrapolation" 但无 NTK 逻辑；head_dim 奇数会切片错位（当前偶数）

### 4.3 core/attention.py — MultiHeadAttention（GQA+RoPE+QK-Norm）

- `head_dim=d/n_heads`；`kv_proj` 输出 2·n_kv·hd；`n_rep = n_heads//n_kv`
- 前向：q/kv 投影 → QK-Norm（每头 RMSNorm）→ RoPE → `repeat_interleave` 复制 KV → SDPA → o_proj
- **Mask 组合**（L109-131）：padding mask `[B,1,1,T]` expand → -inf，再叠加 causal triu；显式 mask 时 `is_causal=False`（禁用 Flash）
- 无 mask 时 `is_causal=True` 走 Flash；eval 时 dropout=0 保 Flash
- **异常**：n_rep 整除无校验；padding mask 全量 [B,H,T,T] 内存放大；`self.scale` 死代码

### 4.4 core/router.py — Router

- `gate_weight[d,n]`（std 0.02）、`gate_bias`、`h_to_bias_weight[d,n]`（**全零初始化**，状态路由逐步学出）
- `verify_threshold` 可学习 Parameter；`expert_util_ema=ones/n`
- forward：logits → **`is_internal` 时 `logits += 2.0`（魔法常数，无注释）** → `+h@h_to_bias_weight` → topk → 仅 top-k 内 softmax
- training 统计：gating 熵、util（scatter_add 计数）、`_last_aux_loss`（Switch 式 n·Σ(P·f)）
- `aggregate_sigma`：加权选中专家 sigma → `.mean().item()`（**Python float，断图**）
- `add_column/remove_column`：**重建 Parameter 对象**（优化器需重建）
- **异常**：`activation_running_mean/var` 死 buffer（从不更新）；aux loss 的 util 未 detach

### 4.5 core/expert.py — Expert（SwiGLU + 不确定性 + Hebbian 缓冲）

- 结构：`w_gate/w_up: d→d_ff`，`w_down: d_ff→d`，SiLU 激活
- `uncertainty_head`：RMSNorm(d_ff)→Linear→GELU→...→Linear(8,1)，sigma=sigmoid
- buffers：`lr_bias`（±2 clamp → `effective_lr = baseline·exp(lr_bias)`）、`uncertainty_ema=0.5`、`activation_ema`
- 激活缓冲（普通属性）：`_input/_gate_pre/_gate/_up/_hidden/_output`（**保图**供 autograd.grad）
- momentum 缓冲：`_mom_w_down/_mom_w_gate/_mom_w_up`（读写全在 hebbian_update.py）
- `id = uuid4().hex[:12]`
- forward：`g=SiLU(w_gate(x)), u=w_up(x), h=g·u`（训练+save 时 detach 缓冲）→ dropout → w_down → sigma
- **异常**：EMA 无条件更新（eval 也污染 `avg_uncertainty`）；`update_local` legacy 死代码（用更新后 w_down，数学不精确）

### 4.6 core/attn_res.py — BlockDeltaAttnRes / AttnResStack

- `q_proj: d→rank, k_proj: d→rank, v_proj: d→d, out_proj: d→d, post_norm`（**weight=1e-3 初始化**，整体近零输出 → 模型初始≈标准 Transformer）
- forward：`deltas=[h0, h1-h0, ...]` → Q=RMSNorm(h_cur) → **K=L2 归一化 delta**（防 sink）→ V=原始 delta（保留幅度）→ einsum 打分 → softmax → 聚合 → 残差
- `get_routing_stats`：max_weight/entropy/n_sources（部署日志 ar0_max/ar0_ent 来源）
- **异常**：model.py 每块后 `block_outputs=[block_outputs[0], x]` → **实际永远只有 2 个 source**（docstring 声称所有块）；delta 链混入 AttnRes 自身输出

### 4.7 core/self_model.py — SelfModel（状态化世界模型）

- `PriorExpert/PosteriorExpert`：RMSNorm→Linear→GELU→mean/logvar 双头（logvar 零初始化=单位方差）
- `MoEPrior/MoEPosterior`：router softmax 选专家 → **矩匹配**：`mean=Σw·μ`，`var=Σw·exp(lv)`（**缺跨组件方差项 Σw(μ-μ̄)²** → KL 被低估）
- SelfModel：`GRUCell(2d+z, d)`（输入 cat[h,z,action]）；decoder 末层 std=0.02（**非零**，注释：零初始化会杀梯度）
- `sample_z`：logvar clamp [-6,4]，`std=exp(0.5·lv)·temp` **clamp(max=3.0)**
- `kl_divergence`：prior_var clamp(min=1e-4)，`kl=0.5·(Δlv+(pv+(Δμ)²)/prior_var-1)`，sum(-1).mean()
- **异常**：矩匹配不完整；PRIOR_VAR_MIN 与 LOGVAR_MIN 冗余矛盾

### 4.8 core/model.py — ReflexMoELayer + ReflexModel

**ReflexMoELayer**
- LR 谱扩展（不足重复末位/超出截断）；`stable/plastic_experts` 是 **all_experts 的视图切片**（共享参数）
- forward：attn 残差 → ln2 → 展平 → router → 遍历专家（未选中 clear_buffers）→ **`output=zeros_like` + `index_add_`（in-place）** → 残差
- sigma 汇总：`expert_sigmas[i]=σ.mean().detach()`；`learnable_sigmas[i]=σ.mean()`（注释称可微但 in-place 赋值实际断图）
- `add_expert/remove_expert_by_idx`：与 router 列严格对齐

**ReflexModel**
- embedding（std 0.02）+ 24 层 + AttnResStack + ln_f + **lm_head=embedding（weight tying）**
- SelfModel + Critic + endosphere + 线程安全（RLock + focal boost 生产消费队列）
- `_apply_layers_with_attnres`：checkpoint 条件 = `training & grad_enabled & not is_internal`（内部循环需要完整图供 Hebbian）
- `forward`：embedding → 层 → ln_f → lm_head；`return_hidden` 提前返回省 15GB（vocab 151936）
- `forward_internal`：1 token 前向，is_internal=True（跳过 checkpoint + h_state 路由偏置）
- `generate`：eval 下 no_grad；repetition penalty（**惩罚全部历史**）→ top-k → top-p → multinomial；三连 token 重复 ≥4 停；stop_ids 单 token 校验
- **异常**：MoE 输出对专家/路由**不可微**（index_add_ 到零张量）——专家只靠 Hebbian、路由只靠 aux+h_to_bias；`get_aux_loss` 尾部死 return；`probs.sum()==0` 仅 B=1 合法

### 4.9 loop/internal_loop.py（911 行）— 核心学习线程

**__init__（L42）**
- 事件驱动线程：`_event`（阻塞/唤醒）、`_step_done`（单步完成）、`_step_count`（触发周期任务）/`_total_steps`（统计）
- GradientManager、PriorityReplayBuffer、CriticalNoiseScheduler
- **endosphere 迁移**：EndoSphereBuffer → DialecticalBuffer（L71-80）
- FluidExpertRoles、ConfusionMap、ArchitectureModifier
- SelfModel 优化器 AdamW(1e-4)；**全局优化器**（L121-134）白名单过滤（attention/router/lm_head/ln/attn_res），lr=1e-6（"Expert 由 Hebbian 更新，Embedding 冻结"）

**start/pause/resume/stop**：pause 等整个 A-K 步完成（非仅锁内部分）→ 取 model._lock

**`_execute_step_inner`（L233）主循环各 Stage：**

| Stage | 行号 | 执行内容 |
|-------|------|---------|
| A | 234-296 | 取状态（_h_state→endosphere→随机初始化）→ 噪声（KL 驱动，一步延迟）→ SelfModel 想象 → EMA 混合（α=0.8）→ forward_internal → loss_int → sigma/gamma |
| B | 298-307 | 倒序逐层 autograd.grad（retain=idx>0）→ focal_update（effective_lr, gamma, focal_boost 一次消费） |
| B2 | 309-317 | plastic 参数 ×(1-1e-6)（渐进衰减替代 soft_reset） |
| C | 319-320 | `_update_self_model`：三目标（想象+好奇+稳定）全图 backward → 只更新 SelfModel |
| C2 | 322-323 | `_train_critic`：伪奖励 + 延迟 V bootstrap |
| E/F | 325-331 | 每 50 步 mini_distill（replay≥4）；每 500 步 major_sleep（replay≥128） |
| K | 333-348 | 每 200 步 + arch_self_mod → 架构自修改 → 重建优化器/梯度管理器 |
| D | 352-364 | replay.add（priority=novelty+\|loss\|）+ endosphere.push（辩证法分类）；_step_count+1 |
| G | 388-397 | 每 100 步 FluidExpertRoles.evaluate |
| H | 399-403 | 每 20 步 can_ask → `_try_generate_verification`（emergent 问题生成） |

**`_compute_loss`（L645）**：`z_post=observe_and_correct(h_t, pred_emb)`；`o_pred=decode(z_t,h_t).detach()`（固定想象目标）；`loss=MSE(pred,o_pred)+0.1·KL(post‖prior)`；`_kl_value` 供下轮噪声

**`_update_self_model`（L704）**：重前向 → `pred=forward_internal(u_decoded, h_state=h_next)`（**transformer 参与本图**）→ 三目标 → `loss.backward()`（L755 全图 backward——C1 已修：输入 detach + backward 前 zero_grad）→ clip(sm)→step

**`_sigma_gamma`（L759）**：`base=clamp(σ/threshold, 0.5, 2.0)`；`td_mod=min(|V(s)|/2, 1.0)`（**用 |V| 非 TD-error**）→ γ 上限 3

**`_train_critic`（L785）**：`pseudo_r=compute_pseudo_reward(loss_int, sigma)`；**TD bootstrap（L818-823）**：`v_next=_prev_v_pred`（M5 已修：延迟一步 TD(0)，用下一步 V 作 bootstrap）→ `target=r+γ·V(s_{t-1})`；`_prev_v_pred=v_pred.detach().item()`

**`_try_generate_verification`（L407）**：sigma≤0.5 return → 最困惑 expert（末层 argmax）→ emergent 问题生成（temp=0.7+0.4·σ，rep=1.3+gap·0.5）→ confusion_map.record → submit_question → `[MODEL ASKS]` 打印

**已知问题**：`_imagine`/`_get_target`/`compute_focal_gradient` 死代码；Stage C 梯度泄漏（C1）；Critic TD 反向（M5）；噪声一步延迟；`_step_count` 只计成功步

### 4.10 loop/gradient_manager.py

- `iterate_layers`：末层→首层倒序，retain=layer_idx>0（第 0 层释放全图）
- `compute_expert_gradients`：**先捕获 (expert, _output) 快照**（防并发前向覆盖）→ `autograd.grad(loss, outputs, retain_graph, allow_unused)`
- `_grads_cache` 死属性；`loss` 参数未用

### 4.11 loop/dialectical_buffer.py

- 三池：`σ>threshold×1.05` → antithesis；`σ<threshold×0.45` → 查 resolved（余弦≥0.7 命中 → synthesis + 移除）否则 thesis；中间 → thesis
- `get_latest` 优先级 antithesis→synthesis→thesis（"系统越想自己困惑的事"）
- **异常**：类 docstring（1.5×/0.5×）与实现（1.05×/0.45×）矛盾；`d_model` 未用；`sample_batch` 无调用点

### 4.12 loop/endosphere.py（legacy）

- 扁平 FIFO deque + seq_buffer；`push` 忽略 sigma 参数；InternalLoop 构造时被 DialecticalBuffer 替换

### 4.13 learn/hebbian_update.py — focal_update（核心 Hebbian）

- 常量：`_MAX_UPDATE_SCALE=1e-2`、`_MAX_DELTA_NORM=1.0`、`_MOMENTUM=0.9`
- `_apply_momentum`：梯度 EMA（0.9·mom+0.1·delta）
- `focal_update(grad_y, expert, lr, base_gamma, focus_boost)`：
  1. 缓冲守卫（grad_y/激活缓冲 None 即 return）
  2. **激活门**：`gate = 1/(1+exp(5·(norm_h-3)))`（norm_h>3 压死更新，防病态激活）
  3. `γ = base_gamma·gate·focus_boost`；`scale = min(lr·γ, 1e-2)`
  4. **先存 w_down_old** → `Δw_down = grad_yᵀh` → isfinite → 动量 → 范数 clip 1.0 → 原地减
  5. `grad_h = grad_y @ w_down_old`（**必须用更新前**）→ 链式 grad_g/grad_u → SiLU' = sig+x·sig·(1-sig)
  6. w_gate/w_up 同流程
- **异常**：bias 更新无动量/无 clip（仅 isfinite 保护）

### 4.14 learn/consolidation.py — mini_distill / major_sleep

- `_safe_sgd_step`：manual SGD，isfinite 守卫 + 单参数 grad clip 1.0
- `mini_distill`（每 50 步）：teacher=detach 快照 → 仅 stable 可训练 → MSE 蒸馏 → `_safe_sgd_step(stable, sleep_lr)` → global replay（+0.01·aux）
- `major_sleep`（每 500 步）：+ estimate_fisher + ewc_penalty（λ=40）→ distill+penalty backward → safe_sgd → global replay → `_soft_reset_plastic(keep=0.5)` + clear_momentum + 清缓冲
- **异常**：backward 前无 model.zero_grad（与 C1 stale 梯度叠加）；fisher 末次采样梯度残留

### 4.15 learn/critic.py

- `ReflexCritic(d, hidden)`：RMSNorm→Linear→GELU→Linear(1)
- `get_normalized_v`：**副作用**——内部做 moving-mean/std 归一化
- `compute_pseudo_reward(loss_int, sigma, entropy=None, lyap=None)`：`w_consistency=0.4` **未被使用**

### 4.16 learn/fisher.py

- `estimate_fisher`：每采样迭代开头 zero_grad，NLL backward；**末次采样梯度残留**
- `ewc_penalty`：`λ·Σf·Δ²` 后 **`.clamp(max=100.0)`（硬截断，大漂移时梯度归零）**

### 4.17 learn/critical_noise.py / confusion_map.py / replay.py / fluid_roles.py

- `CriticalNoiseScheduler`：sigma 高 → 噪声大；anti-windup 积分项（条件符号存疑）
- `confusion_map.record`：**用模型生成的 emergent question 文本做 hash key**——同概念两次生成文本不同 → count 永不累积（追踪失效）
- `PriorityReplayBuffer`：`random.choices` 有放回采样（batch 可重复）
- `FluidExpertRoles.evaluate`：只调 baseline_lr，**不迁移 stable/plastic 列表**（角色迁移不生效）

### 4.18 interaction/pipeline.py — ReflexPipeline

- `process_text`：AWAITING_ANSWER → `_process_feedback`；否则 `_process_new_query`
- **PATH A `_process_feedback`**：retrieve_answer → StructuredFeedback.process → generate → 拼接 `full_ids=[问题,eos,回答,回复,eos]` → `_train_on_full(feedback_ctx)` → finalize_feedback
- **PATH B `_process_new_query`**：generate → `_train_on_full(None)`；QUESTION_PENDING 时追加 `[QUESTION] {confused_text}` → notify_displayed
- **`_train_on_full`（在线训练，L186-242）**：
  - 超长取尾部截断（H5 修复）
  - **（I2 已修：reply_start mask，只监督回复段）**，clamp 10.0
  - **sigma 校准**：`target=tanh(ce)` → MSE(learnable.mean(), target)·0.05（让 uncertainty_head 学"高 CE→高 sigma"）
  - alignment loss（feedback 时 0.2 加权）；异常静默
  - `_update_experts(loss)`
- **`_update_experts`（L273-367）**：逐层 autograd.grad → focal_update；目标专家 focus_boost；**未路由专家用 hidden_states 代理梯度更新 w_down**（×0.01，isfinite 防御）
- **异常**：docstring 声称 chat template 但 `_build_chat_input` 裸 encode

### 4.19 interaction/manager.py — InteractionManager（问答协议状态机）

- 状态：IDLE / QUESTION_PENDING / AWAITING_ANSWER / PROCESSING_ANSWER
- **`can_ask` 三条件**：`state==IDLE` 且 `questions<5` 且 `current_sigma > verify_threshold(0.5)`
- 涌现冷却哲学：回答→学习→sigma 降→不再问；新困惑→sigma 升→再问
- 生命周期：submit_question → notify_displayed → retrieve_answer → finalize_feedback（回 IDLE，无冷却计时器）

### 4.20 interaction/feedback.py — StructuredFeedback

- `process`：`reward=_keyword_feedback()`；对齐参照用 `confused_text`（非 query_ids）；alignment<0.3 放弃
- focal_boost 公式（S=0.3）：`reward<0 → 1+S·excess·plasticity·alignment`（强化）；`reward>0 → 1-0.5·S·...`（弱化）；clamp [0.1,5.0]
- `_keyword_feedback`：**正向词表先匹配** → （**M2 已修：负向词先匹配 + 长短语 + 否定前后缀正则**）
- `feedback_alignment_weight` 配置 deprecated

### 4.21 improve/architecture.py

- 阈值：entropy_high=1.5, entropy_low=0.3, util_low=0.02, util_prune=0.005, util_high=0.4
- `step`：冷却 200/500 → 熵 streak → 三级决策（prune→replace→split）→ add_layer（连续 20 次低熵）
- `_replace_expert`：复制最优专家权重+0.02 噪声；`_split_expert`：add_expert(lr×2)+复制
- **异常**：**split/prune 只改单层专家数 → model.py:314 torch.stack 崩溃（C2，部署默认关）**；`_prev_avg_entropy` 只写不读

### 4.22 train/train_reflex.py（1203 行）

**cosine_schedule（L128-141）**：三段式 LR——`warmup 线性升 → plateau 峰值保持 30% → cosine 衰减到 min_factor=0.1`（防短训练全程爬坡 + 末端空转）

**chunked_loss（L144-191）**：`hidden.detach().requires_grad_()` → 每 256 token 块 lm_head → CE sum → `/count/n_chunks/accum` → backward → del → 最后 `hidden.backward(hd.grad)` 打通主干

**chunked_distill_loss（L194-278）**：稀疏 CE 数学——`p_k=exp(t_logp_k)`, `q_k=gather(q_full)`, `p_rest=1-Σp_k`, `q_rest=1-Σq_k`；`CE_sparse=-Σp_k·log q_k - p_rest·log q_rest`（≡KL，teacher 熵常数）；`loss=ce_weight·ce + kd_weight·kld·T²`（T² 量纲校正）
- **异常**：teacher 存 T=1 logp，student 用 T=2（M6 已修：teacher 按温度缩放存储，文件 v3）

**ReflexTrainer**：
- `__init__`：冻结列表（self_model/critic/query_proj/uncertainty_head/lr_bias/verify_threshold/h_to_bias_weight...）；monitor 配 sigma_collapse_patience=999999（禁用）
- `save_checkpoint`：**fp16 权重存储 + 原子写（.tmp→os.replace）+ 先 prune 再写**
- `_prune_checkpoints`：**按步数数值排序**保留最新 2 个 + 清孤儿 .tmp
- `_restore_optimizer_state(phase)`：fresh_optimizer → 跳过；跨阶段 → 跳过；同阶段 → 恢复
- `_build_optimizer`：**warmup 自适应 min(warmup, total//8)**；LambdaLR+cosine(plateau=0.3)

**train_pretrain（L496）**：循环 `unscale→monitor→clip→step→zero→scheduler`（update 块内）；log(50)/save(2000) 在 update 块外（2000%8==0 与 accum-1 互斥）

**train_consciousness_warmup（L748）**：全冻结→只解冻 self_model+critic；SM AdamW(1e-4)/Critic AdamW(1e-3)；每步 emb 均值→v_t→sm 前向→forward_internal（no_grad）→三目标（**Critic 纯 MC 回归，M10 已修**）→ 状态推进；结尾解冻全部

**train_distill（L900）**：teacher logits v2（.jsonl.gz）；`.partial` 原子生成；reservoir max_samples；每步 chunked_distill_loss（T/ce/kd 可 CLI 覆盖）

### 4.23 train/data_pipeline.py（557 行）

- `JsonlReader`：逐行 loads（坏行跳过）；有界 shuffle_buffer 洗牌弹一个；.gz 自动解压
- `WudaoReader`：按字符切块 `chunk_size×3`（~3 字符/中文 token）
- `_pretrain_iter`：CLM 移位，<32 token 丢弃
- `_sft_iter`（L146）：轮次提取 → 有 chat_template 走 `_sft_iter_template`
- `_sft_iter_template`（L202）：**渐进构造**——逐轮 apply_chat_template，记录 assistant 内容区间（start=旧串长, end=新串长），labels 只监督 assistant 区间
- `_distill_iter`：4 张量 + 截断三件套（labels[-1]=-100）
- `__iter__`：random.choices 按比例选源 → 有界缓冲 shuffle
- `collate_fn`：batch 内 pad（_PAD_ID/-100）；teacher 字段 K 自适应
- `generate_teacher_logits`：TOP_K=64；reservoir 采样（均匀覆盖全语料）；chat_template 输入（v2）；logp round 4 位；gzip 输出

### 4.24 train/monitor.py（375 行）

- GradientMonitor：norm 阈值 10/1e-6（loss 参数未用）
- ExpertActivityMonitor：`_output is not None` 判激活 → 立即置 None
- SigmaMonitor：var<0.01 连续 patience 次（训练中 999999 禁用）
- ReflexMonitor.step → actions：grad_explosion（LR×0.5，**只降不升无下界**）/reset_expert/sigma_collapse（注噪）/runaway

### 4.25 scripts/generate_qa.py（429 行）

- `filter_answer`：长度[5,500]；重复字符/短词；纯符号；非文字<75%；复述问题；拒绝式开头
- 40 模板 × ~150 实体（build_questions 全组合）；15 条追问池
- `batch_generate`：**left-padding** 批量生成（decoder-only 必须）
- 三模式：qa（模板问答）/ self-instruct（教师自产问题）/ multi-turn（每轮批量推进 3-4 轮）
- **异常**：`generate_multi_turn`/`self_instruct_questions` 死代码

### 4.26 scripts/filter_qa.py / dedup_sft.py / prepare_data.py / init_from_qwen.py

- `filter_qa`：+ 整句重复检测（分句 Counter≥2）；md5 去重；统计分类
- `dedup_sft`：流式 md5(instruction|input|output) 去重
- `prepare_data`：Pretrain 四源（wiki/skypile/cwt/fineweb）；SFT 四源（firefly/belle/sharegpt/coig）→ `{"instruction","input","output"}`
  - **异常**：belle 按行解析 .json（若为数组几乎全丢）
- `init_from_qwen`：Qwen2.5-0.5B embedding 前 512 维 + 后 128 维 N(0,0.02)

---

## 5. 训练流水线

### 数据流

```
原始语料（wudao/skypile/fineweb…）→ prepare_data → pretrain_all.jsonl
教师生成: generate_qa.py（模板 QA / self-instruct / multi-turn）→ QA 数据
filter_qa.py 清洗（重复/噪声/去重）→ sft_kd_clean.jsonl
generate_teacher_logits（离线，top-64 稀疏）→ _teacher_logits_v2.jsonl.gz（KD 用）
```

### 阶段命令

| 阶段 | 命令 | 说明 |
|------|------|------|
| 一键全流程 | `bash scripts/run_pipeline.sh` | 数据生成→清洗→SFT→KD→精修（幂等可断点续跑） |
| Pretrain | `--mode pretrain` | 15B tokens 中文语料，lr 3e-4 |
| SFT | `--mode sft` | 教师生成 QA/多轮硬标签，lr 2e-5 |
| KD | `--mode distill --distill-temperature 4.0` | 软标签联合（CE+KD），lr 1e-5 |
| 精修 | `--distill-temperature 1.0 --fresh-optimizer` | 低 T 精确对齐，lr 5e-6 |

### LR 调度（三段式）

```
warmup(≤1/8 总步) → plateau(峰值保持 30%) → cosine 衰减到 10% 峰值
三阶段阶梯: SFT 2e-5 → KD 1e-5 → 精修 5e-6
```

### 训练结论（实测）

- 硬标签 SFT（教师 QA 数据）：真实 loss 1.81，**能答"北京"**（3/5 测试通过 + 泛化倾向）
- 软标签 KD（T=4）+ 低 T 精修：**质量反而退化**（拒绝话术被学走 + 英文噪声）——0.77B 容量下 SFT 硬监督优于 KD 软蒸馏

---

## 6. 部署

```bash
python run_mini.py --checkpoint <ckpt.pt> --device cuda
```

- 内循环全开（Hebbian/Critic/SelfModel 在线运行）
- `--no-internal-loop`：纯生成对照；`--arch-self-mod`：**架构自修改（C2 已修复，可安全开启）**
- `stats`：internal steps / loss_int / sigma / can_ask / state / asked / 最近内循环异常

### 部署测试结论（2026-08-07）

- 内循环稳定运行（500+ 步 loss_int 有界 ~0.5）
- 主动求证未触发：sigma < 0.5（当前状态不够困惑，`can_ask` 机制本身正常；可用 `--verify-threshold 0.4` 验证链路）
- SelfModel 未预热（SFT 不训练它）——部署时从随机状态在线学习
- 修复了部署崩溃 bug（pipeline `self.training` → `model.training`）
- **major_sleep 首次真正运行**（原 500 步必崩：`_safe_sgd_step` 收到 Parameter 列表）——修复后内循环可无限步运行，EWC/soft-reset/回放清理等长期机制第一次生效

---

## 7. 已知问题表

> 2026-08-07 全表清零：C1/C2/M2/M3/M5/M6/I2/D1 均已修复（见"已修复清单"）。

### 未修（无——全部已修复）

### 已修复

| ID | 位置 | 问题 | 修复方案 |
|----|------|------|---------|
| C1 | internal_loop.py:729-755 | Stage C backward 梯度泄漏进 transformer .grad | `forward_internal` 输入 detach + mini_distill/major_sleep backward 前 `model.zero_grad()` |
| C2 | model.py:314 + architecture | split/prune 后各层专家数不一 → torch.stack 崩 | `_learnable_sigmas` 改逐层标量聚合（不再跨层 stack）；`_add_layer` 专家数对齐源层；architecture `.item()` 防御 |
| M2 | feedback.py:116-129 | 正向词先匹配，"不对"含"对"判 +1.0 | 负向词表先匹配 + 正向改长短语 + 否定词前后缀正则兜底 |
| M3 | monitor + train 循环 | 降 LR 被 scheduler.step 覆盖 | `_lr_scale` 持久化乘数：apply_actions 记录，scheduler.step 后重新应用（三处循环统一） |
| M5 | internal_loop.py:818 | TD 用过去 V(s_{t-1}) | 延迟一步 TD(0)：缓存 `(s_{t-1}, r_{t-1})`，下步用 `r_{t-1}+γ·V(s_t).detach()` 训练 V(s_{t-1}) |
| M6 | data_pipeline vs train | teacher 存 T=1，student 用 T=2 | `generate_teacher_logits` 按 `temperature` 参数缩放存储（`log_softmax(logits/T)`）；文件升 v3 |
| I2 | pipeline.py:194 | 在线训练 CE 无 mask（prompt 也学） | `_train_on_full` 加 `reply_start` + `ignore_index=-100`，只监督回复段 |
| D1 | train_reflex:518 | DDP + IterableDataset 无 \_\_len\_\_ | `StreamingDataset.__len__ = 1<<30`（近似，sampler 可构造） |
| — | internal_loop 500 步崩溃 | major_sleep 的 `_safe_sgd_step` 收到 Parameter 列表调 `.parameters()` 崩 | `_safe_sgd_step` 兼容 Module/Parameter 双输入（major_sleep 首次真正运行） |
| — | pipeline `self.training` | 非 nn.Module 属性错误 | 改 `model.training` |
| — | generate_qa right-padding | decoder-only 右 pad 伤生成 | `padding_side='left'` |
| — | teacher gzip 后缀判断 | `.gz.partial` 写明文 | 后缀判断兼容 `.gz.partial` |
| — | SFT 截断 label | 最后 1 token 错位 | `labels[-1]=-100` |
| — | resume 跨阶段 LR | base_lrs 被旧阶段覆盖 | `_resume_phase` + `_restore_optimizer_state(phase)` |
| — | cosine 末端空转 | LR 归零后无效步 | `min_factor=0.1` + plateau 30% |
| — | warmup Critic TD | 同 M5 方向问题 | 改纯 MC 回归 |
| — | 磁盘满崩溃 | checkpoint 9.3G 无滚动 | fp16 存储 + 原子写 + 滚动保留 2 个 |
| — | teacher 全量生成 | 115 万条 → 1.1TB | reservoir 采样 + max_samples + top-64 + gzip |

---

## 8. 设计哲学

- **混沌边缘**：噪声 + KL 好奇 + 辩证缓冲维持"有序-混乱"边界
- **Hebbian 非反向传播**：专家权重由局部梯度强化（focal_update：动量 + 激活门 + 双 clip）
- **固化解耦**：plastic（快学）→ consolidation（慢固化）→ stable（长期记忆）
- **惊讶驱动**：sigma/TD-error → gamma 调制更新强度——"不懂才用力学"
- **涌现冷却**：回答→学习→sigma 降→不再求证（无硬编码冷却）
- **教师蒸馏**：0.77B 学 1.5B Instruct；实测硬标签 SFT > 软标签 KD（容量限制）
- **内部循环不"看数据"，它"想"**：SelfModel 预测自己的下一步状态，Hebbian 强化被路由的专家，用户输入只是状态空间的扰动
