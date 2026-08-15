# NeuroStream-Reflex 内循环架构升级 v2

> **哲学前提**：思考不是预测错误的工具。思考是系统维持自身内部动力学丰富性的方式。预测是思考的副产物，不是目的。

---

## 0. 核心分歧：对"思考"的定义决定架构

当前内循环面临一个根本性的设计张力：

| | 路线 A：预测纠正机 | 路线 B：思维空间（本方案） |
|--|-------------------|------------------------|
| 驱动力 | 最小化预测误差 | 最大化内部动力学相干性 |
| 终止条件 | MSE → 0 → 停止思考 | KL 散度自然振荡 → 永不停止 |
| 与外部的关系 | 预测外部世界/自己 | 维持自持的闭环动力学 |
| 失败模式 | 真空中的完美预测器 | 混沌发散（受 Lyapunov 约束） |
| 生物类比 | 反射弧 | 默认模式网络（DMN） |

**本方案选择路线 B**。Contemplator 的核心职责不是"预测下步状态"，而是**生成可探索的内部世界**——在这个世界里，模型可以想象、反事实推理、维持矛盾想法、直到它们自然合成。

### 0.1 对"预测自己"批评的直接回应

之前的讨论提出 RSSM 的 KL 散度作为驱动力，被批评为"功利主义的自我纠错"。

**这个批评在 KL 散度的标准解释下成立**——如果 KL 散度只被理解为"我猜错了多少"。

但 KL 散度有第二个等价的解释：

> KL[Posterior || Prior] = **当你获得新信息后，你的世界观改变了多少**

这个量不是"错误"，而是"惊讶"（surprisal）。一个有生命的认知系统追求的**不是消除惊讶，而是最大化惊讶的信息价值**——这正是好奇心（curiosity）的数学形式（Schmidhuber 1991, Oudeyer 2007）。

### 0.2 本方案的关键机制变更

```
当前内循环：
  Contemplator → u_next → MSE 对齐 forward_internal 的输出
  └── 驱动力：MSE 下降 → 停止

v2 内循环：
  Contemplator → h_t (循环状态) + z_t (随机想象) → forward_internal(h_t, z_t)
  └── 驱动力：KL[P(z|o) || P(z)] 作为惊讶/好奇心信号
  └── 永不为零，因为世界和自我都有内在随机性
  └── 噪声调度由 KL 自适应驱动，而非硬编码
```

---

## 1. 架构总览：升级后的内循环

```
                    ┌─────────────────────────────┐
                    │      EndoSphere / h, z       │
                    │  (RSSM 循环状态 + 随机状态)    │
                    └──────────┬──────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
   Prior Network         GRU Cell              Posterior Network
   z_prior ~ P(z|h)      h_t = GRU(...)        z_post ~ Q(z|h,o)
          │                    │                    │
          └───────┬────────────┴──────────┬────────┘
                  │                       │
                  ▼                       ▼
           MoE Transition           Decoder/Imagination
           (多专家预测不同             z_t → o_pred
            世界动态模式)                 │
                  │                       │
                  └───────────┬──────────┘
                              │
                              ▼
                     forward_internal(h_t, z_t)
                              │
                              ▼
                      Actual Observation o_t
                              │
                              ▼
                     KL[Posterior || Prior]
                     + Reconstruction Loss
                     + Lyapunov Stability
```

### 1.1 关键差异：不再是你说的"预测自己"

**v1（当前）**：
```
v_t → Contemplator → u_next → target = Contemplator(u_next)
└── 自指但无状态：每步独立，没有时间保持
└── MSE 损失是"我应该像我想的那样想"
```

**v2（升级后）**：
```
h_{t-1}, z_{t-1} → GRU → h_t → Prior(z_t|h_t) → 想象 z_t
                                     → forward_internal(h_t, z_t) → o_t
                                     → Posterior(z_t|o_t) → 纠正世界观
└── 有状态循环：h_t 保持时间结构
└── KL 散度的解读：不是"我预测错了"，而是"这个世界观新信息量丰富"
└── z_t 的随机性保证思考不会收敛到固定点
```

### 1.2 为什么这不是"自我对齐"

| 特征 | "自我对齐" | "思维空间"（本方案） |
|------|-----------|-------------------|
| 目标 | 减少差异 | 维持丰富的内部动力学 |
| 随机性 | 噪声是工程的必要恶 | z_t 是思维的核心载体 |
| MoE 定位 | 提高预测精度 | 不同专家对应不同的思考模式 |
| 循环状态 | 可有可无 | h_t 是"正在想什么"的核心 |
| 终止 | 收敛 | 永远在混沌边缘振荡 |

---

## 2. 模块级升级说明

### 2.1 Contemplator → S elfModel (GRU + MoE Transition)

**当前实现**（`core/model.py:191-194`）：
```python
self.contemplator = Expert(d_model, cont_hidden_dim, dropout, baseline_lr=cont_lr)
```
一个单独 Expert 实例，无状态、无循环、无随机性。

**升级后的设计**（全部可学习，无程序性分支）：

```python
class SelfModel(nn.Module):
    """
    v2 Contemplator: 有状态的内部世界模型。

    功能不是"预测 u_next"，而是"维持 h_t（正在想什么）+ z_t（想象内容）"。
    """
    def __init__(self, d_model, n_experts, ...):
        # 1. GRU Cell: 循环状态保持
        self.gru = nn.GRUCell(d_model * 3, d_model)  # 输入: h, z, action

        # 2. Prior Network: P(z_t | h_t) — 不依赖观测，纯想象
        self.prior = nn.ModuleList([
            MoEPriorExpert(d_model) for _ in range(n_experts)
        ])

        # 3. Posterior Network: Q(z_t | h_t, o_t) — 被观测纠正的信念
        self.posterior = nn.ModuleList([
            MoEPosteriorExpert(d_model) for _ in range(n_experts)
        ])

        # 4. Decoder: z_t, h_t → o_pred (想象的输出)
        self.decoder = MoEDecoder(d_model, n_experts)

    def forward(self, h, z, action):
        # 时间步进
        h_next = self.gru(torch.cat([h, z, action], dim=-1), h)

        # 先验想象（不依赖观测）
        z_prior = self.prior(h_next)

        return h_next, z_prior

    def observe_and_correct(self, h, o):
        # 后验纠正（依赖观测）
        z_post = self.posterior(h, o)
        return z_post
```

### 2.2 内循环 Loss 升级

**当前**（`loop/internal_loop.py:536-550`）：
```python
mse = F.mse_loss(pred_emb, target_emb)
lyap = F.mse_loss(cont, v_t)
return mse + lyap_lambda * lyap
```

**升级后**：
```python
# 三目标 loss，每一项有独立的理论支撑
L_imagination = ‖decoder(z_t_post, h_t) - o_t‖²
  └── 确保想象的内容与观测一致（不是自我对齐，而是 grounded）

L_curiosity = β * KL[Q(z_t|h_t,o_t) || P(z_t|h_t)]
  └── 核心驱动力：Observing o_t changed my beliefs by KL bits
  └── 这不是"纠正误差"，这是"惊讶的有信息量"
  └── β 是温度参数，高 β → 探索倾向，低 β → 利用倾向

L_stability = λ * ‖h_t - h_{t-1}‖² + γ * ‖z_t‖²
  └── 防止混沌发散，保持思考在有序-混沌边界

L_total = L_imagination + L_curiosity + L_stability
```

**关键哲学转变**：
- `L_imagination` 不是"预测正确"，而是"想象有根据"（grounded）
- `L_curiosity` 不是"减少错误"，而是"最大化信息增益"
- `L_stability` 不是"不要改变"，而是"维持自同一性"

### 2.3 MoE 在升级后的角色

升级前 MoE 在 `ReflexMoELayer` 的 FFN 中，与 Router 一起做 token 级预测。升级后：

| MoE 位置 | 角色 | 路由依据 |
|----------|------|---------|
| `Prior` | 不同专家想象不同未来 | `h_t`（正在想什么） |
| `Posterior` | 不同专家对同一观测做出不同修正 | `h_t, o_t` |
| `Decoder` | 不同专家用不同方式重建 | `h_t, z_t` |
| `Router`（已有） | token 级 FFN | 不变，保持 |

**每个专家自然分化为不同的"思考模式"**：一个专家擅长想象对话发展，一个擅长推理链，一个擅长沉默等待。不需要硬编码。

### 2.4 EndoSphere / DialecticalBuffer 的升级

**当前**：`EndoSphereBuffer` 或 `DialecticalBuffer`——存储已完成的向量，下次取用。

**升级**：不再需要"存向量 → 取向量"。RSSM 的 `h_t` 本身就是"正在想什么"的活的状态。

```
当前：
  endosphere.push(v_t) → endosphere.get_latest() → v_t = endosphere[-1]

v2：
  h_t (GRU hidden state) 本身就是思维空间的当前状态
  z_t ≈ "当前想象的内容"
  └── 不需要 deque 存储历史，GRU 的 h_t 就是活的历史压缩
```

**DialecticalBuffer 的 thesis/antithesis/synthesis 三分池**可以保留为**外部记忆**（episodic memory），供 consolidation 使用，但不作为内循环的状态来源。内循环状态来自 `h_t`。

### 2.5 噪声驱动的升级

**当前**：`CriticalNoiseScheduler` 用 sigma 误差做 PI 控制。

**升级**：噪声由 KL 散度直接驱动。

```python
# v2 噪声规则
if KL[Q||P] > threshold_high:
    # 惊讶太大 → 降低噪声 → 收敛注意力
    noise *= 0.9
elif KL[Q||P] < threshold_low:
    # 太确定 → 增加噪声 → 探索新方向
    noise *= 1.1
```

这个机制比 PI 控制更本质：KL 散度是系统自己的"惊讶程度"，用它来调节探索，形成一个自稳定的混沌边缘动力学。

---

## 3. 执行路线图

### Phase 1: SelfModel 创建（核心架构变化）
- 新建 `core/self_model.py`：SelfModel（GRU + Prior/Posterior + Decoder）
- 修改 `config/model_config.py`：新增 SelfModel 配置项
- 修改 `core/model.py`：`contemplator Expert → self_model SelfModel`
- 不修改内循环逻辑，只替换 Contemplator

### Phase 2: 内循环 Loss 升级
- 修改 `loop/internal_loop.py`：`_compute_loss` → 三目标 loss
- 修改 `_execute_step`：增加 Posterior 观测 → KL 计算
- 修改噪声调度：KL 驱动替代 sigma 驱动

### Phase 3: h_t 取代 EndoSphere 作为内循环状态
- 修改 `_get_state` / `_init_state`：返回 h_t 而非 endosphere[-1]
- DialecticalBuffer 保留为外部记忆，但不再承包"当前状态"

### Phase 4: MoE Prior/Posterior 扩展
- 为 SelfModel 的 Prior 和 Posterior 添加 MoE 支持
- Router(h_t) 选择哪些专家参与想象

### Phase 5: 整合 ArchitectureModifier
- `internal_loop.py` 添加 Stage K 调用链
- 为 SelfModel 的专家添加增减支持

---

## 4. 验证原则

升级后的内循环是否成功，不依据"loss 更低"或"预测更准"，而依据：

| 指标 | 含义 | 测量方式 |
|------|------|---------|
| KL 散度不归零 | 系统始终有好奇心 | `KL[Q||P]` 随时间的变化：不应单调递减 |
| h_t 轨迹不收敛到固定点 | 思考在持续 | h_t 的 L2 变化率应持续 > 0 |
| MoE 专家分化 | 不同思考模式出现 | 各专家的激活频率分布应不均匀 |
| 内循环贡献可归因 | 内循环确实改善了外循环的生成质量 | 有/无内循环的对比实验 |

---

## 5. 升级原则总结

```
不要问"这个模块预测什么"
而要问"这个模块如何增加内部动力学的丰富性"

KL 散度不是误差，是惊讶
随机性不是噪声，是思维空间
GRU 状态不是记忆，是"正在想什么"
MoE 专家不是分工，是"不同的思考方式"
```
