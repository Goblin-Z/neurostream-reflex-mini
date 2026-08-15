# NeuroStream-Reflex 架构 v2 文档

## 概述

NeuroStream-Reflex v2 在保留原有意识流架构（MoE 专家、Hebbian 局部更新、辩证法、意识内循环、consolidation 巩固）的基础上，全面升级了 Transformer 底层架构，引入 2024-2026 年的最新研究成果。

## 架构升级清单

### 1. 注意力现代化

| 组件 | v1 (旧) | v2 (新) | 论文来源 |
|------|---------|---------|----------|
| 位置编码 | 可学习绝对位置嵌入 | **RoPE** (Rotary Position Embedding) | RoFormer (2021) |
| QKV 结构 | 单 Linear(d→3d), 等头分割 | **GQA** (Grouped Query Attention) | LLaMA-2 (2023) |
| Q/K 稳定性 | 无 | **QK-Norm** (RMSNorm on Q, K) | Gemma-2 (2024) |
| 注意力计算 | 手动 einsum + softmax | **Flash Attention** (F.scaled_dot_product_attention) | FlashAttention (2022) |

**RoPE** (`core/rope.py`):
- 旋转位置编码，通过复数平面旋转编码相对位置
- 移除 `position_embedding` buffer，永久修复位置嵌入溢出问题
- θ=10000, 支持长度外推
- 预计算 sin/cos 缓存，O(1) 查找

**GQA** (`core/attention.py`):
- Q 头数 = 16, KV 头数 = 4 (4:1 比例)
- KV cache 减少 75%，推理加速
- KV 头通过 `repeat_interleave` 扩展到 Q 头数

**QK-Norm**:
- 每个 Q/K 头独立 RMSNorm，防止 attention logits 爆炸
- 无偏置，初始化 scale=1

**Flash Attention**:
- 使用 PyTorch 2.0+ `F.scaled_dot_product_attention`
- 自动选择 Flash/efficient kernel
- 显存 O(N) 而非 O(N²)

### 2. 归一化现代化

| v1 | v2 | 理由 |
|----|-----|------|
| LayerNorm (weight+bias) | **RMSNorm** (仅 weight) | 无均值减法，快 10-20%，效果相当 |

**替换位置**: LN1, LN2 (每层), LN_f (最终层), Expert 内部, Critic 内部, QK-Norm, AttnRes post_norm

文件: `core/rmsnorm.py`

### 3. FFN 现代化 - SwiGLU

| v1 | v2 | 理由 |
|----|-----|------|
| `GELU(x@W1) @ W2` (2矩阵) | `SiLU(x@W_gate) ⊙ (x@W_up) @ W_down` (3矩阵) | LLaMA/PaLM 验证 |
| d_ff = 4096 | d_ff = 2730 (≈2/3×4096) | 保持参数量不变 |

文件: `core/expert.py`

**SwiGLU 前向**:
```
g_pre = x @ W_gate           # [*, d_ff]
g     = SiLU(g_pre)          # Swish 激活
u     = x @ W_up             # [*, d_ff]
h     = g ⊙ u                # 门控隐状态 [*, d_ff]
out   = h @ W_down           # [*, d_model]
sigma = sigmoid(uncertainty_head(h))
```

**Hebbian 更新适配** (`learn/hebbian_update.py`):
```
∂L/∂W_down = grad_y^T @ h
∂L/∂h = grad_y @ W_down
∂L/∂g = ∂L/∂h * u           (链式穿过 ⊙)
∂L/∂u = ∂L/∂h * g
∂L/∂g_pre = ∂L/∂g * SiLU'(g_pre)
∂L/∂W_gate = ∂L/∂g_pre^T @ x
∂L/∂W_up = ∂L/∂u^T @ x
```

Expert 缓存扩展: `_input, _gate_pre, _gate, _up, _hidden, _output`

### 4. Block Delta Attention Residuals (核心创新)

融合三篇论文:
- **Attention Residuals** (arXiv:2603.15031, Kimi Team 2026.03)
- **Delta Attention Residuals** (arXiv:2605.18855, 2026.05)
- **Low-Rank Attention Residuals** (arXiv:2607.09694, 2026.07)

文件: `core/attn_res.py`

**问题**: 标准 PreNorm 残差以固定 +1 权重累加所有层输出，导致:
- 隐状态随深度无界增长 (PreNorm dilution)
- 深层贡献被稀释

**解决方案**: 用 softmax 注意力替代固定残差累加，实现选择性跨层聚合。

**架构** (Block Delta AttnRes):
```
12 层分为 3 个 Block (每 Block 4 层)
Block 内: 标准 PreNorm 残差
Block 间: Delta Attention Residual 聚合
```

**Delta 机制**: 对增量 δ_j = h_j - h_{j-1} 做注意力（非累积状态）
- 累积状态高度冗余 → 路由坍缩 (max weight ≈0.2)
- 增量表示多样 → 高对比度路由 (max weight ≈0.6)

**低秩键**: K 投影到 r=d/4 维，降低开销

**初始化**: W_v=0, post_norm scale=1e-3 → 训练起始等价于标准 Transformer

**PostDA-Norm**: `h = h + RMSNorm(AttnRes_output)` (残差式添加)

**Sink mitigation**: delta L2-normalized before K projection (direction-based routing). V uses unnormalized delta (full info preserved).
- Prevents delta_0 (h_0, magnitude ~32) from dominating attention over subsequent deltas (magnitude ~4-8)

**Routing monitor**: training logs ar0_max (max attn weight) and ar0_ent (attn entropy) every 100 steps
- max_weight > 0.4 = healthy routing; max_weight ~ 1/n_sources = routing collapse

### 5. 权重绑定

```python
self.lm_head.weight = self.token_embedding.weight  # 绑定
```
- 减少约 155M 参数 (151936 × 1024)
- lm_head 无 bias
- 正则化效果

### 6. MoE 改进 (保留原有设计)

- **Router Z-loss**: 防止 router logits 爆炸 (训练时添加)
- **Load balancing loss**: 确保专家均衡使用
- **EWC**: 通过 object identity 保护 stable expert (已适配 SwiGLU 参数名)

### 7. 全局优化器集成

AttnRes 参数纳入 AdamW 全局巩固:
```python
# internal_loop.py 中的参数过滤器
['attention', 'router', 'lm_head',
 'ln_f', 'ln1', 'ln2',
 'q_norm', 'k_norm', 'q_proj', 'kv_proj', 'o_proj',
 'attn_res', 'post_norm']
```

AttnRes 参数:
- 由全局 AdamW (lr=1e-6) 在 consolidation 时更新
- 不受 Hebbian 局部更新影响
- 不受 EWC 保护 (只有 stable expert 受保护)

---

## 保留的关键设计 (全部验证通过)

| 设计 | 状态 | 说明 |
|------|------|------|
| MoE stable/plastic 专家分离 | ✅ | stable (低LR + EWC) vs plastic (高LR + Hebbian) |
| Hebbian 局部更新 | ✅ | `focal_update` 适配 SwiGLU 3 矩阵梯度 |
| 内部意识流 | ✅ | `forward_internal` 走完整层 + AttnRes |
| Contemplator | ✅ | 独立 Expert 实例，预测世界模型 |
| DialecticalBuffer | ✅ | 辩证法 (thesis/antithesis/synthesis) |
| CriticalNoiseScheduler | ✅ | PI 控制器自调节噪声 |
| Consolidation | ✅ | mini (50步) + major (500步) + EWC + 全局回放 |
| GradientManager | ✅ | 逐层梯度提取，O(n) 复杂度 |
| EndoSphere | ✅ | 内部状态注入 Router |
| Critic | ✅ | TD-error 调制 gamma |
| InteractionManager | ✅ | 交互状态机 |
| ReplayBuffer | ✅ | 优先级回放巩固 |

---

## 完整前向路径 (v2)

```
token_emb(x)                                    # [B, T, d]  (无 position embedding)
    │
    ▼
┌─ Block 1 (layers 0-3) ──────────────────────┐
│  for l in [0,1,2,3]:                         │
│    h = h + Attn(RMSNorm(h))                  │  ← GQA + RoPE + QK-Norm + Flash
│    h = h + MoE(RMSNorm(h))                   │  ← SwiGLU experts + Router
│  block_outputs = [h_0, h]                    │
└──────────────────────────────────────────────┘
    │
    ▼ AttnRes(deltas=[h_0, h-h_0])              ← Delta Attention Residual
    │   Q = W_q(RMSNorm(h))  [B,T,r]
    │   K = W_k(δ_j)          [B,T,n,r]  (低秩)
    │   V = W_v(δ_j)          [B,T,n,d]  (全维)
    │   h = h + post_norm(out_proj(softmax(Q·K^T)·V))
    │
┌─ Block 2 (layers 4-7) ──────────────────────┐
│  ... 同上 ...                                 │
└──────────────────────────────────────────────┘
    │
    ▼ AttnRes(deltas=[h_0, δ_1, δ_2])
    │
┌─ Block 3 (layers 8-11) ─────────────────────┐
│  ... 同上 ...                                 │
└──────────────────────────────────────────────┘
    │
    ▼
RMSNorm -> lm_head (tied with token_emb) -> logits
```

---

## 单卡 H800 云端训练方案

### 硬件配置

| 项目 | 规格 |
|------|------|
| GPU | H800 80GB × 1 |
| 存储 | 高性能 NVMe SSD |
| 内存 | ≥ 128 GB CPU RAM |
| 网络 | 用于下载 teacher 模型和数据集 |

### 显存估算 (单卡 H800 80GB)

| 组件 | bf16 显存 |
|------|----------|
| 模型权重 | ~3.4 GB |
| 梯度 | ~3.4 GB |
| AdamW 优化器状态 | ~13.6 GB |
| 激活值 (bs=64, seq=2048) | ~15-20 GB |
| 杂项/预留 | ~10 GB |
| **总计** | **~45-50 GB / 80 GB** |

> 预留 30GB+ 安全余量，支持 micro-batch 最大到 64 (pretrain) 而无需激活重计算。

### 模型规模 (ReflexMediumConfig)

| 参数 | 值 |
|------|-----|
| d_model | 1024 |
| n_layers | 12 |
| n_heads | 16 (Q), 4 (KV) |
| d_ff | 2730 (SwiGLU ~2/3) |
| n_experts/layer | 12 (9 stable + 3 plastic) |
| top_k | 3 |
| max_seq_len | 2048 |
| AttnRes block_size | 4 (3 blocks, 2 boundaries) |
| AttnRes rank | 256 (d_model/4) |
| 总参数 | ~1.5-1.7B |

### 三阶段训练 (单卡 H800)

通过 **micro-batch + gradient accumulation** 达到与多卡相同的等效 batch size:

| 阶段 | 数据 | Tokens | Steps | Micro-BS | Grad Accum | 等效 Batch | LR | 时间估算 |
|------|------|--------|-------|----------|------------|------------|-----|---------|
| 1. 预训练 | Wudao 10% | 20B | 230k | 64 | 2 | 128 | 3e-4 | ~3-4 天 |
| 2. SFT | Firefly+BELLE | 2B | 45k | 32 | 2 | 64 | 5e-5 | ~12-15 小时 |
| 3. 蒸馏 | Self-Instruct | 500M | 23k | 16 | 2 | 32 | 2e-5 | ~6-8 小时 |
| **总计** | | **22.5B** | **298k** | | | | | **~4-5 天** |

> 时间估算假设 H800 每步 1.5-3 秒 (pretrain bs=64, top_k=3) 到 0.8-1.5 秒 (distill bs=16)。

### 梯度累积 (Gradient Accumulation)

单卡 H800 80GB 无法直接容纳等效 batch 128，因此将 micro-batch 减半并通过 **2 步梯度累积** 还原等效 batch size:

```python
# Pretrain 示例
micro_batch = 64
grad_accum = 2
loss = loss / grad_accum          # 每 micro-batch 损失缩放
loss.backward()                   # 梯度累加 (不 zero_grad)

if step % grad_accum == grad_accum - 1:
    clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()              # 等效 batch = 64 × 2 = 128
    optimizer.zero_grad()
    scheduler.step()
```

**优点**:
- 单卡即可达到与 8 卡 DDP 相同的等效 batch size
- 避免多卡通信 overhead
- 云端单卡租赁成本通常低于多卡

**监控适配**: 梯度范数只在累积完成后的 update step 检测，避免 partial gradient 误报。

### 启动方式

```bash
# 直接单卡启动
python train/train_reflex.py --mode full --device cuda --dtype bfloat16

# 使用脚本
bash train/run_training.sh

# 仅 SFT
bash train/run_training.sh --mode sft

# 调整 micro-batch / grad-accum (显存不足或充足时)
bash train/run_training.sh --batch-size 32 --grad-accum 4  # 等效 128

# 从 checkpoint 恢复
bash train/run_training.sh --resume /checkpoints/reflex/pretrain_final.pt
```

### DDP 兼容 (可选)

训练脚本仍自动检测 `RANK`/`WORLD_SIZE` 环境变量并初始化 DDP。如果未来切换到多卡，无需修改代码:

```bash
# 8 卡 DDP (未来可扩展)
python -m torch.distributed.run --nproc_per_node=8 \
    train/train_reflex.py --mode full --device cuda --dtype bfloat16
```

但默认脚本 `run_training.sh` 已改为单卡启动。

### 监控系统

`train/monitor.py` 提供四合一监控:
- **梯度监控**: 爆炸/消失检测，自动 LR 缩放
- **专家监控**: 死专家检测，自动重置
- **Sigma 监控**: 崩塌检测，噪声注入
- **权重监控**: 异常增长检测

### 数据路径配置

| 数据 | 默认路径 | 格式 | 下载来源（国内镜像） |
|------|---------|------|---------------------|
| 预训练 | `/data/wudao/wudao_10pct.jsonl` | JSONL | [OpenDataLab](https://opendatalab.com) |
| SFT | `/data/firefly/firefly_1.6m.jsonl` | JSONL (conversational) | [HF 镜像 / YangyiYin/Firefly](https://hf-mirror.com/YangyiYin/Firefly) |
| 蒸馏 | `/data/distill/self_instruct.jsonl` | JSONL + teacher logits | 由 Teacher 模型本地生成 |
| 输出 | `/checkpoints/reflex/` | PyTorch .pt | - |
| Teacher | `Qwen/Qwen2.5-7B-Instruct` | HuggingFace | 自动从 `HF_ENDPOINT` 下载 (预设 `hf-mirror.com`) |

> 启动脚本 `run_training.sh` 已预设 `HF_ENDPOINT=https://hf-mirror.com`，Teacher 模型和 Tokenizer 会自动走镜像下载。

### 训练配置参数 (TrainConfig)

```python
# 单卡 H800 默认配置
pretrain_batch_size = 64      # micro-batch
pretrain_grad_accum = 2       # gradient accumulation
pretrain_lr = 3e-4            # 等效 batch 128 的 LR

sft_batch_size = 32
sft_grad_accum = 2
sft_lr = 5e-5

distill_batch_size = 16
distill_grad_accum = 2
distill_lr = 2e-5
```

---

## 文件结构

### 新增文件
| 文件 | 说明 |
|------|------|
| `core/rmsnorm.py` | RMSNorm 实现 |
| `core/rope.py` | RoPE 旋转位置编码 |
| `core/attn_res.py` | Block Delta Attention Residuals |

### 修改文件
| 文件 | 修改内容 |
|------|----------|
| `core/attention.py` | 重写: GQA + RoPE + QK-Norm + Flash Attention |
| `core/expert.py` | 重写: SwiGLU FFN, 缓存扩展, 移除 gelu_derivative |
| `core/model.py` | 移除 position_embedding, 加 AttnRes, 权重绑定, RMSNorm |
| `config/model_config.py` | 新增 n_kv_heads, rope_theta, attnres_* 配置 |
| `learn/hebbian_update.py` | 重写: SwiGLU 3 矩阵梯度 |
| `learn/critic.py` | LayerNorm → RMSNorm |
| `learn/consolidation.py` | 注释更新, AttnRes 纳入全局回放 |
| `learn/fisher.py` | 注释更新 (w_gate/w_up/w_down) |
| `loop/internal_loop.py` | 全局优化器加 AttnRes 参数, global_drift 修复 |
| `interaction/pipeline.py` | proxy gradient 用 w_down 替代 fc2 |
| `train/train_reflex.py` | 单卡 H800 配置, 梯度累积, DDP 兼容 |
| `train/run_training.sh` | 单卡启动脚本 (兼容 DDP) |
| `train/monitor.py` | endosphere.push 加 sigma 参数 |

### 未修改文件 (架构无关)
- `core/router.py` - MoE 路由器 (不变)
- `loop/endosphere.py` - EndoSphere 缓冲区 (不变)
- `loop/dialectical_buffer.py` - 辩证法缓冲区 (不变)
- `loop/gradient_manager.py` - 梯度管理器 (不变)
- `learn/critical_noise.py` - PI 控制器 (不变)
- `learn/replay.py` - 回放缓冲区 (不变)
- `learn/fluid_roles.py` - 专家角色评估 (不变)
- `interaction/manager.py` - 交互状态机 (不变)
- `interaction/digestion.py` - 异步消化队列 (不变)
- `train/data_pipeline.py` - 数据管道 (不变)

---

## 验证测试

16 项测试全部通过:

1. ✅ position_embedding 已移除
2. ✅ 权重绑定 (lm_head = token_embedding)
3. ✅ 所有 Norm 为 RMSNorm
4. ✅ GQA 配置正确 (4 Q头, 2 KV头)
5. ✅ QK-Norm 存在
6. ✅ RoPE 存在
7. ✅ SwiGLU Expert (w_gate/w_up/w_down, SiLU)
8. ✅ AttnResStack 初始化正确 (近零起始)
9. ✅ 前向传播 (文本输入)
10. ✅ forward_embeddings (consolidation 用)
11. ✅ forward_internal (意识流)
12. ✅ SwiGLU Hebbian 更新 (3 矩阵均变化)
13. ✅ EWC stable 参数名 (198 个, object identity)
14. ✅ 内循环运行 (23步, 辩证事件检测, 85 全局参数)
15. ✅ 文本生成
16. ✅ AttnRes 在 block 边界触发

---

## 论文引用

- **Attention Residuals**: Kimi Team, arXiv:2603.15031, 2026.03
- **Delta Attention Residuals**: Luo et al., arXiv:2605.18855, 2026.05
- **Low-Rank Attention Residuals**: Su, arXiv:2607.09694, 2026.07
- **DenseFormer**: Pagliardini et al., arXiv:2402.02622, 2024.02
- **MUDDFormer**: Xiao et al., arXiv:2502.12170, 2025.02 (ICML'25)
- **Residual Stream Duality**: Zhang, arXiv:2603.16039, 2026.03
- **RoPE**: Su et al., RoFormer, 2021
- **GQA**: Ainslie et al., 2023
- **SwiGLU**: Shazeer, 2020
- **RMSNorm**: Zhang & Sennrich, 2019
- **Flash Attention**: Dao et al., 2022
