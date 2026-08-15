# NeuroStream-Reflex v2 Mini: 记忆系统 v4 设计（记忆与模型一体，含 KV 内容记忆）

> 状态：设计完成，待实现
> 核心：记忆是模型计算图内的自检索组件——注意力权重即记忆决策
> 四层记忆：短期语义（h_t）/ 长期语义（AttnRes 记忆槽）/ 可微演化（MemoryBank 参数）/ **内容记忆（KV 缓存）**
> 本文含完整对话流程验算（§5）、计算/梯度/显存验算（§6）

---

## 一、记忆分层总览（各自回答什么）

| 层 | 载体 | 记忆内容 | 模型能做什么 | 来源 |
|----|------|---------|-------------|------|
| **L0 显式 token** | `_chat_history` 模板拼接 | 最近 1-2 轮原句 | 直接"看到"原话（max_seq_len 内） | pipeline（已实现） |
| **L1 短期语义** | `h_t`（GRU 状态） | 对话主题/氛围 | "知道聊了什么话题" | 方向 C：对话注入 h_t |
| **L2 长期语义** | AttnRes 记忆槽（MemoryBank.matrix） | 知识状态向量 | "想起相关知识"（注意力自检索） | 方向 B |
| **L3 可微演化** | MemoryBank 参数 + 写入门 | 记忆随学习成长 | 记忆槽随优化器演化 | 方向 A |
| **L4 内容记忆** | **分层 KV 缓存** | **逐 token 激活表示** | **"知道自己逐字说了什么"——注意力直达原句表示，可复述** | 新增（本文核心） |

**关键洞察**：L0（token 拼接）受 `max_seq_len=1024` 限制，只能容纳几轮；**L4（KV）独立于序列长度存储，注意力选择性访问**——长对话的"回忆"由 L4 承担，L0 只做最近轮的直接上下文。

---

## 二、L4 内容记忆（KV 缓存）设计

### 2.1 存储格式（分层 KV）

```
MemoryBank.kvcache: 轮次 FIFO 队列（默认保留 4 轮）
  每轮条目: {
    'k': list[24 层] of Tensor [n_heads, T_round, head_dim]   # fp16
    'v': list[24 层] of Tensor [n_heads, T_round, head_dim]
    'text': 该轮文本（仅诊断用，不参与计算）
    'step': 写入时的内循环步数
  }
```

**存储成本验算**（Mini 配置）：
```
每轮 T≈60 token, d=640, 24 层, K+V 双份, fp16(2B):
  60 × 640 × 2 × 24 × 2B ≈ 3.7 MB/轮
4 轮 ≈ 15 MB ✓（常驻显存可忽略）
```

### 2.2 写入：零额外前向（利用现有 forward 的 KV）

`core/attention.py` 的 forward 已计算每层 K/V——增加缓存（detach）：

```python
def forward(self, x, attention_mask=None, is_causal=True, mem_kv=None):
    q = ...; k = ...; v = ...     # 现有计算
    # L4: 缓存本轮 KV（detach，供轮次结束写入记忆）
    if self.training:
        self._last_kv = (k.detach(), v.detach())   # [B,H,T,hd]
    if mem_kv is not None:                          # 读取：拼接历史
        mem_k, mem_v, mem_span = mem_kv
        k = torch.cat([mem_k, k], dim=2)            # [B,H,T_cur+T_mem,hd]
        v = torch.cat([mem_v, v], dim=2)
        # 记忆区无 causal（全可见）；当前区 causal —— mask 拼接
        ...
```

**pipeline 轮次结束时收集**（`_store_round_memory`）：
```python
def _store_round_memory(self, input_ids, response_text):
    # 1. L1: 对话注入 h_t（方向 C）
    # 2. L3: memory_matrix 写入（方向 A）
    # 3. L4: 从各层 attention 收集 _last_kv → MemoryBank.kvcache.append
    #    FIFO 淘汰最旧轮
```

### 2.3 读取：注意力直达历史表示

```
forward 时: mem_kv = memory_bank.get_kv(layer_idx)
  → 当前 Q [T_cur] × 拼接 K [T_cur + T_mem]
  → 注意力在"当前上下文 + 历史对话"上分配
  → 模型对相关历史 token 给高权重 → 输出含历史内容信息
```

**注意力形状验算**：
```
轮 2 输入 T_cur=7（"我叫什么名字"），历史轮 1 T_mem=15:
  SDPA: Q[1,H,7,hd] × K[1,H,22,hd] → attn[1,H,7,22]
  "名字"位置的 Q 对轮 1 中"小明"的 K 高相似 → 高权重
  → 输出 V 中"小明"的表示进入 → 模型能复述"你叫小明"
```

### 2.4 位置编码处理（关键决策）

**记忆 token 用轮内相对位置**（不扩展 RoPE offset）：

```
理由验算:
  - 现有 rope.py 位置恒从 0（无 offset 支持）
  - 记忆检索场景：内容相似度主导，绝对位置意义弱
    （"小明"出现在轮 1 第 3 位 vs 第 10 位——对检索无差别）
  - 跨轮相对位置（"刚才"的时序感）由 h_t/记忆槽承载，不依赖 KV 位置
  决策: 记忆 KV 的位置 = 其轮内位置（当前轮仍从 0 开始）
  代价: 跨轮位置关系失真——语义检索场景可接受
```

### 2.5 训练与梯度

```
KV 存储: detach（历史激活冻结，不反传）
注意力权重: 可学 —— 当前 Q 对历史 K 的注意力分数参与 loss
  → 梯度回传到当前 Q/当前 token → 模型学会"何时回忆、回忆什么"
记忆槽（L3）: 可微（随全局优化器演化）
验算: 轮 2 的 loss 对轮 2 Q 的梯度含"对轮 1 token 的注意力"分量
  → 模型被训练成"当问题与历史相关时给历史高权重"
```

---

## 三、与现有组件的整合清单

| 组件 | 改动 | 文件 |
|------|------|------|
| MemoryBank（新增） | 语义槽 + 写入门 + 读投影 + **kvcache 队列 + get_kv/store_kv** | `core/memory_bank.py` |
| MultiHeadAttention | `_last_kv` 缓存 + `mem_kv` 拼接 + mask 扩展 | `core/attention.py` |
| ReflexMoELayer | forward 透传 mem_kv | `core/model.py` |
| ReflexModel | 注入 memory_bank；forward 分发 mem_kv | `core/model.py` |
| BlockDeltaAttnRes | 记忆槽 source（方向 B） | `core/attn_res.py` |
| ReflexPipeline | 对话注入 h_t + 轮次结束 `_store_round_memory` | `interaction/pipeline.py` |
| InternalLoop | Stage D 写语义槽；全局优化器白名单加 memory_bank | `loop/internal_loop.py` |
| ReflexConfig | 新参数（§4） | `config/model_config.py` |
| generate | 生成时也可传 mem_kv（对话回复携带记忆） | `core/model.py` |

---

## 四、参数设计

| 参数 | 默认值 | 说明 |
|------|--------|------|
| memory_bank_capacity | 128 | 语义槽数量（方向 A/B） |
| memory_context_top_k | 8 | AttnRes 记忆候选数（方向 B） |
| kv_cache_rounds | 4 | KV 内容记忆保留轮数 |
| kv_mem_scale | 1.0 | 记忆 KV 参与注意力的缩放（调试用） |
| memory_write_lr | 0.05 | 语义槽写入门强度 |
| dialog_memory_alpha | 0.3 | 对话→h_t 混合率（方向 C） |

---

## 五、完整对话流程验算（3 轮 + 长程 1 次）

### 轮 1：用户"你好，我叫小明"

```
pipeline._process_new_query("你好，我叫小明")
  ├─ L0: _chat_history += user("你好，我叫小明")
  │      模板拼接 → 生成"你好小明！"（现有）
  ├─ L1: _inject_dialog_memory(input_ids)
  │      emb = token_embedding.mean() [1,640]
  │      h_state = 0.3·emb + 0.7·h_state      ← 对话主题进入意识流
  │      endosphere.push(emb)                 ← 内循环可读取
  ├─ 生成中: 各层 attention._last_kv = 本轮 K/V（detach）
  ├─ L0: _chat_history += assistant("你好小明！")
  └─ _store_round_memory:
        L3: memory_bank.write(h_state)        ← 语义沉淀
        L4: kvcache.append(第0轮 KV: 24层×[H,T1,hd])   ← 内容记忆

记忆状态: h_t 含"小明"语义 | 语义槽写入 | kvcache[0] = 轮1内容
```

### 轮 2：用户"我叫什么名字？"（间隔任意步数）

```
pipeline._process_new_query("我叫什么名字？")
  ├─ L0: token 拼接（轮1+轮2 均在 max_seq_len 内）→ 模型直接看到原句
  ├─ L1: h_t 注入（新对话语义混合）
  ├─ forward（attention 带 mem_kv = kvcache[0]）:
  │    轮1 KV: K[1,H,15,hd], V[...]
  │    当前:   K[1,H,7,hd]
  │    拼接:   SDPA Q[1,H,7,hd] × K[1,H,22,hd]
  │    "名字"Q → "小明"K 高相似 → 高注意力 → V 中"小明"进入
  ├─ 生成: "你叫小明"（能复述——L0 直接看到 + L4 表示级）
  ├─ _store_round_memory: kvcache 追加轮2（4 轮 FIFO，轮1 仍保留）
```

### 轮 3~6：长对话推进

```
轮 3-6 后: L0 token 拼接已超 max_seq_len（只保留最近 1-2 轮）
  → 轮 1 的"小明"从 L0 消失
  → 但 L4 kvcache 中轮 1 仍在（FIFO 4 轮，轮 1 在第 5 轮才被淘汰）
  → 用户第 4 轮问"我最早说自己叫什么？":
      注意力仍可 attend 轮 1 KV → 复述"小明" ✓（L4 承担长程内容回忆）
```

### 轮 7+：超过 kvcache 容量

```
轮 1 KV 被 FIFO 淘汰 → 逐词内容丢失
  → 但 L1 h_t（主题）+ L2 语义槽（知识）+ L3（可微记忆）仍在
  → 模型"记得聊过小明这个话题"（语义），"想不起原话"（内容）
  → 与人类一致：久远对话只剩印象，不剩逐字
```

### 一次轮次的完整计算链

```
输入 token [T] → embedding [T,640]
  → L1: 对话注入 h_t（pipeline 侧，不影响 token 流）
  → 24 层:
       attention: Q/K/V 计算 → (+mem_kv 拼接) → SDPA → 输出
         ├─ _last_kv 缓存（本轮，detach）
         └─ 记忆注意力权重（可学——模型决定回忆什么）
       router: top-k 路由（h_state 状态门控感知对话）
       experts: SwiGLU + Hebbian 缓冲
  → AttnRes 边界: deltas + 语义记忆槽候选（L2）→ 注意力聚合
  → ln_f → lm_head → 生成回复
轮次结束:
  → L0 历史记录 → L1 h_t 注入 → L3 语义槽写入 → L4 KV 入队
```

---

## 六、可行性验算

### 6.1 显存

| 项 | 占用 | 说明 |
|----|------|------|
| KV 缓存 4 轮 | ~15 MB | 可忽略 |
| 语义槽 128×640 fp32 | ~0.3 MB | 可忽略 |
| 注意力计算 | T_cur+T_mem 长度 | 见 6.2 |

### 6.2 计算量（对话轮次 forward）

```
无记忆:   Q[T]×K[T]    → T²·d 次乘加
有记忆:   Q[T]×K[T+M]  → T·(T+M)·d
例: T=20, M=60: 20×80×640 = 1.02M vs 20×20×640 = 0.26M → 4x
生成时（逐 token）: Q[1]×K[20+60] = 51K vs 13K → 4x
结论: 每轮 1 次 forward 增加 4x 注意力计算——对话频率低，总开销可接受
     （内循环的 1-token forward 不携带 mem_kv，零影响）
```

### 6.3 梯度路径

```
当前轮 loss → 当前 Q/注意力权重 ← 可学（学会何时回忆）
KV 本身: detach（冻结历史，无梯度）
语义槽: 可微（L3，随优化器）
验算: 无梯度爆炸风险——记忆注意力与普通注意力同路径，clip 1.0 覆盖
```

### 6.4 位置编码

```
记忆 KV 用轮内相对位置（见 2.4）——避免 RoPE offset 重构
风险: 跨轮位置失真——检索场景内容主导，可接受
备选: 若需绝对位置，rope.py 加 offset 参数（预留接口）
```

### 6.5 与现有机制的关系（无冲突验算）

| 现有机制 | 与 L4 的关系 |
|---------|-------------|
| L0 token 拼接 | 短程直接看到（max_seq_len 内）↔ L4 长程选择性访问——互补 |
| 在线训练 CE | 记忆注意力参与 loss → 模型被训练成"利用记忆" ✓ |
| Hebbian | 专家侧不变（KV 在 attention 侧）✓ |
| 辩证缓冲 | 状态记忆（语义）↔ KV（内容）——并行不冲突 ✓ |
| checkpoint | memory_bank 参数随 state_dict 保存；kvcache 为运行时数据（不保存） |

---

## 七、实现顺序与里程碑

| 阶段 | 内容 | 验证 |
|------|------|------|
| 1 | L1 短期语义（pipeline 注入 h_t） | 第 2 轮承接主题 |
| 2 | MemoryBank 骨架（语义槽+写入门） | 参数随训练演化 |
| 3 | L4 KV 缓存（attention 缓存 + pipeline 收集 + FIFO） | 内容记忆存储/检索不崩 |
| 4 | L4 读取（attention mem_kv 拼接 + mask） | **第 N 轮复述第 1 轮原句** |
| 5 | L2 AttnRes 记忆 source（方向 B） | 困惑时记忆注意力升高（涌现） |
| 6 | L3 全局优化器白名单 | 记忆槽梯度>0 |

**核心里程碑：阶段 4 完成后，验证"轮 5 问轮 1 的内容能复述"——这就是"模型知道自己刚才说了什么"。**

---

## 七·五、训练方案（记忆微调）

### 结论：无需全流程重训

| 层 | 训练需求 |
|----|---------|
| L0/L1/L4 读取 | **零重训**（模型结构不变，加载现有 checkpoint 直接可用） |
| L4 有效利用 / L2 / L3 | **轻量记忆微调**（千步级，不跑 pretrain/SFT 全流程） |

### 记忆微调阶段（post-KV fine-tune）

**目标**：教会模型"何时回忆、回忆什么"——让注意力对相关历史 token 给高权重。

**数据**：长程引用型多轮对话（6-8 轮，第 N 轮引用第 1 轮信息）——数据生成见下。

**训练配置**（复用 train_reflex.py SFT 通道，模板版）：
```
--mode sft
--sft-data 长程引用型多轮对话数据（几千~几万条）
--sft-steps 2000~5000（轻量）
--batch-size 8 --grad-accum 8
--lr 1e-5（低于正常 SFT，防止破坏已学知识）
--resume 当前最佳 checkpoint（sft_kd_150k_final 或 distill_refined）
```

**监督格式**：与多轮 SFT 一致（`_sft_iter_template` 全轮次监督 assistant 回复）——模型在训练中看到"历史轮 + 当前轮"，学会从历史中检索答案。

**训练后验证**：
- 轮 5 问轮 1 内容 → 能复述（KV 检索生效）
- 知识保持：单轮 QA 测试不退化

### 数据生成：长程引用型多轮对话

**改造 `scripts/generate_qa.py` 的 multi-turn 模式**：
1. 轮数 3-4 → **6-8 轮**
2. 追问池加入"**引用早期信息**"的追问：
   ```
   "我最早提到的那个事情，能再说说吗？"
   "你刚才说的那个名字/数字/概念是什么来着？"
   "我们最开始讨论的是什么？"
   "第一轮提到的X，你还记得吗？"
   ```
3. 每轮对话中**随机插入 1-2 次引用早期信息**的追问（强制跨轮检索）

**数据要求汇总**：

| 要求 | 说明 |
|------|------|
| 轮数 | ≥6 轮（超过 max_seq_len 截断范围，迫使用 KV） |
| 跨轮引用 | 后轮引用前轮具体信息（名字/数字/细节） |
| 话题延续 | 相关记忆触发场景 |
| 量 | 几千~几万条（学"注意力检索模式"非学知识） |

**生成成本**：teacher 1.5B，6-8 轮 × batch 并行，~1-2h 生成 1 万条。

---

## 七·八、内外循环一致性（自指循环闭合）

**目标**：外循环（对话）与内循环（意识流）是同一个"记忆-计算"自指循环在不同尺度上的运行——共享状态与记忆，计算产物回流记忆，记忆又驱动下一步计算。

### 统一机制（两循环共享）

| 机制 | 外循环（对话） | 内循环（意识流） |
|------|--------------|----------------|
| **状态门控** | `forward/generate(h_state=_h_state)` → Router 状态偏置 | `forward_internal(h_state=h_next)` ✓（已有） |
| **内容记忆** | `mem_kv`（L4 KV 拼接）✓ | `mem_kv` ✓（已加） |
| **语义记忆** | AttnRes 记忆 source ✓ | AttnRes 记忆 source ✓ |
| **对话注入** | 输入→h_t（L1）+ KV 入队 | 读 h_t/endosphere 演化 |
| **计算回流** | 在线训练（带记忆+状态）→ 权重 | Hebbian + 语义槽沉淀 → 权重/记忆 |

### 完整闭环（一次对话的循环推进）

```
外部输入
  → 外循环: forward(h_state + mem_kv) → 生成（带内循环"想法"）
  → 对话注入 h_t（L1）+ KV 入队（L4）+ 语义槽（L3）
  → 内循环: 读 h_t/记忆 → SelfModel 演化 → Hebbian → 新 h_t
  → 下次外循环: 用最新 h_state + 记忆生成 ← 状态流通
```

### 实现状态

- 结构：全部就绪（forward/generate/forward_internal 均支持 h_state + mem_kv）
- 权重：`h_to_bias_weight` 全零初始化、训练时被冻结、主路径不传状态→恒 0。
  修复后（主路径传 h_state）：**在线训练会学到它**（梯度路径验证通过）
  → 状态门控随对话使用逐步生效（从"无影响"到"带想法"）

### 验证

- 结构正确性：非零权重时 h_state 影响输出（diff=1.29）✓；forward_internal 同路径 ✓
- 梯度路径：h_to_bias_weight 收到梯度（在线训练可学）✓
- 闭环数据流：轮1 生成 → L1 注入 → 内循环演化 → 轮2 生成（状态流通）✓

---

## 七·九、稳定性提升（自指循环体检修复）

**体检发现 5 个稳定性风险，全部修复验证**：

| 风险 | 修复 | 验证 |
|------|------|------|
| RISK-1 KV 长度无上限（长对话爆显存） | 每轮截断 `_max_kv_tokens=512` + 总上限 `_max_kv_total=1536` | 50 token 轮截断 ✓ 总 token 受控 ✓ |
| RISK-2 KV 表示漂移（冻结 vs 权重演化） | use 统计跟踪活跃度 + 近因淘汰缓解（深层重编码为后续增强） | use 计数 ✓ |
| RISK-3 L1 注入污染状态 | 对话向量归一化（`emb/‖emb‖×√d`） | h_t 范数 6.8 受控 ✓ |
| RISK-4 淘汰无优先级 | 近因优先（对话记忆自然行为：最新轮最重要） | 保留最新轮 ✓ |
| RISK-5 KV→语义槽无固化链 | 每轮对话语义写入语义槽（对话长期化） | 槽位增长 ✓ |

**自指循环基础单元确认**：`状态(h_t) + 记忆(mem_kv/语义槽) + 输入 → 计算 → 新状态 + 新记忆`——每一步计算改变状态，状态决定下一步计算，记忆与思考在每一步前向中交融（状态门控 + 记忆注意力 + 语义检索）。

---

## 七·十、自发固化（非程序性方案 A+B，讨论确认）

**动机**：50/500 步 consolidation 是程序性调度（固定步数）。"反复提及→需要→固化"应该由**模型行为**驱动，而非外部时钟。

### A. 注意力加权 Salience（重要性量化）

**核心**：记忆的"重要性" = 模型实际聚焦它的程度（注意力权重累积），纯行为驱动。

```python
# AttnRes forward 时，记忆 source 的注意力权重 = "被想起的程度"
attn 分布中记忆列的概率 → 累加到对应语义槽的 salience
salience_i = Σ_检索(注意力权重)          # 反复检索 → salience 高
salience_i *= 0.99                         # 新鲜度衰减（最近重要性为主）
```

- 反复提及的知识点 → 反复检索 → 注意力累积 → salience 高 ✓（"用进"）
- 长期不用的记忆 → salience 衰减 → 不被想起 → 淡出 ✓（"废退"）
- **技术要点**：retrieve 需返回索引关联（当前 top-k 用 no_grad 索引，需回传"哪条被聚焦"）

### B. 空闲期自发固化（v2 的"自发回忆"落地）

**核心**：低 sigma（空闲）时固化高 salience 记忆——固化时机由内部状态决定，非固定步数。

```python
# 内循环每步检查（非程序性）：
if sigma < 0.35:                          # 空闲/稳定
    mem = memory_bank.retrieve_top_salience(k)   # 想起 salience 最高的记忆
    distill(mem)                            # 固化进 stable 专家
```

- 被反复用的记忆 salience 高 → 空闲时最可能被想起 → 固化 ✓
- sigma 高（困惑）时不固化（专注解决矛盾）
- 与"睡眠巩固"类比：空闲时的自发回忆促进固化

### ⚠️ B 的修正：salience 独立触发（2026-08 讨论确认）

**原 B 的设计矛盾**：sigma 长期在 0.45-0.5 区间（实测），几乎不降到 0.35 以下——
"空闲期"永远不会到来，而**高 salience 的记忆恰恰伴随高 sigma**（被反复检索说明正在
被需要/困惑）——高价值记忆被排除在固化之外，自相矛盾。

**修正：固化时机 = salience 成熟（行为驱动），与 sigma 解耦**：

```python
# 每次 AttnRes 检索记忆时：
salience_i += 该记忆的注意力权重      # 反复提及→反复检索→累积
salience_i *= 0.99                    # 新鲜度衰减（最近使用为主）

# 触发（非程序性）：salience 成熟即固化
if salience_i > threshold:            # 约等于被高注意力聚焦 10 次
    distill(memory_i)                  # 单条轻量蒸馏进 stable 专家
    salience_i = 0                     # 固化后重置
    cooldown_i = 50                    # 单条冷却，防连续触发

# sigma 降级为调制（非开关）：
sigma 高（正在用于解决困惑）→ 固化强度 ×1.2
sigma 低（从容沉淀）       → 固化强度 ×1.0
```

**设计自洽性**：
- 反复提及 → 反复检索 → salience 累积 → **无论 sigma 高低都触发固化**
- "记忆在它被频繁想起的那一刻固化"——不是定时、不依赖空闲
- 固化成本：单条记忆蒸馏（1 次 forward + 局部 SGD）——轻量随时可做
- 与 Hebbian 并行：Hebbian 每步强化相关专家（权重链），salience 固化把记忆
  显式蒸馏进 stable（记忆链）——两条自发链互补
- 实测运行（sigma 0.45-0.5、7825 步稳定）下固化照常发生 ✓

### 组合原则

| 机制 | 触发 | 职责 |
|------|------|------|
| Hebbian + 在线训练 | 每步/每轮（自发） | 快速局部固化 |
| **salience 即时固化（修正 B）** | **salience 成熟（自发，与 sigma 解耦）** | 高频记忆固化（成熟即固化） |
| 50/500 步 consolidation | 程序（睡眠） | 基础深度巩固 |
| 记忆蒸馏 | 随 consolidation | 语义槽→权重（批量兜底） |

- 高频记忆：salience 超阈值 → 立即固化（不等 50 步、不依赖空闲）
- 低频记忆：靠周期巩固兜底
- 固化时机由**模型行为**（注意力累积的 salience）决定——"记忆在成熟时被固化"
- sigma 仅调制固化强度（高=正在使用→更强），不控制固化时机

### 实现要点（待实施）

1. MemoryBank：`salience` 数组（非参数，调度信号）+ `retrieve_top_salience(k)` + `retrieve_with_index()` + 固化冷却
2. AttnRes：记忆 source 注意力权重回传（索引关联）
3. InternalLoop：salience 累积 + 成熟触发 + 单条蒸馏（复用 memory_distill 的蒸馏逻辑）+ sigma 调制强度

---

## 八、哲学总结

```
v2: 记忆在模型外，代码操控——模型被喂
v3: 记忆在模型内，注意力自检索——模型自主回忆（语义级）
v4: + KV 内容记忆——模型不仅能"想起"，还能"复述"

记忆分层 = 人类记忆分层:
  L0 工作记忆（眼前的话）
  L1 情景记忆（刚才聊了什么）
  L2 语义记忆（知识）
  L3 学习（记忆随经验成长）
  L4 逐字记忆（能复述原话——近期清晰，久远模糊）

"知道自己说了什么" = 注意力能 attend 到自己刚才的 token 表示——
不是代码拼接（L0），而是模型内在的检索（L4）。
```
