# NeuroStream-Reflex v2 Mini: 记忆系统 v3 设计（记忆与模型一体）

> 状态：设计完成，待实现
> 核心转变：v2 的记忆是"代码操控的外部仓库"→ v3 的记忆是"模型计算图内的自检索组件"
> 设计原则：模型通过自己的注意力/门控决定"何时回忆、回忆什么"——无外部 sigma 判断
> 依据：MEMORY_SYSTEM_v2_DESIGN.md 的三层方案 + 方向 A/B/C 改进分析

---

## 一、设计哲学

### v2 的根本问题

| v2 层 | 本质缺陷 |
|-------|---------|
| 记忆回放 | 代码判断 sigma → 代码采样 → 额外训练——模型被动接受 |
| 辩证张力 | 代码算张力 → 代码注入 loss——矛盾解决由外部规则驱动 |
| 记忆注入 | 代码检索 → `v_t += 0.1·context`——模型没有"选择权" |

**共同点：记忆在模型外面（缓冲数组），检索/判断/注入全是代码逻辑。模型永远不会"主动想起"。**

### v3 的核心转变

```
v2: 代码问记忆要数据 → 喂给模型          （外部操控）
v3: 记忆是模型的计算组件 → 模型自己检索    （自由交互）
```

**记忆访问 = 模型的注意力权重**。模型觉得需要回忆时，注意力自然给记忆高权重；不需要时自然遗忘——sigma 驱动的行为从模型内部**涌现**，而非代码判断。

---

## 二、三层记忆架构（与现有代码逐组件整合）

### 层 1：短期对话记忆（方向 C）——对话编码进 h_t

**设计**：用户对话通过 SelfModel 的 `h_t`（GRU 状态）自然编码——模型内在记忆，零外部检索代码。

**与现有代码的整合**（`interaction/pipeline.py` + `loop/internal_loop.py`）：

```
现状: pipeline 生成用 token 拼接（v3 保留，token 级短期上下文）
      pipeline 不碰 model._h_state（内循环持有）
改进: 用户输入 embedding → 注入 model._h_state（对话融入意识流）
```

```python
# interaction/pipeline.py —— _process_new_query 生成后：
def _inject_dialog_memory(self, input_ids):
    """方向C: 用户对话编码进 h_t（短期记忆，模型内在）"""
    sm = getattr(self.model, 'self_model', None)
    h = getattr(self.model, '_h_state', None)
    if sm is None or h is None:
        return
    with torch.no_grad():
        emb = self.model.token_embedding(input_ids).mean(dim=1)  # [1, d]
        alpha = self.config.dialog_memory_alpha   # 默认 0.3
        # 对话状态混合进 GRU 记忆（EMA 式）
        self.model._h_state = (
            alpha * emb + (1 - alpha) * h
        ).detach()
        # 同时 push 到辩证缓冲（内循环 get_latest 读取）
        self.model.endosphere.push(emb[0].detach().cpu(), sigma=0.5)
```

**效果**：内循环下一轮 `_get_state()` 取到含对话信息的 h_state → SelfModel 演化、Router 状态门控（`h_to_bias_weight`）、AttnRes 全部自然感知对话——**零代码检索，记忆即状态**。

**参数**：`dialog_memory_alpha = 0.3`（对话记忆混合率；太大覆盖意识流，太小无感知）

---

### 层 2：长期记忆（方向 B）——记忆向量作为 AttnRes 额外 source

**设计**：长期记忆存于 `MemoryBank`（可写可读），作为 **AttnRes 跨块注意力的额外 source**——模型自己的注意力在 [块 delta, 记忆候选] 之间分配权重。检索决策完全由模型做出。

**与现有代码的整合**（`core/attn_res.py` + `core/model.py` + 新增 `core/memory_bank.py`）：

```
现状: BlockDeltaAttnRes.forward(h_current, block_outputs)
      block_outputs 只有 2 个 source（model.py 每块后重置为 [h0, x]）
改进: block_outputs 增加记忆候选向量 → 注意力在 3+ source 上自由选择
```

**新组件 `MemoryBank(nn.Module)`**（`core/memory_bank.py`）：

```python
class MemoryBank(nn.Module):
    """
    可微长期记忆槽（方向 A：随全局优化器演化）。

    - 槽位: memory_matrix [capacity, d_model]（nn.Parameter，可训练）
    - 写入: 状态经写入门（可微）更新槽位
    - 读取: 与当前状态余弦相似取 top-k 候选（索引），
            最终检索决策由 AttnRes 注意力完成（决策）
    """
    def __init__(self, d_model: int, capacity: int = 128):
        super().__init__()
        self.capacity = capacity
        # 可训练记忆槽（方向 A：梯度回传随优化器演化）
        self.memory_matrix = nn.Parameter(
            torch.randn(capacity, d_model) * 0.02)
        # 写入门（可微）
        self.write_gate = nn.Linear(d_model, 1, bias=False)
        # 读投影（记忆参与注意力的 K/V 空间）
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self._pos = 0  # 环形写入指针

    def retrieve(self, query, top_k=8):
        """索引：返回 top-k 相似记忆向量（不含决策——决策在注意力）"""
        q = query.mean(dim=(-2, -1)).detach()      # [1, d]
        sim = torch.cosine_similarity(
            q, self.memory_matrix, dim=-1)         # [capacity]
        idx = sim.topk(min(top_k, self.capacity)).indices
        return self.memory_matrix[idx]             # [k, d]

    def write(self, vector, lr=0.1):
        """写入：可微更新槽位（方向 A）"""
        gate = torch.sigmoid(self.write_gate(vector)).squeeze()
        with torch.no_grad():
            self.memory_matrix.data[self._pos] *= (1 - lr * gate)
            self.memory_matrix.data[self._pos] += lr * gate * vector
            self._pos = (self._pos + 1) % self.capacity
```

**AttnRes 整合**（`core/attn_res.py`）：

```python
class BlockDeltaAttnRes(nn.Module):
    # __init__ 增加: self.memory_bank = None（由 model 注入）

    def forward(self, h_current, block_outputs, memory_bank=None):
        # ... 现有 deltas 构造 ...
        # 方向 B: 记忆作为额外 source
        if memory_bank is not None and len(memory_bank.memory_matrix) > 0:
            mem = memory_bank.retrieve(h_current, top_k=config.memory_context_top_k)
            for m in mem:
                deltas.append(m)          # K 归一化后参与注意力
                vals.append(memory_bank.v_proj(m))   # V 投影
        # 注意力在 [deltas..., 记忆...] 上自由分配 ← 模型自检索
```

**关键点**：
- **索引与决策分离**：`MemoryBank.retrieve` 只做 top-k 候选筛选（索引，O(capacity) 轻量）；**最终"回忆什么"由模型注意力权重决定**（决策）——模型有权忽略所有记忆候选（给低权重），有权聚焦某一条（高权重）
- 记忆的 K 与 delta 同空间归一化（现有 L2 归一化逻辑复用）
- 梯度路径：注意力输出 → V 投影 → 记忆槽参数 → 随全局优化器/在线训练演化（方向 A）

---

### 层 3：可微记忆演化（方向 A）——记忆随模型训练演化

**设计**：MemoryBank 的槽位是 `nn.Parameter`，写入/读取全程可微——记忆不是静态仓库，而是**与模型一起演化的组件**。

**与现有代码的整合**：

| 现有机制 | 整合点 |
|---------|--------|
| 全局优化器（internal_loop L121-134 白名单） | 白名单增加 `memory_bank`——随 consolidation 全局回放演化 |
| 在线训练（pipeline `_train_on_full`） | 注意力路径回传 → 记忆槽梯度 ✓（无需额外代码） |
| 内循环 Stage D（状态 push） | 增加 `memory_bank.write(u_next_input)`——意识流状态自动沉淀为长期记忆 |
| 用户对话（pipeline 注入） | 对话 embedding 也写入 memory_bank（对话→长期记忆） |

```python
# loop/internal_loop.py Stage D 增强：
def _write_memory(self):
    """方向 A: 当前状态沉淀为长期记忆（可微）"""
    mb = getattr(self.model, 'memory_bank', None)
    if mb is None:
        return
    state = self._u_next_input.squeeze(0)  # [d]
    mb.write(state.detach(), lr=self.config.memory_write_lr)
    # 对话记忆也写（pipeline 注入的 h_state 变化）
```

---

## 三、整合后的完整数据流

```
用户对话 ──► pipeline
              ├─ token 拼接（现有，短期上下文）       [保留]
              ├─ 注入 h_t（方向C，对话→意识流）        [新增]
              └─ push endosphere（现有）              [保留]

内循环每步:
  Stage A: v_t = get_state()            ← h_state 已含对话记忆（方向C）
           forward_internal(v_t)
              └─ 每层 AttnRes 边界:
                   block_outputs + MemoryBank 候选    [新增，方向B]
                   模型注意力自由分配 ← 检索决策
  Stage B:  Hebbian 更新（含记忆 source 的梯度）      [现有+增强]
  Stage D:  push 辩证缓冲（现有）
            memory_bank.write(状态)                   [新增，方向A]
  Stage E/F: consolidation（全局回放）→ memory_bank 随优化器演化 [新增]

推理时: 模型"想起"= 注意力给记忆 source 高权重
       模型"遗忘"= 注意力给记忆低权重（自然衰减，无需删除）
```

## 四、与现有组件的整合清单

| 组件 | 改动 | 文件 |
|------|------|------|
| MemoryBank（新增） | 记忆槽 + 写入门 + 读投影 + retrieve/write | `core/memory_bank.py` |
| ReflexModel | 注入 `memory_bank`；AttnResStack 引用 | `core/model.py` |
| BlockDeltaAttnRes | forward 增加记忆候选 source | `core/attn_res.py` |
| ReflexPipeline | 对话注入 h_t + memory_bank 写入 | `interaction/pipeline.py` |
| InternalLoop | Stage D 记忆写入；全局优化器白名单 | `loop/internal_loop.py` |
| ReflexConfig | 新增记忆参数（见五） | `config/model_config.py` |
| 在线训练 | 注意力路径自动回传记忆槽梯度 | 无需改动（方向A天然可微） |

## 五、参数设计

| 参数 | 默认值 | 说明 |
|------|--------|------|
| memory_bank_capacity | 128 | 记忆槽数量 |
| memory_context_top_k | 8 | 每边界参与注意力的记忆候选数 |
| memory_write_lr | 0.05 | 写入门更新强度 |
| dialog_memory_alpha | 0.3 | 对话→h_t 混合率 |
| memory_attn_init_scale | 0.1 | AttnRes 记忆 source 初始权重（从低调，防止记忆主导） |

## 六、实现顺序

1. **层 1（短期记忆）**：pipeline 对话注入 h_t（约 15 行，零风险）
2. **MemoryBank 骨架**：新组件 + model 注入 + Stage D 写入（独立可测）
3. **层 2（AttnRes 整合）**：记忆候选 source（核心，改动 attn_res.py）
4. **层 3（可微演化）**：全局优化器白名单 + 验证梯度回传
5. 逐层验证后叠加

## 七、验证指标

| 指标 | 测量 | 期望 |
|------|------|------|
| 短期记忆 | 第 2 轮引用第 1 轮信息（"我叫什么名字"） | 能承接 |
| 长期记忆 | 间隔 100+ 步后检索相似状态 | 记忆 source 仍被选中 |
| 自检索行为 | AttnRes 对记忆 source 的注意力权重分布 | 困惑时权重升高（涌现，非代码判断） |
| 可微演化 | memory_bank 参数随训练变化量 | > 0（非冻结） |
| 稳定性 | 记忆注入后 loss_int 有界 | 不劣化 |

## 八、兼容性

- 记忆关闭时（`memory_bank=None`）行为与现有完全一致（AttnRes 退化为 2 source）
- MemoryBank 参数在 checkpoint 中随模型保存/恢复（state_dict 自动包含）
- 不删除任何现有机制（token 拼接/辩证缓冲/回放保留）

## 九、哲学总结

```
v2: 记忆在模型外，代码操控——模型被喂
v3: 记忆在模型内，注意力自检索——模型自主回忆

短期: 对话编码进 h_t——记忆即状态（零代码）
长期: 记忆作 AttnRes source——回忆即注意力（模型决策）
演化: 记忆槽随优化器可微更新——记忆随学习成长

模型"想不想回忆"由自身注意力决定——这才叫自由交互。
```
