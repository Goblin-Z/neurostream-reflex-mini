# NeuroStream-Reflex Mini

English | [简体中文](README.zh-CN.md)

**0.77B 中文神经形态语言模型**——外部训练 + 内部意识流的双循环架构。记忆与思考在每一步计算中交融，形成持续向前的自指循环。

## 核心亮点

- **双循环架构**：内循环持续"思考"（状态演化、Hebbian 学习、记忆巩固），外循环把思考转化为对话——两者共享记忆与状态：模型"记得什么"影响它说什么，它说的话又成为新记忆
- **多层记忆**：能逐词记住最近对话（注意力直达历史表示）、理解对话主题（内部状态）、长期保存知识（可学习记忆库）——类似工作记忆 + 情景记忆 + 语义记忆
- **自发巩固**：记忆被模型**实际使用**得越多（由其自身注意力衡量）就越重要——记忆成熟时自动蒸馏进权重，无需任何固定计划
- **Hebbian 学习**：专家通过局部关联强化（"一起放电的神经元连在一起"），非全局反向传播
- **架构自修改**：模型随学习自动分裂/裁剪/新增专家

## 快速开始

### 部署（内循环全开）

```bash
python run_mini.py --checkpoint <ckpt.pt> --device cuda
# 交互：提问 / quit 退出 / stats 查看内循环 / clear 清空记忆
```

### 测试 checkpoint

```bash
python chat_sft.py --checkpoint <ckpt.pt> --device cpu --prompt "中国的首都是"
```

> 训练管线（pretrain/SFT/distill/记忆微调）保留在本地，未随本仓库公开。

## 架构

```
用户输入 → [外循环 Pipeline] ──► 生成（带 h_state + mem_kv）
              │                        │
              ▼                        ▼
         对话注入 h_t + KV 入队    在线训练（带记忆+状态）
              │                        │
              ▼                        ▼
       [内循环 InternalLoop] ◄─────────┘
        Stage A-K（噪声→SelfModel→Hebbian→巩固→求证）
              │ 记忆（语义槽/KV/salience）与计算交融
              ▼
        新状态 + 新记忆 → 供下一次外循环
```

| 组件 | 说明 |
|------|------|
| `core/` | 模型主干（MoE/Attention/SelfModel/MemoryBank） |
| `loop/` | 内部循环（意识流：Hebbian/巩固） |
| `learn/` | 在线学习（Hebbian 更新/Critic） |
| `interaction/` | 外部交互（对话/求证/反馈） |
| `config/` | 模型配置 |

## 模型规格

- **架构**：24 层 GQA + RoPE + SwiGLU MoE（4 稳定 + 2 可塑专家）+ 块间 Delta 注意力残差
- **词表**：Qwen2.5（151936）
- **参数量**：769M
- **训练**：15B tokens 中文预训练 + 教师（Qwen2.5-1.5B-Instruct）QA/多轮蒸馏
- **记忆**：128 个可学习记忆槽 + 4 轮 KV 对话缓存 + 注意力驱动的自发巩固

## 部署命令速查

| 命令 | 功能 |
|------|------|
| `stats` | 内循环状态（steps/loss_int/sigma/can_ask/异常） |
| `clear` | 清空会话记忆（L0 历史 + KV） |
| `quit` | 退出 |
| `--no-internal-loop` | 纯生成对照 |
| `--no-arch-self-mod` | 关闭架构自修改（默认开） |
| `--verify-threshold 0.45` | 调低主动求证门槛 |

## 依赖

```bash
pip install torch transformers safetensors tqdm
export HF_ENDPOINT=https://hf-mirror.com   # 国内镜像
```

## 已知限制

- 0.77B 容量：复杂推理/长文知识有限
- 模型回答能力受训练数据覆盖限制（未学过的指令会答非所问）
- 记忆的有效利用依赖记忆微调（`train_memory.py`）——结构就绪 + 训练强化
- 复杂角色设定指令效果差（建议短指令交互）

## 许可证

MIT License（神经形态研究项目，欢迎 fork 与改进）。

---

*NeuroStream-Reflex Mini — 记忆与思考交融的自指循环系统*
