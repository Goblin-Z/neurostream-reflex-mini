# NeuroStream-Reflex Mini

English | [简体中文](README.zh-CN.md)

**0.77B 中文神经形态语言模型**——外部训练 + 内部意识流的双循环架构。记忆与思考在每一步计算中交融，形成持续向前的自指循环。

## 核心亮点

- **双循环架构**：外循环（对话）与内循环（意识流）共享状态与记忆，本质是同一个"记忆→计算→新记忆"自指循环在两个尺度上运行
- **记忆系统 v4（L1-L4）**：
  - L0 显式历史（模板拼接）
  - L1 短期语义（对话注入 `h_t`）
  - L2/L3 长期语义（AttnRes 记忆 source + 可微语义槽）
  - L4 内容记忆（分层 KV 缓存——注意力直达历史表示，可逐词复述）
- **自发固化**：记忆的重要性由模型行为量化（注意力聚焦累积 salience），成熟即蒸馏进 stable 权重——非程序性、不依赖固定步数
- **Hebbian 学习**：专家权重由局部梯度强化（动量 + 激活门 + 双 clip），非全局反向传播
- **架构自修改**：专家分裂/裁剪/加层（修复后安全，部署默认开启）

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

### 训练（云端 AutoDL）

```bash
# 一键全流程（数据生成 → 清洗 → SFT → KD → 精修，幂等可断点续跑）
nohup bash scripts/run_pipeline.sh > pipeline.log 2>&1 &
tail -f pipeline.log
```

### 记忆微调（教模型"使用记忆"）

```bash
# 1. 生成长程引用型多轮数据（7 轮 + 跨轮引用追问）
python scripts/generate_qa.py --teacher <teacher> \
  --output mt_memory_20k.jsonl --max-samples 20000 \
  --mode multi-turn --memory-tune --device cuda

# 2. 微调（KV 历史真实参与训练）
python train_memory.py --checkpoint <ckpt.pt> \
  --data mt_memory_20k.jsonl --steps 3000 --lr 1e-5 --batch-size 4
```

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
| `core/` | 模型主干（MoE/Attention/AttnRes/SelfModel/MemoryBank） |
| `loop/` | 内部循环（Stage A-K 意识流） |
| `learn/` | 在线学习（Hebbian/consolidation/Critic） |
| `interaction/` | 外部交互（对话/求证/反馈） |
| `train/` | 训练（pretrain/SFT/distill） |
| `scripts/` | 数据生成/流水线 |

## 模型规格

- **架构**：24 层 GQA + RoPE + SwiGLU MoE（4 stable + 2 plastic）+ 块间 Delta 注意力残差
- **词表**：Qwen2.5（151936）
- **参数量**：769M
- **训练**：15B tokens 中文预训练 + 教师（Qwen2.5-1.5B-Instruct）QA/多轮蒸馏
- **记忆**：MemoryBank（128 语义槽 + 4 轮 KV 缓存 + salience 自发固化）

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
