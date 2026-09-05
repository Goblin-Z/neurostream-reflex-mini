# NeuroStream-Reflex × DeepSeek-V2-Lite-Chat（嫁接实现）

NeuroStream-Reflex 双循环框架（内循环意识 / 主动求证 / 边缘混沌 / 部署即学习）嫁接在 DeepSeek-V2-Lite-Chat 之上的代码包。

## 模型与 Tokenizer（下载）

权重与分词器在魔搭社区（本仓库**不含**模型文件）：

- **下载地址**：https://modelscope.cn/models/Marshauv/Neurostream_reflex_deepseek_chat
- 文件：`reflex_dsv2lite_graft.pt`（46GB bf16，23.03B 参数 = 15.7B DeepSeek 主干 1:1 直拷 + ~7.3B Reflex 附加件）、`tokenizer.json`、`tokenizer_config.json`

```bash
# 下载模型（魔搭）
modelscope download --model Marshauv/Neurostream_reflex_deepseek_chat \
    --local_dir /root/autodl-tmp/models/Neurostream_reflex_deepseek_chat
```

## 快速开始

```bash
pip install torch transformers safetensors tqdm modelscope
bash run_chat.sh
# 常用附加参数:
#   --verify-threshold 0.3   主动求证触发阈值（σ 校准后建议 0.3）
#   --online-ce              打开每轮 CE 在线训练
#   --hebbian-lr 0           关闭 Hebbian（对照/回炉）
```

手动运行：

```bash
python run_mini.py --checkpoint reflex_dsv2lite_graft.pt \
    --tokenizer <模型目录> --device cuda --dtype bfloat16 \
    --no-online-ce --hide-think --gen-debug --sigma-cal
```

## 架构说明

- 主干：DeepSeek-V2-Lite-Chat（27 层 MLA：1 dense + 26 MoE；64 路由专家 top-6 + 合并共享专家 1:1 直拷；YaRN rope）——权重零训练复用
- Reflex 附加件（~7.3B）：每专家 query_proj + uncertainty_head（sigma）、SelfModel（GRU+VAE）、AttnRes、MemoryBank、Critic
- 学习率光谱（仅影响 Hebbian）：路由 16×1e-5 / 32×1e-4 / 16×8e-2，共享 1e-4，dense 1e-5（run_mini 强制覆盖 checkpoint 旧值，`--hebbian-lr 0` 一键关闭）
- 数值一致性：fp32 下与 transformers built-in 逐位一致（max|Δ|=6e-8、top-1 100%）；bf16 27 层 ≈78%（MoE 路由浮点内核噪声）
- 主动求证（P3）：σ>阈值 → 模型从对话尾部续写涌现（`[MODEL SAYS]`）/ 无对话自由联想（`[MODEL FREE]` 观测模式）

## 目录

| 路径 | 说明 |
|---|---|
| `run_chat.sh` | 一键运行入口（下载/生成 checkpoint → 双循环对话） |
| `run_mini.py` | 完整部署入口 |
| `config/ core/ loop/ interaction/ learn/ improve/` | 反射框架运行模块（`improve/` 仅架构自修改；自博弈训练未发布） |
| `scripts/` | 嫁接 checkpoint 生成 / 数值验证 / 冒烟测试 / Chat 切换 |

## 许可

项目代码：MIT；主干权重：DeepSeek-V2-Lite-Chat（MIT，deepseek-ai/DeepSeek-V2-Lite-Chat）——衍生权重保留 MIT，使用需遵守上游条款。
