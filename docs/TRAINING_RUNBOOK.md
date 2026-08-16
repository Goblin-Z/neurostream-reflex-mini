# NeuroStream-Reflex V2-Mini 训练全流程傻瓜式操作手册（v2）

> **版本**：2026-08 v2（核心训练目的 = **组织语言能力**：逻辑提问 + 问答 + 多轮对话）
> **适用**：单卡 24~32GB GPU（RTX 3090/4090/A5000/H800），AutoDL 或自建服务器
> **总耗时**：约 2~3 天（数据生成 2-3h + SFT ~6h + KD ~1 天 + 精修 ~0.5 天）

---

## 0. 训练目标与策略（先读这个）

### 核心目的（本手册的一切围绕它）

```
阶段一（训练）：让模型拥有"组织语言"的能力——
    ✓ 提出逻辑顺畅的问题（信息不足时主动澄清，而不是瞎猜）
    ✓ 流畅的问答（知识性问题能组织成有条理的回答）
    ✓ 多轮对话（承接上下文、追问、引用前文）
阶段二（部署验证）：在语言能力成熟的基础上，开启内循环（意识流）——
    ✓ 验证模型是否会主动提问（sigma 驱动的主动求证）
    ✓ 验证模型是否会自我进化（对话反馈 → 学习 → 行为变化）
```

**关键决策（2026-08 用户确认）**：
- ❌ **弃用 `sft_all.jsonl`**（实测对能力提升无效，不再使用）
- ✅ 数据只用：`sft_kd_clean.jsonl`（教师生成 QA/多轮 10 万条）+ `clarify_30k.jsonl`（澄清式提问对话，新增生成）
- ✅ SFT 步数按数据量校准（~3-4 epoch），**不再用 12000 步跑 10 万条数据（那是 13 epoch 过拟合的根源）**

### 需要重做预训练吗？不需要。

- 预训练产物（`pretrain_final.pt`）有效：embedding 已充分训练（±0.9 范围），15B tokens ≈ 19.5× 参数接近 Chinchilla 最优；
- 上一轮诊断的全部问题在 SFT 及之后（数据量/epoch 错配、无 warmup、sigma 未校准），本次已修复；
- 从 SFT 接着做，直接复用预训练权重。

### 起点选择

| 路线 | 起点 | 适用 |
|------|------|------|
| **A（推荐）** | `pretrain_final.pt` | 有干净预训练产物（228882 步 ≈ 15B tokens，完整） |
| B | `sft_kd_150k_final.pt` | 只有旧 SFT 产物（含预训练能力，可当起点） |
| **立即部署验证** | `distill_refined_final.pt` | **不用等新训练**：这是已完整跑过 SFT→KD→精修的最终模型（语言能力为现有 4 个 checkpoint 中最好），可直接用于 §6 的主动提问/自我进化验证，与新训练并行 |

> 注意：`distill_refined_final.pt` 的 SelfModel/Critic 未跑 warmup（实测仍为随机初始化），部署时内循环会从零在线学习——这正是"部署即学习"的验证场景，不影响语言能力部分。

---

## 1. 前置条件与资产清单

```bash
# 目录
mkdir -p /root/autodl-tmp/data /root/autodl-tmp/checkpoints/reflex-mini
export HF_ENDPOINT=https://hf-mirror.com
pip install -q torch transformers safetensors tqdm pandas requests
nvidia-smi
df -h /root/autodl-tmp   # 需要 ≥ 100GB
```

**需上传到服务器的文件**（AutoDL 实例释放后重新上传）：

| 文件 | 本地位置 | 必需 | 用途 |
|------|---------|------|------|
| 项目代码（本仓库，含本次修复） | GitHub clone | ✅ | 全部训练/部署代码 |
| `pretrain_final.pt`（7.94GB，228882 步） | `D:\codingfile\pretrain_final.pt` | ✅（路线 A） | 新 SFT 起点 |
| `sft_kd_clean.jsonl`（0.1GB，99,982 条） | `D:\codingfile\sft_kd_clean.jsonl` | ✅ | SFT 主数据 |
| `distill_refined_final.pt`（7.94GB，KD+精修完整） | `D:\codingfile\distill_refined_final.pt` | ✅（部署验证） | 立即部署验证 §6，与新训练并行 |
| `sft_kd_150k_final.pt`（7.94GB，SFT 3000 步） | `D:\codingfile\sft_kd_150k_final.pt` | 可选 | 路线 B 备用 |
| `pretrain_step220000.pt`（7.94GB，中间产物） | `D:\codingfile\pretrain_step220000.pt` | 可选 | 备用（loss 9.69 甚至低于 final） |
| `qwen2.5-0.5b/`、`qwen2.5-1.5b-instruct/` | 重新下载（~4GB） | ✅ | tokenizer + teacher |

> 注：checkpoint/数据文件因 GitHub 单文件 100MB 限制**不入库**，需手动上传（AutoDL 网盘/SCP/oss 传输）。5 个文件共 ~32GB。

```bash
# 上传后确认
ls -lh /root/autodl-tmp/data/ | head -20
ls -lh /root/autodl-tmp/checkpoints/reflex-mini/
```

---

## 2. 一键跑通（最傻瓜方式）

```bash
cd /root/autodl-tmp/neurostream_reflex_v2_mini

nohup bash scripts/run_pipeline.sh > /root/autodl-tmp/pipeline.log 2>&1 &
tail -f /root/autodl-tmp/pipeline.log

# 中断后重跑：自动跳过已完成阶段（幂等）
bash scripts/run_pipeline.sh
```

`run_pipeline.sh` 各阶段（v2）：

| 阶段 | 内容 | 产物 | 时间 |
|------|------|------|------|
| 0a-0d | 教师生成 QA/多轮 + 清洗（`sft_kd_clean.jsonl` 已上传则跳过） | `sft_kd_clean.jsonl` | 0~3h |
| 0e | **生成澄清式提问数据 3 万条**（核心：训练逻辑提问） | `clarify_30k.jsonl` | ~2h |
| 0f | 合并 `sft_kd_clean + clarify`（去重） | `sft_mix.jsonl` | 10min |
| 1 | **硬标签 SFT**：2000 步 ≈ 3-4 epoch，lr 2e-5 | `sft_final.pt` | ~6h |
| 1.5 | **Consciousness Warmup**：1000 步（SelfModel + Critic + sigma 校准） | `warmup_final.pt` | ~15min |
| 2 | **软标签 KD**：4000 步，T=4（首次自动生成 teacher logits ~2-3h） | `distill_final.pt` + `distill_highT.pt` | ~1 天 |
| 3 | **低 T 精修**：6000 步，T=1，fresh optimizer | `distill_refined.pt` | ~0.5 天 |
| 4 | **记忆微调**（可选，需 `mt_memory_20k.jsonl`） | `memory_tuned.pt` | ~30min |

---

## 3. 数据方案详解（组织语言能力的三个支柱）

### 3.0 为什么当前 10 万条数据不够（2026-08 诊断）

| 缺陷 | 具体表现 | 后果 |
|------|---------|------|
| 问题多样性低 | 模板 QA 只有 ~9200 个唯一问题（40 模板 × 230 实体） | 模型只会回答见过的模式 |
| 追问模式单一 | 多轮追问池仅 15 条通用 + 10 条引用（**已扩充到 45+20**） | 对话推进方式雷同 |
| 教师质量上限 | Qwen2.5-1.5B 生成，回答偏模板化/机翻风 | 语言上限被锁死 |
| 形态单一 | 全部是"用户问→模型答"，缺少真实对话的指代/省略/情绪 | 对话感差 |
| **终止符断链（已修复）** | SFT 监督了 `<\|im_end\|>` 但 generate 的 stop_ids 不含它 | **模型生成 `<\|im_end\|>` 后继续续写——"不会适时停止"的直接根因**（`core/model.py _get_stop_ids` 已修复） |

### 3.1 数据配方（对话优先，总量 20~30 万条）

| 支柱 | 数据 | 数量 | 教模型什么 |
|------|------|------|-----------|
| **逻辑提问** | `clarify_30k.jsonl`（`scripts/generate_ask_data.py`） | 3~5 万 | 信息不足 → 主动问 1-3 个关键问题（具体、问号结尾、一次问全）→ 补充后完整回答 |
| **多轮对话** | `sft_kd_clean.jsonl` 的 multi-turn + 重新生成扩量 | 8~12 万 | 承接上下文、多样化追问（45 条追问池）、引用前文 |
| **问答** | `sft_kd_clean.jsonl` 的 QA 部分 + self-instruct 扩量 | 8~12 万 | 知识性问题 → 有条理的回答（分点/结构） |
| 真实对话（可选） | ShareGPT 中文子集（`scripts/download_data.py` 已支持） | ~5 万 | 真实人类对话分布（指代/省略/口语） |

**优先级**：多轮对话 ≥ 逻辑提问 > 单轮 QA。对话能力靠"多轮 + 提问"数据，单轮 QA 只是辅助。

**数据生成命令（服务器上跑，需 teacher 模型）**：

```bash
# 澄清式提问数据（新脚本）
python scripts/generate_ask_data.py \
    --teacher /root/autodl-tmp/data/qwen2.5-1.5b-instruct \
    --output /root/autodl-tmp/data/clarify_30k.jsonl \
    --max-samples 30000 --batch-size 4 --device cuda

# 若 sft_kd_clean.jsonl 未上传，重新生成（0a-0d）
python scripts/generate_qa.py --teacher ... --output $DATA/qa_tpl_20k.jsonl --max-samples 20000 --mode qa --batch-size 4 --device cuda
python scripts/generate_qa.py --teacher ... --output $DATA/qa_si_40k.jsonl --max-samples 40000 --mode self-instruct --batch-size 4 --device cuda
python scripts/generate_qa.py --teacher ... --output $DATA/mt_40k.jsonl --max-samples 40000 --mode multi-turn --batch-size 4 --device cuda
python scripts/filter_qa.py --inputs $DATA/qa_tpl_20k.jsonl $DATA/qa_si_40k.jsonl \
    --multiturn-files $DATA/mt_40k.jsonl --output $DATA/sft_kd_clean.jsonl

# 合并
python -c "
import json
seen = set(); n = 0
with open('/root/autodl-tmp/data/sft_mix.jsonl', 'w', encoding='utf-8') as out:
    for path in ['/root/autodl-tmp/data/sft_kd_clean.jsonl',
                 '/root/autodl-tmp/data/clarify_30k.jsonl']:
        try:
            with open(path, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try: item = json.loads(line)
                    except Exception: continue
                    key = (item.get('instruction',''), item.get('output',''))
                    if key in seen: continue
                    seen.add(key); out.write(line + '\n'); n += 1
        except FileNotFoundError:
            print(f'[warn] 缺少 {path}')
print(f'sft_mix.jsonl: {n} 条')
"
```

**数据量不够时的建议**（10~13 万条是下限，质量>数量）：
- 优先保证 clarify 数据 ≥ 2 万（直接决定"主动提问"能力）；
- 可用更强 teacher（Qwen2.5-7B/14B）重新生成（质量上限更高）；
- 不要为提高数量引入 sft_all.jsonl（实测无效）。

---

## 4. 训练步骤（与脚本等价的手跑命令）

### 4.1 SFT（约 6 小时）

```bash
python -m train.train_reflex --mode sft \
    --sft-data /root/autodl-tmp/data/sft_mix.jsonl \
    --sft-steps 2000 --batch-size 8 --grad-accum 8 \
    --lr 2e-5 --dtype bfloat16 \
    --resume /root/autodl-tmp/checkpoints/reflex-mini/pretrain_final.pt \
    --output-dir /root/autodl-tmp/checkpoints/reflex-mini
# 预期：loss 从 ~5 降到 < 2.5；显存 ~10-12GB
```

### 4.2 Consciousness Warmup（约 15 分钟，必做）

```bash
python -c "
import torch, sys
sys.path.insert(0, '.')
from config.model_config import ReflexMiniConfig
from core.model import ReflexModel
from train.train_reflex import ReflexTrainer, TrainConfig
from transformers import AutoTokenizer
config = ReflexMiniConfig()
model = ReflexModel(config)
tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/data/qwen2.5-0.5b', trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model._decode_tokenizer = tokenizer
cfg = TrainConfig(mode='full', device='cuda', dtype='bfloat16',
                  sft_data='/root/autodl-tmp/data/sft_mix.jsonl',
                  output_dir='/root/autodl-tmp/checkpoints/reflex-mini',
                  resume='/root/autodl-tmp/checkpoints/reflex-mini/sft_final.pt')
model.to('cuda')
trainer = ReflexTrainer(model, tokenizer, cfg)
trainer.load_checkpoint('/root/autodl-tmp/checkpoints/reflex-mini/sft_final.pt')
trainer.train_consciousness_warmup(steps=1000)
"
```

### 4.3 KD（约 1 天，首次含 teacher logits 生成 2-3h）

```bash
python -m train.train_reflex --mode distill \
    --distill-data /root/autodl-tmp/data/sft_mix.jsonl \
    --teacher /root/autodl-tmp/data/qwen2.5-1.5b-instruct \
    --distill-steps 4000 --batch-size 8 --grad-accum 4 \
    --lr 1e-5 --distill-temperature 4.0 --distill-kd-weight 0.8 \
    --dtype bfloat16 \
    --resume /root/autodl-tmp/checkpoints/reflex-mini/warmup_final.pt \
    --output-dir /root/autodl-tmp/checkpoints/reflex-mini
cp $CKPT/distill_final.pt $CKPT/distill_highT.pt
```

### 4.4 低 T 精修（约 0.5 天）

```bash
python -m train.train_reflex --mode distill \
    --distill-data /root/autodl-tmp/data/sft_mix.jsonl \
    --teacher /root/autodl-tmp/data/qwen2.5-1.5b-instruct \
    --distill-steps 6000 --batch-size 8 --grad-accum 4 \
    --lr 5e-6 --distill-temperature 1.0 --distill-kd-weight 0.9 \
    --fresh-optimizer --dtype bfloat16 \
    --resume /root/autodl-tmp/checkpoints/reflex-mini/distill_final.pt \
    --output-dir /root/autodl-tmp/checkpoints/reflex-mini
mv $CKPT/distill_final.pt $CKPT/distill_refined.pt
```

### 4.5 记忆微调（可选）

```bash
# 先生成数据：generate_qa.py --mode multi-turn --memory-tune（7轮+长程引用）
python train_memory.py \
    --checkpoint /root/autodl-tmp/checkpoints/reflex-mini/distill_refined.pt \
    --data /root/autodl-tmp/data/mt_memory_20k.jsonl \
    --steps 3000 --lr 1e-5 --batch-size 4 \
    --tokenizer /root/autodl-tmp/data/qwen2.5-0.5b \
    --output-dir /root/autodl-tmp/checkpoints/reflex-mini
```

---

## 5. 阶段一验收：组织语言能力测试（过不了不部署）

```bash
python chat_sft.py --checkpoint .../sft_final.pt --device cuda \
    --prompt "中国的首都是" --prompt "1+1等于几？" \
    --prompt "请介绍一下量子力学。"
```

| 测试 | 合格标准 |
|------|---------|
| 知识问答 | "中国的首都是" → 直接答"北京"，无循环论证 |
| 算术 | "1+1等于几？" → "2"，无胡言 |
| 知识组织 | "介绍一下量子力学" → 分点、有条理、术语不空转 |
| **主动提问** | 输入"帮我写个方案" → **反问**"什么方案？给谁看？目标？"（而不是直接编） |
| **多轮** | "我叫小明" → 隔几轮问"我叫什么" → 能答"小明" |

> 主动提问测试脚本（部署前先用 chat_sft 裸测一轮）：
> ```python
> # prompt: "帮我写一份方案"（无任何细节）
> # 合格：输出以问句结尾/含澄清问题
> # 不合格：直接输出一份编造方案
> ```

---

## 6. 阶段二：部署 + 主动提问/自我进化验证（核心实验）

### 6.1 部署

```bash
cd /root/autodl-tmp/neurostream_reflex_v2_mini
# 立即验证（现有模型，无需等训练）：distill_refined_final.pt
python run_mini.py --checkpoint /root/autodl-tmp/checkpoints/reflex-mini/distill_refined_final.pt \
    --tokenizer /root/autodl-tmp/data/qwen2.5-0.5b
# 新训练完成后：改用 distill_refined.pt（新流程产物）
# stats 查看内循环；clear 清记忆；quit 退出
```

> **两条并行路线**：
> 1. **立即**：上传 `distill_refined_final.pt` → 直接开始 §6.2/6.3 实验（验证现有模型是否会主动提问/自我进化）——不等新训练；
> 2. **训练中**：同时跑 `run_pipeline.sh` 新流程（组织语言优先），产出新 `distill_refined.pt` 后复测同一组实验，对比新旧模型的行为差异。

### 6.2 实验一：主动提问验证（模型会不会问）

```
1. 启动（内循环全开），输入模糊请求："帮我推荐一个手机"
   → 期待模型反问（预算？用途？品牌偏好？）——来自 SFT 学到的澄清行为
2. 输入"预算三千，平时拍照和打游戏"
   → 期待给出具体推荐
3. stats 查看 sigma：提问时 sigma 是否偏高、回答后是否回落（涌现冷却）
4. 对照组：--no-internal-loop 跑同样对话，对比提问行为差异
   （验证：主动求证是模型行为，内循环影响提问时机/内容）
```

| 观测 | 健康信号 |
|------|---------|
| 提问内容 | 逻辑顺畅、覆盖关键缺失信息（而非"能详细点吗"式空泛） |
| sigma 与提问 | 提问发生在 sigma 高时；反馈后 sigma 下降并停止追问 |
| 有无内循环对照 | 有差异（内循环可归因贡献） |

### 6.3 实验二：自我进化验证（模型会不会变）

```
1. 连续 20 轮与模型讨论一个领域（如"帮我写代码/讲历史"），每轮用 stats 记录：
   internal steps / loss_int / sigma / h_to_bias(max)
2. 观察：
   - h_to_bias(max) 是否从 0 缓慢增长（状态门控随使用生效）
   - loss_int 是否收敛到更低区间（自指动力学稳定）
   - 专家权重变化：plastic 专家有效更新（stats 无 ERROR）
3. 会话结束时用同一问题复测（如"我前面说过要什么颜色？"）：
   - 记忆验证：L4 KV 内容记忆 → 能复述
4. 重启后（记忆清空）同问题 → 对比"会话内记忆 vs 权重固化"的差异
```

### 6.4 内循环健康仪表（stats 解读）

| 指标 | 健康值 | 含义 |
|------|--------|------|
| internal steps | 持续增长 | 意识流在运行 |
| loss_int | 有界（<5） | 自指动力学不爆炸 |
| sigma | 0.3~0.7 振荡 | 不塌陷到常数（混沌边缘） |
| h_to_bias(max) | 从 0 缓慢增长 | 状态门控随对话生效 |
| can_ask | warmup 500 步内受限，之后按 sigma | 保守冷却 → 涌现冷却 |
| [ERRORS] | 无 | 无内部异常 |

---

## 7. 常见问题

**Q1：显存不够（24GB OOM）？** SFT 用 `--batch-size 4 --grad-accum 16`。

**Q2：KD 后变差？** 0.77B 容量下 KD 可能不如硬标签 SFT。回退：
`cp $CKPT/distill_highT.pt $CKPT/deploy_ckpt.pt`（或直接用 sft_final.pt）。

**Q3：模型不提问？** 检查：① clarify 数据是否生成并进了 sft_mix（`wc -l` 确认 >2 万条）；② SFT 是否过拟合（loss 是否 <2.5）；③ 部署时 `--verify-threshold 0.4` 调低 sigma 门槛。

**Q4：提问太频繁/太随机？** warmup 500 步内保守冷却自动生效；之后若仍乱问，调高 `verify_threshold`（0.5→0.6）。

**Q5：断点续跑？** 直接重跑 `run_pipeline.sh`（按产物标记跳过）；训练中断用 `--resume 最新step.pt`（脚本自动选）。

**Q6：数据只有 sft_kd_clean 没有 clarify？** 能跑（脚本自动跳过缺失文件），但"主动提问"能力会弱——建议补生成。

---

## 8. 时间线速查

```
Day 1:   上传资产 + 数据（clarify 生成 2h）→ SFT 2000 步 (~6h)
Day 2:   Warmup (15min) → KD 4000 步 (~1 天) → 验收 §5
Day 3:   低T精修 (~0.5 天) → 部署 → 实验一/二（§6.2/6.3）
Day 3+:  持续对话使用（部署即学习），定期 stats 观察进化
```

---

*配套文档：`docs/PHILOSOPHY_INTEGRATED_ANALYSIS.md`（设计哲学与修复依据）、`docs/TRAINING_QUALITY_AUDIT*.md`（已知问题基线）。*
