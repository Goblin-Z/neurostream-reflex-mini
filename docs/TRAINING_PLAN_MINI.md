# NeuroStream-Reflex V2-Mini 完整训练方案 (最终版)

> **模型**: ReflexMiniConfig (1.03B, 24L, d_model=768, 6 experts)
> **目标**: 优秀日常对话 + 充足百科知识
> **硬件**: 单卡 24GB GPU (RTX 3090/4090/A5000)

---

## 0. 数据清单与磁盘占用

### Pretrain 数据

| # | 数据源 | HF-mirror ID | 下载范围 | 文件数 | 磁盘占用 | 估计 tokens |
|---|--------|-------------|---------|--------|---------|------------|
| 1 | 中文维基百科 (全量) | `fjcanyue/wikipedia-zh-cn` | 6 个快照全部下载 | 6 JSON | **22.9 GB** | ~5.1B (含跨快照重复) |
| 2 | SkyPile-150B (最后批次) | `Skywork/SkyPile-150B` | `2023-14_zh_head_*` (8文件) + `2023-06_zh_head_*` (10文件) | 18 JSONL | **26.4 GB** | ~4.8B |
| 3 | ChineseWebText2.0-HQ (子集) | `Morton-Li/ChineseWebText2.0-HighQuality` | 前 30 个 parquet 分片 | 30 parquet | **5.4 GB** | ~2.5B |
| 4 | FineWeb-2-zh (子集) | `TiWu-Lab/fineweb-2-zh` | `000_0000[0-4].parquet` (5文件) | 5 parquet | **7.1 GB** | ~3.0B |
| | **Pretrain 合计** | | | **59 文件** | **61.8 GB** | **~15.4B** |

### SFT 数据

| # | 数据源 | HF-mirror ID | 文件数 | 磁盘占用 | 条数 |
|---|--------|-------------|--------|---------|------|
| 5 | Firefly-train-1.1M | `YeungNLP/firefly-train-1.1M` | 1 JSONL | **10.9 GB** | 115万 |
| 6 | BELLE-train-1M-CN | `BelleGroup/train_1M_CN` | 1 JSON | **3.5 GB** | 100万 |
| 7 | ShareGPT-Chinese-English-90k | `shareAI/ShareGPT-Chinese-English-90k` | 多文件 | **1.8 GB** | 9万 |
| 8 | COIG | `BAAI/COIG` | 多文件 | **4.9 GB** | 6万 |
| | **SFT 合计** | | | **21.1 GB** | ~330万 |

### 模型权重

| # | 数据源 | HF-mirror ID | 磁盘占用 | 用途 |
|---|--------|-------------|---------|------|
| 9 | Qwen2.5-1.5B-Instruct | `Qwen/Qwen2.5-1.5B-Instruct` | **3.1 GB** | 蒸馏 teacher |
| 10 | Qwen2.5-0.5B | `Qwen/Qwen2.5-0.5B` | **1.1 GB** | Embedding 初始化 |

### 总磁盘占用

| 类别 | 占用 |
|------|------|
| Pretrain 数据 | 61.8 GB |
| SFT 数据 | 21.1 GB |
| 模型权重 | 4.2 GB |
| **总下载量** | **87.1 GB** |
| 预处理后合并文件 | +约 60 GB (pretrain_all.jsonl + sft_all.jsonl) |
| Checkpoint 存储 | +约 10 GB (多次保存) |
| **建议磁盘空间** | **≥ 200 GB** |

---

## 1. Wikipedia 文件明细

`fjcanyue/wikipedia-zh-cn` 全量下载 6 个快照：

| 文件名 | 大小 | 快照日期 |
|--------|------|---------|
| wikipedia-zh-cn-20240901.json | ~2.12 GB | 2024-09-01 |
| wikipedia-zh-cn-20241020.json | ~2.13 GB | 2024-10-20 |
| wikipedia-zh-cn-20250320.json | ~2.18 GB | 2025-03-20 |
| wikipedia-zh-cn-20250901.json | ~2.25 GB | 2025-09-01 |
| wikipedia-zh-cn-20260201.json | ~7.1 GB | 2026-02-01 |
| wikipedia-zh-cn-20260501.json | ~7.1 GB | 2026-05-01 |
| **合计** | **22.9 GB** | |

> 后两个快照 (2026年) 文件显著增大，可能包含更多条目或格式扩展。全量下载确保获得最新最全的百科知识。
> 6 个快照间有内容重叠（维基条目跨时间变化不大），训练时相当于对百科知识做 ~6 次强化，有利于知识记忆。

## 2. SkyPile 文件明细

`Skywork/SkyPile-150B` 仅下载最后两个批次的 head（高质量）文件：

| 批次 | 文件范围 | 文件数 | 单文件大小 | 小计 |
|------|---------|--------|-----------|------|
| 2023-14 (最新) | `data/2023-14_zh_head_0000~0007.jsonl` | 8 | ~1.38 GB | 11.0 GB |
| 2023-06 (次新) | `data/2023-06_zh_head_0000~0009.jsonl` | 10 | ~1.54 GB | 15.4 GB |
| **合计** | | **18** | | **26.4 GB** |

> SkyPile 全量 1.31 TB / ~408 文件。仅下载最后 18 个 head 文件（4.4% 的文件量），获取最新最高质量的中文网页文本。

---

## 3. 数据下载命令

```bash
export HF_ENDPOINT=https://hf-mirror.com

# ═══════════════════════════════════════════════════
# Pretrain 数据
# ═══════════════════════════════════════════════════

# 1. Wikipedia 全量 (6 个快照, 22.9 GB)
huggingface-cli download fjcanyue/wikipedia-zh-cn \
    --repo-type dataset \
    --local-dir /data/wikipedia

# 2. SkyPile 最后批次 head (18 文件, 26.4 GB)
huggingface-cli download Skywork/SkyPile-150B \
    --repo-type dataset \
    --include "data/2023-14_zh_head_*.jsonl" \
    --include "data/2023-06_zh_head_*.jsonl" \
    --local-dir /data/skypile

# 3. ChineseWebText2.0-HQ 前 30 分片 (5.4 GB)
huggingface-cli download Morton-Li/ChineseWebText2.0-HighQuality \
    --repo-type dataset \
    --include "data/CASIA-LM_ChineseWebText2.0_partial-0000[0-2]*.parquet" \
    --local-dir /data/cwt-hq

# 4. FineWeb-2-zh 前 5 文件 (7.1 GB)
huggingface-cli download TiWu-Lab/fineweb-2-zh \
    --repo-type dataset \
    --include "000_0000[0-4].parquet" \
    --local-dir /data/fineweb-zh

# ═══════════════════════════════════════════════════
# SFT 数据 (21.1 GB)
# ═══════════════════════════════════════════════════

huggingface-cli download YeungNLP/firefly-train-1.1M \
    --repo-type dataset --local-dir /data/firefly

huggingface-cli download BelleGroup/train_1M_CN \
    --repo-type dataset --local-dir /data/belle

huggingface-cli download shareAI/ShareGPT-Chinese-English-90k \
    --repo-type dataset --local-dir /data/sharegpt

huggingface-cli download BAAI/COIG \
    --repo-type dataset --local-dir /data/coig

# ═══════════════════════════════════════════════════
# 模型权重 (4.2 GB)
# ═══════════════════════════════════════════════════

huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct \
    --local-dir /data/qwen2.5-1.5b-instruct

huggingface-cli download Qwen/Qwen2.5-0.5B \
    --local-dir /data/qwen2.5-0.5b
```

---

## 4. 数据预处理

### 4.1 Pretrain 合并

```python
# scripts/prepare_pretrain.py
import json, glob, pandas as pd

OUTPUT = "/data/pretrain_all.jsonl"

def extract_wikipedia():
    """Wikipedia 全量 6 个快照, 提取 title + body"""
    for path in sorted(glob.glob("/data/wikipedia/wikipedia-zh-cn-*.json")):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                title = item.get('title', '')
                body = item.get('body', item.get('content', ''))
                text = f"{title}\n\n{body}" if title else body
                if len(text) > 100:
                    yield {"text": text}

def extract_skypile():
    """SkyPile 18 个 head 文件"""
    for pattern in ["/data/skypile/data/2023-14_zh_head_*.jsonl",
                    "/data/skypile/data/2023-06_zh_head_*.jsonl"]:
        for filepath in sorted(glob.glob(pattern)):
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    text = item.get('text', item.get('content', ''))
                    if len(text) > 200:
                        yield {"text": text}

def extract_cwt():
    """ChineseWebText2.0-HQ 30 分片"""
    for filepath in sorted(glob.glob("/data/cwt-hq/data/*.parquet")):
        df = pd.read_parquet(filepath)
        for text in df['text'].dropna():
            if len(text) > 200:
                yield {"text": text}

def extract_fineweb():
    """FineWeb-2-zh 5 文件"""
    for filepath in sorted(glob.glob("/data/fineweb-zh/*.parquet")):
        df = pd.read_parquet(filepath)
        for text in df['text'].dropna():
            if len(text) > 200:
                yield {"text": text}

def main():
    count = 0
    with open(OUTPUT, 'w', encoding='utf-8') as out:
        for gen in [extract_wikipedia, extract_skypile, extract_cwt, extract_fineweb]:
            for item in gen():
                out.write(json.dumps(item, ensure_ascii=False) + '\n')
                count += 1
    print(f"Pretrain: {count:,} documents -> {OUTPUT}")

if __name__ == '__main__':
    main()
```

### 4.2 SFT 合并

```python
# scripts/prepare_sft.py  (与之前相同，略)
```

### 4.3 运行预处理

```bash
python scripts/prepare_pretrain.py   # ~30 min
python scripts/prepare_sft.py        # ~10 min
```

---

## 5. Embedding 初始化

```bash
python scripts/init_from_qwen.py \
    --qwen-path /data/qwen2.5-0.5b \
    --output /data/reflex_mini_init.pt
```

从 Qwen2.5-0.5B 提取 token embedding [151936, 512]，零填充到 [151936, 768]。

---

## 6. 训练三阶段

### Phase 1: Pretrain

| 参数 | 值 |
|------|-----|
| 数据 | /data/pretrain_all.jsonl (~15.4B tokens) |
| 微批次 / 累积 | 8 / 8 (有效 64) |
| tokens/step | 65,536 |
| 总步数 | 60,000 (~3.9B tokens seen) |
| 学习率 | 3e-4, warmup 1000, cosine -> 3e-5 |
| 精度 | bfloat16 + gradient checkpointing |
| 显存 | ~12 GB |
| 时间 | ~40-48 小时 |

### Phase 2: SFT

| 参数 | 值 |
|------|-----|
| 数据 | /data/sft_all.jsonl (~330万条) |
| 微批次 / 累积 | 16 / 4 (有效 64) |
| 总步数 | 20,000 (~1.3B tokens) |
| 学习率 | 1e-5, warmup 200, cosine -> 1e-6 |
| 指令遮蔽 | instruction 部分 label=-100 |
| 显存 | ~10 GB |
| 时间 | ~15 小时 |

### Phase 3: Distill

| 参数 | 值 |
|------|-----|
| Teacher | Qwen2.5-1.5B-Instruct |
| 数据 | /data/sft_all.jsonl (复用) |
| 微批次 / 累积 | 8 / 4 |
| 总步数 | 10,000 |
| 学习率 | 2e-5, warmup 200, cosine -> 2e-6 |
| 温度 T | 2.0, CE=0.3, KL=0.7 |
| 显存 | ~14 GB |
| 时间 | ~10 小时 |

---

## 7. 一键训练脚本

```bash
#!/bin/bash
set -e
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0
OUT=/checkpoints/reflex-mini

echo "=== Phase 1: Pretrain ==="
python -m train.train_reflex --mode pretrain \
    --pretrain-data /data/pretrain_all.jsonl \
    --pretrain-steps 60000 --batch-size 8 --grad-accum 8 \
    --lr 3e-4 --dtype bfloat16 \
    --resume /data/reflex_mini_init.pt --output-dir $OUT

echo "=== Phase 2: SFT ==="
python -m train.train_reflex --mode sft \
    --sft-data /data/sft_all.jsonl \
    --sft-steps 20000 --batch-size 16 --grad-accum 4 \
    --lr 1e-5 --dtype bfloat16 \
    --resume $OUT/pretrain_final.pt --output-dir $OUT

echo "=== Phase 3: Distill ==="
python -m train.train_reflex --mode distill \
    --distill-data /data/sft_all.jsonl \
    --teacher /data/qwen2.5-1.5b-instruct \
    --distill-steps 10000 --batch-size 8 --grad-accum 4 \
    --lr 2e-5 --dtype bfloat16 \
    --resume $OUT/sft_final.pt --output-dir $OUT

echo "=== Done: $OUT/distill_final.pt ==="
```

---

## 8. 时间线

```
Day 1:    数据下载 (87 GB, 3-6h) + 预处理 (40 min) + Embedding init
Day 2-3:  Phase 1 Pretrain (60k steps, ~40h)
Day 4:    Phase 2 SFT (20k steps, ~15h)
Day 5:    Phase 3 Distill (10k steps, ~10h) + 评估
Day 6+:   部署 + 持续学习
```

**总时间: ~4-5 天**

---

## 9. 数据配比

### Pretrain 配比 (按 tokens)

| 数据源 | tokens | 占比 | 作用 |
|--------|--------|------|------|
| Wikipedia (6快照) | ~5.1B | 33% | 百科知识，6 次强化记忆 |
| SkyPile 2023 head | ~4.8B | 31% | 最新中文网页，语言流畅度 |
| ChineseWebText-HQ | ~2.5B | 16% | 高质量文本，补充知识 |
| FineWeb-2-zh | ~3.0B | 20% | 多样性，防过拟合 |

> Wikipedia 占比 33%：6 个快照的重复内容等效于对百科知识做 ~6 次强化，确保事实知识牢固记忆。这是"充足百科知识储备"目标的核心保障。

### SFT 配比 (按条数)

| 数据源 | 条数 | 占比 | 作用 |
|--------|------|------|------|
| Firefly | 115万 | 35% | 23 种任务多任务泛化 |
| BELLE | 100万 | 30% | 通用中文指令 |
| ShareGPT | 27万 (3x过采样) | 20% | 真实多轮对话 |
| COIG | 18万 (3x过采样) | 15% | 考试/价值观/代码 |

---

## 10. 部署与持续学习

```bash
# 交互模式 (内循环 + Hebbian 在线学习)
python run_reflex.py --resume /checkpoints/reflex-mini/distill_final.pt

# 纯推理
python run_reflex.py --resume /checkpoints/reflex-mini/distill_final.pt --no-internal-loop
```

训练后模型自动具备：Hebbian 在线学习、辩证缓冲、巩固机制、SelfModel 意识流、求正通路。

---

*方案基于 ReflexMiniConfig (1.03B)，Wikipedia 全量 + SkyPile 最新批次，总计 87 GB 下载，~15B pretrain tokens。*
