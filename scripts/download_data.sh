#!/bin/bash
# ============================================================
# download_data.sh - 从 HF-mirror 下载全部训练数据
# 在 AutoDL 上运行: bash scripts/download_data.sh
# ============================================================
set -e
export HF_ENDPOINT=https://hf-mirror.com

DATA_DIR=/root/autodl-tmp/data
mkdir -p $DATA_DIR

echo "============================================"
echo "1/4 下载 Pretrain 数据"
echo "============================================"

# 1. Wikipedia 全量 (6 快照, 22.9 GB)
echo "  [1/4] Wikipedia 全量..."
hf download fjcanyue/wikipedia-zh-cn \
    --repo-type dataset \
    --local-dir $DATA_DIR/wikipedia

# 2. SkyPile 最后批次 head (18 文件, 26.4 GB)
echo "  [2/4] SkyPile 最后批次..."
hf download Skywork/SkyPile-150B \
    --repo-type dataset \
    --include "data/2023-14_zh_head_*.jsonl" \
    --include "data/2023-06_zh_head_*.jsonl" \
    --local-dir $DATA_DIR/skypile

# 3. ChineseWebText2.0-HQ 前 30 分片 (5.4 GB)
echo "  [3/4] ChineseWebText2.0-HQ..."
hf download Morton-Li/ChineseWebText2.0-HighQuality \
    --repo-type dataset \
    --include "data/CASIA-LM_ChineseWebText2.0_partial-0000[0-2]*.parquet" \
    --local-dir $DATA_DIR/cwt-hq

# 4. FineWeb-2-zh 前 5 文件 (7.1 GB)
echo "  [4/4] FineWeb-2-zh..."
hf download TiWu-Lab/fineweb-2-zh \
    --repo-type dataset \
    --include "000_0000[0-4].parquet" \
    --local-dir $DATA_DIR/fineweb-zh

echo "============================================"
echo "2/4 下载 SFT 数据"
echo "============================================"

hf download YeungNLP/firefly-train-1.1M \
    --repo-type dataset --local-dir $DATA_DIR/firefly

hf download BelleGroup/train_1M_CN \
    --repo-type dataset --local-dir $DATA_DIR/belle

hf download shareAI/ShareGPT-Chinese-English-90k \
    --repo-type dataset --local-dir $DATA_DIR/sharegpt

hf download BAAI/COIG \
    --repo-type dataset --local-dir $DATA_DIR/coig

echo "============================================"
echo "3/4 下载 Teacher 模型"
echo "============================================"

hf download Qwen/Qwen2.5-1.5B-Instruct \
    --local-dir $DATA_DIR/qwen2.5-1.5b-instruct

echo "============================================"
echo "4/4 下载 Embedding 初始化源"
echo "============================================"

hf download Qwen/Qwen2.5-0.5B \
    --local-dir $DATA_DIR/qwen2.5-0.5b

echo "============================================"
echo "下载完成!"
echo "============================================"
du -sh $DATA_DIR/*
echo "---"
du -sh $DATA_DIR
