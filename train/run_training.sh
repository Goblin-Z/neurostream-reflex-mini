#!/bin/bash
# ============================================================
# NeuroStream-Reflex 单卡云端训练启动脚本 (H800 80GB)
# ============================================================
# 用法:
#   bash train/run_training.sh              # 三阶段全量训练
#   bash train/run_training.sh --mode sft   # 仅 SFT
#   bash train/run_training.sh --resume /path/to/ckpt.pt
# ============================================================

set -e

# ── 镜像配置 ──────────────────────────────────────────────────────────
# HuggingFace 镜像（国内用户必须设置）
# 下载 Teacher 模型 (Qwen2.5-7B-Instruct) 和 Tokenizer 时使用
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

# 数据集下载来源（需提前下载到服务器本地路径）：
#   Wudao 2.0 10%:  https://opendatalab.com  → /data/wudao/wudao_10pct.jsonl
#   Firefly 1.6M:   https://hf-mirror.com/YangyiYin/Firefly  → /data/firefly/firefly_1.6m.jsonl
#   BELLE:          https://hf-mirror.com/BelleGroup  → 合并到 SFT 数据
# 蒸馏数据由 Teacher 模型本地生成，无需提前下载。

# ── 环境配置 ──────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

# 数据路径 (按实际挂载修改)
PRETRAIN_DATA=${PRETRAIN_DATA:-/data/wudao/wudao_10pct.jsonl}
SFT_DATA=${SFT_DATA:-/data/firefly/firefly_1.6m.jsonl}
DISTILL_DATA=${DISTILL_DATA:-/data/distill/self_instruct.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-/checkpoints/reflex}
TEACHER=${TEACHER:-Qwen/Qwen2.5-7B-Instruct}

# ── 解析参数 ──────────────────────────────────────────────────────────
MODE="full"
RESUME=""
BATCH_SIZE=""
GRAD_ACCUM=""
LR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode) MODE="$2"; shift 2 ;;
        --resume) RESUME="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="--batch-size $2"; shift 2 ;;
        --grad-accum) GRAD_ACCUM="--grad-accum $2"; shift 2 ;;
        --lr) LR="--lr $2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── 日志 ──────────────────────────────────────────────────────────────
mkdir -p $OUTPUT_DIR
LOG_FILE="$OUTPUT_DIR/train_$(date +%Y%m%d_%H%M%S).log"

echo "============================================"
echo " NeuroStream-Reflex Training (H800 single)"
echo " Mode: $MODE"
echo " Output: $OUTPUT_DIR"
echo " Teacher: $TEACHER"
echo "============================================"

# ── 训练命令 ──────────────────────────────────────────────────────────
CMD="python train/train_reflex.py \
    --mode $MODE \
    --device cuda \
    --dtype bfloat16 \
    --pretrain-data $PRETRAIN_DATA \
    --sft-data $SFT_DATA \
    --distill-data $DISTILL_DATA \
    --teacher $TEACHER \
    --output-dir $OUTPUT_DIR \
    $BATCH_SIZE $GRAD_ACCUM $LR"

if [ -n "$RESUME" ]; then
    CMD="$CMD --resume $RESUME"
fi

echo "Running: $CMD"

# ── 执行（前台运行，可直接看到进度条） ──────────────────────────────────
$CMD
