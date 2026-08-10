#!/bin/bash
# ============================================================
# run_all.sh - AutoDL 一键完整训练流程
#
# 用法:
#   cd /root/autodl-tmp/neurostream_reflex_v2_mini
#   bash scripts/run_all.sh
#
# 前提: 已将项目代码上传到 /root/autodl-tmp/neurostream_reflex_v2_mini
# ============================================================
set -e

# ── 路径配置 ──
PROJECT_DIR=/root/autodl-tmp/neurostream_reflex_v2_mini
DATA_DIR=/root/autodl-tmp/data
CKPT_DIR=/root/autodl-tmp/checkpoints/reflex-mini
export HF_ENDPOINT=https://hf-mirror.com
export DATA_DIR=$DATA_DIR

cd $PROJECT_DIR
mkdir -p $CKPT_DIR

# ── 环境检查 ──
echo "============================================"
echo "环境检查"
echo "============================================"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# ── Step 0: 安装依赖 ──
echo "============================================"
echo "Step 0: 安装依赖"
echo "============================================"
pip install -q transformers datasets safetensors pandas tqdm huggingface_hub 2>/dev/null || true
echo "Done."
echo ""

# ── Step 1: 下载数据 ──
# 检查所有关键数据是否完整 (pretrain_all.jsonl 和 sft_all.jsonl 存在且有大小)
if [ ! -f "$DATA_DIR/pretrain_all.jsonl" ] || [ ! -f "$DATA_DIR/sft_all.jsonl" ] \
    || [ "$(stat --format=%s $DATA_DIR/pretrain_all.jsonl 2>/dev/null)" -lt 1000000 ] \
    || [ "$(ls $DATA_DIR/skypile/data/2023-*_zh_head_*.jsonl 2>/dev/null | wc -l)" -lt 18 ] \
    || [ "$(ls $DATA_DIR/cwt-hq/data/*.parquet 2>/dev/null | wc -l)" -lt 5 ] \
    || [ "$(stat --format=%s $DATA_DIR/firefly/firefly-train-1.1M.jsonl 2>/dev/null)" -lt 100000000 ] \
    || [ ! -f "$DATA_DIR/qwen2.5-0.5b/config.json" ]; then
    echo "============================================"
    echo "Step 1: 下载数据 (部分缺失, 补充下载)"
    echo "============================================"
    python scripts/download_data.py
else
    echo "============================================"
    echo "Step 1: 数据已完整, 跳过下载"
    echo "============================================"
    du -sh $DATA_DIR/*/
fi
echo ""

# ── Step 2: 预处理数据 ──
if [ ! -f "$DATA_DIR/pretrain_all.jsonl" ] || [ ! -f "$DATA_DIR/sft_all.jsonl" ] \
    || [ "$(stat --format=%s $DATA_DIR/pretrain_all.jsonl 2>/dev/null || echo 0)" -lt 1000000 ] \
    || [ "$(stat --format=%s $DATA_DIR/sft_all.jsonl 2>/dev/null || echo 0)" -lt 1000000 ]; then
    echo "============================================"
    echo "Step 2: 预处理数据"
    echo "============================================"
    python scripts/prepare_data.py
else
    echo "============================================"
    echo "Step 2: 预处理数据已存在, 跳过"
    echo "============================================"
    wc -l $DATA_DIR/pretrain_all.jsonl $DATA_DIR/sft_all.jsonl
fi
echo ""

# ── Step 3: Embedding 初始化 ──
if [ ! -f "$DATA_DIR/reflex_mini_init.pt" ]; then
    echo "============================================"
    echo "Step 3: Embedding 初始化"
    echo "============================================"
    python scripts/init_from_qwen.py
else
    echo "============================================"
    echo "Step 3: Embedding 初始化已存在, 跳过"
    echo "============================================"
fi
echo ""

# ── Step 4: Phase 1 - Pretrain ──
# Skip pretrain if pretrain_final.pt exists (already complete)
if [ -f "$CKPT_DIR/pretrain_final.pt" ]; then
    echo "============================================"
    echo "Step 4: Pretrain 已完成, 跳过"
    echo "  $CKPT_DIR/pretrain_final.pt"
    echo "============================================"
else
    echo "============================================"
    echo "Step 4: Phase 1 - Pretrain (228883 steps, 15B tokens)"
    echo "  batch=8, accum=8, seq=1024, lr=3e-4, bf16"
    echo "  预计时间: ~3 天"
    echo "============================================"
    # Resume from latest checkpoint if exists
    RESUME_CKPT=$DATA_DIR/reflex_mini_init.pt
    if ls $CKPT_DIR/pretrain_step*.pt >/dev/null 2>&1; then
        RESUME_CKPT=$(ls -t $CKPT_DIR/pretrain_step*.pt | head -1)
    fi
    echo "  [INFO] Resuming from: $RESUME_CKPT"
    python -m train.train_reflex \
        --mode pretrain \
        --pretrain-data $DATA_DIR/pretrain_all.jsonl \
        --pretrain-steps 228883 \
        --batch-size 8 \
        --grad-accum 8 \
        --lr 3e-4 \
        --dtype bfloat16 \
        --resume $RESUME_CKPT \
        --output-dir $CKPT_DIR
fi
echo ""

# ── Step 5: Phase 2 - SFT ──
# Skip SFT if sft_final.pt exists (already complete)
if [ -f "$CKPT_DIR/sft_final.pt" ]; then
    echo "============================================"
    echo "Step 5: SFT 已完成, 跳过"
    echo "============================================"
else
    echo "============================================"
    echo "Step 5: Phase 2 - SFT (38000 steps, 2 epochs)"
    echo "  batch=8, accum=8, lr=1e-5, bf16 (batch/2 + accum x2: same effective batch, fits 32GB)"
    echo "  预计时间: ~1 天"
    echo "============================================"
    # Resume from latest SFT checkpoint if exists
    if ls $CKPT_DIR/sft_step*.pt >/dev/null 2>&1; then
        SFT_RESUME=$(ls -t $CKPT_DIR/sft_step*.pt | head -1)
    else
        SFT_RESUME="$CKPT_DIR/pretrain_final.pt"
    fi
    echo "  [INFO] SFT resuming from: $SFT_RESUME"
    python -m train.train_reflex \
        --mode sft \
        --sft-data $DATA_DIR/sft_all.jsonl \
        --sft-steps 38000 \
        --batch-size 8 \
        --grad-accum 8 \
        --lr 1e-5 \
        --dtype bfloat16 \
        --resume $SFT_RESUME \
        --output-dir $CKPT_DIR
fi
echo ""

# ── Step 6: Phase 2.5 - Consciousness Warmup ──
echo "============================================"
echo "Step 6: Phase 2.5 - Consciousness Warmup (1000 steps)"
echo "  Pre-train SelfModel + Critic before deployment"
echo "  预计时间: ~10 分钟"
echo "============================================"
python -c "
import torch, sys
sys.path.insert(0, '.')
from config.model_config import ReflexMiniConfig
from core.model import ReflexModel
from train.train_reflex import ReflexTrainer, TrainConfig
from transformers import AutoTokenizer

config = ReflexMiniConfig()
model = ReflexModel(config)
tokenizer = AutoTokenizer.from_pretrained('$DATA_DIR/qwen2.5-0.5b', trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model._decode_tokenizer = tokenizer

cfg = TrainConfig(mode='full', device='cuda', dtype='bfloat16',
                  sft_data='$DATA_DIR/sft_all.jsonl',
                  output_dir='$CKPT_DIR',
                  resume='$CKPT_DIR/sft_final.pt')
model.to('cuda')
trainer = ReflexTrainer(model, tokenizer, cfg)
trainer.load_checkpoint('$CKPT_DIR/sft_final.pt')
trainer.train_consciousness_warmup(steps=1000)
"
echo ""

# ── Step 7: Phase 3 - Distill ──
# Skip distill if distill_final.pt exists (already complete)
if [ -f "$CKPT_DIR/distill_final.pt" ]; then
    echo "============================================"
    echo "Step 7: Distill 已完成, 跳过"
    echo "============================================"
else
    echo "============================================"
    echo "Step 7: Phase 3 - Distill (10000 steps)"
    echo "  teacher=Qwen2.5-1.5B-Instruct, batch=8, accum=4 (fits 32GB)"
    echo "  预计时间: ~0.3 天"
    echo "============================================"
    # Resume from the latest distill_step checkpoint if one exists
    # (interrupted distill resumes in-place); otherwise from warmup_final.pt
    # so the warmed-up SelfModel/Critic states carry into the distilled
    # model (otherwise warmup is wasted work).
    if ls $CKPT_DIR/distill_step*.pt >/dev/null 2>&1; then
        DISTILL_RESUME=$(ls -t $CKPT_DIR/distill_step*.pt | head -1)
    elif [ -f "$CKPT_DIR/warmup_final.pt" ]; then
        DISTILL_RESUME=$CKPT_DIR/warmup_final.pt
    else
        DISTILL_RESUME=$CKPT_DIR/sft_final.pt
    fi
    echo "  [INFO] Distill resuming from: $DISTILL_RESUME"
    python -m train.train_reflex \
        --mode distill \
        --distill-data $DATA_DIR/sft_all.jsonl \
        --teacher $DATA_DIR/qwen2.5-1.5b-instruct \
        --distill-steps 10000 \
        --batch-size 8 \
        --grad-accum 4 \
        --lr 2e-5 \
        --dtype bfloat16 \
        --resume $DISTILL_RESUME \
        --output-dir $CKPT_DIR
    echo ""
fi
# ── 完成 ──
echo "============================================"
echo "训练完成!"
echo "============================================"
echo "Final checkpoint: $CKPT_DIR/distill_final.pt"
echo ""
echo "交互测试:"
echo "  python run_reflex.py --resume $CKPT_DIR/distill_final.pt"
echo ""
ls -lh $CKPT_DIR/
