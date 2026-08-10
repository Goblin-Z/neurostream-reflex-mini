#!/bin/bash
# ============================================================
# 全自动训练流水线：数据生成 → 清洗 → 硬标签SFT → 软标签KD → 低T精修
#
# 幂等：每阶段完成即产生标记（输出文件），重跑自动跳过已完成阶段。
# 中断后重跑 = 从断点继续。nohup 友好：
#   nohup bash scripts/run_pipeline.sh > /root/autodl-tmp/pipeline.log 2>&1 &
#   查看: tail -f /root/autodl-tmp/pipeline.log
# ============================================================
set -e

ROOT=/root/autodl-tmp/neurostream_reflex_v2_mini
DATA=/root/autodl-tmp/data
CKPT=/root/autodl-tmp/checkpoints/reflex-mini
TEACHER=$DATA/qwen2.5-1.5b-instruct
BS=8

cd $ROOT

step() {
  echo ""
  echo "=================================================="
  echo "  $1"
  echo "  $(date '+%Y-%m-%d %H:%M:%S')"
  echo "=================================================="
}

# ── 前置检查 ─────────────────────────────────────────────
if [ ! -f "$CKPT/pretrain_final.pt" ]; then
  echo "[FATAL] 缺少 pretrain_final.pt（唯一允许保留的旧产物）: $CKPT/pretrain_final.pt"
  exit 1
fi

# ── 阶段 0a: 模板 QA 2万 ─────────────────────────────────
if [ ! -f "$DATA/qa_tpl_20k.jsonl" ]; then
  step "0a. 生成模板 QA (2万)"
  python -u scripts/generate_qa.py --teacher $TEACHER \
    --output $DATA/qa_tpl_20k.jsonl \
    --max-samples 20000 --mode qa --batch-size $BS --device cuda
else
  echo "[skip] qa_tpl_20k.jsonl 已存在"
fi

# ── 阶段 0b: self-instruct QA 4万 ────────────────────────
if [ ! -f "$DATA/qa_si_40k.jsonl" ]; then
  step "0b. 生成 self-instruct QA (4万)"
  python -u scripts/generate_qa.py --teacher $TEACHER \
    --output $DATA/qa_si_40k.jsonl \
    --max-samples 40000 --mode self-instruct --batch-size $BS --device cuda
else
  echo "[skip] qa_si_40k.jsonl 已存在"
fi

# ── 阶段 0c: multi-turn 对话 4万 ─────────────────────────
if [ ! -f "$DATA/mt_40k.jsonl" ]; then
  step "0c. 生成多轮对话 (4万)"
  python -u scripts/generate_qa.py --teacher $TEACHER \
    --output $DATA/mt_40k.jsonl \
    --max-samples 40000 --mode multi-turn --batch-size $BS --device cuda
else
  echo "[skip] mt_40k.jsonl 已存在"
fi

# ── 阶段 0d: 清洗合并 ────────────────────────────────────
if [ ! -f "$DATA/sft_kd_clean.jsonl" ]; then
  step "0d. 清洗合并 (模板QA + self-instruct + 多轮)"
  python -u scripts/filter_qa.py \
    --inputs $DATA/qa_tpl_20k.jsonl $DATA/qa_si_40k.jsonl \
    --multiturn-files $DATA/mt_40k.jsonl \
    --output $DATA/sft_kd_clean.jsonl
else
  echo "[skip] sft_kd_clean.jsonl 已存在"
fi

# ── 阶段 1: 硬标签 SFT (1.3 epoch) ───────────────────────
if [ ! -f "$CKPT/sft_final.pt" ]; then
  step "阶段1. 硬标签 SFT (3000步, lr=2e-5)"
  python -u -m train.train_reflex --mode sft \
    --sft-data $DATA/sft_kd_clean.jsonl \
    --sft-steps 3000 --batch-size $BS --grad-accum 8 \
    --lr 2e-5 --dtype bfloat16 \
    --resume $CKPT/pretrain_final.pt --output-dir $CKPT
else
  echo "[skip] sft_final.pt 已存在"
fi

# ── 阶段 2: 软标签 KD (T=4, CE+KD 联合) ─────────────────
# teacher logits 自动生成（首次运行 ~2-3h，之后存在即跳过）
if [ ! -f "$CKPT/distill_final.pt" ]; then
  step "阶段2. 软标签 KD (4000步, T=4, lr=1e-5)"
  python -u -m train.train_reflex --mode distill \
    --distill-data $DATA/sft_kd_clean.jsonl --teacher $TEACHER \
    --distill-steps 4000 --batch-size $BS --grad-accum 4 \
    --lr 1e-5 --distill-temperature 4.0 --distill-kd-weight 0.8 \
    --dtype bfloat16 \
    --resume $CKPT/sft_final.pt --output-dir $CKPT
  # 备份高 T 产物（阶段3会覆盖 distill_final.pt）
  cp $CKPT/distill_final.pt $CKPT/distill_highT.pt
  echo "[info] 高T产物已备份: distill_highT.pt"
else
  echo "[skip] distill_final.pt 已存在"
fi

# ── 阶段 3: 低 T 精修 (T=1, fresh optimizer) ────────────
if [ ! -f "$CKPT/REFINED_DONE" ]; then
  step "阶段3. 低T精修 (2000步, T=1, lr=5e-6)"
  python -u -m train.train_reflex --mode distill \
    --distill-data $DATA/sft_kd_clean.jsonl --teacher $TEACHER \
    --distill-steps 6000 --batch-size $BS --grad-accum 4 \
    --lr 5e-6 --distill-temperature 1.0 --distill-kd-weight 0.9 \
    --fresh-optimizer --dtype bfloat16 \
    --resume $CKPT/distill_final.pt --output-dir $CKPT
  # 精修产物归档
  mv $CKPT/distill_final.pt $CKPT/distill_refined.pt
  touch $CKPT/REFINED_DONE
  echo "[info] 精修完成: distill_refined.pt"
else
  echo "[skip] 精修已完成 (REFINED_DONE 存在)"
fi

# ── 完成 ─────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "  全部完成: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================="
echo "最终模型: $CKPT/distill_refined.pt"
echo "高T版本:  $CKPT/distill_highT.pt"
echo ""
echo "测试命令见 README（模板 prompt 问答/多轮）"
