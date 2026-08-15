#!/bin/bash
# ============================================================
# 全自动训练流水线：数据生成 → 清洗 → 硬标签SFT → Warmup → 软标签KD → 低T精修
#
# 幂等：每阶段完成即产生标记（输出文件），重跑自动跳过已完成阶段。
# 中断后重跑 = 从断点继续。nohup 友好：
#   nohup bash scripts/run_pipeline.sh > /root/autodl-tmp/pipeline.log 2>&1 &
#   查看: tail -f /root/autodl-tmp/pipeline.log
#
# 2026-08 修订 v2（训练核心目的 = 组织语言能力：逻辑提问 + 问答 + 多轮对话）：
#   - 弃用 sft_all.jsonl（实测对能力提升无效）
#   - 数据 = sft_kd_clean.jsonl（教师 QA/多轮，10万）+ clarify_30k.jsonl（澄清式提问对话，新增）
#   - SFT 步数按数据量校准（~3-4 epoch，防过拟合）
#   - 新增阶段 1.5 Consciousness Warmup（SelfModel/Critic/sigma 校准，KD 前置）
#   - KD/精修 resume 链：warmup_final.pt → distill_final.pt → distill_refined.pt
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
  echo "[FATAL] 缺少 pretrain_final.pt: $CKPT/pretrain_final.pt"
  echo "  若只有 sft_kd_150k_final.pt，请先执行:"
  echo "    cp sft_kd_150k_final.pt $CKPT/pretrain_final.pt"
  echo "  （该文件 = 预训练 + SFT 3000 步产物，可作继续训练起点）"
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

# ── 阶段 0d: 清洗合并教师合成数据 ────────────────────────
if [ ! -f "$DATA/sft_kd_clean.jsonl" ]; then
  step "0d. 清洗合并 (模板QA + self-instruct + 多轮)"
  python -u scripts/filter_qa.py \
    --inputs $DATA/qa_tpl_20k.jsonl $DATA/qa_si_40k.jsonl \
    --multiturn-files $DATA/mt_40k.jsonl \
    --output $DATA/sft_kd_clean.jsonl
else
  echo "[skip] sft_kd_clean.jsonl 已存在"
fi

# ── 阶段 0e: 生成澄清式提问对话数据（核心：训练"逻辑提问"能力）──
# 信息不足请求 → 主动提问 → 用户补充 → 完整回答。可选但强烈推荐。
if [ ! -f "$DATA/clarify_30k.jsonl" ]; then
  step "0e. 生成澄清式提问数据 (3万条)"
  python -u scripts/generate_ask_data.py --teacher $TEACHER \
    --output $DATA/clarify_30k.jsonl \
    --max-samples 30000 --batch-size $BS --device cuda
else
  echo "[skip] clarify_30k.jsonl 已存在"
fi

# ── 阶段 0f: 合并 SFT 数据（教师 QA/多轮 + 澄清式提问；无 sft_all）──
if [ ! -f "$DATA/sft_mix.jsonl" ]; then
  step "0f. 合并 SFT 数据 (sft_kd_clean + clarify)"
  python -u -c "
import json
seen = set()
n = 0
with open('$DATA/sft_mix.jsonl', 'w', encoding='utf-8') as out:
    for path in ['$DATA/sft_kd_clean.jsonl', '$DATA/clarify_30k.jsonl']:
        try:
            with open(path, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except Exception:
                        continue
                    key = (item.get('instruction', ''), item.get('output', ''))
                    if key in seen:
                        continue
                    seen.add(key)
                    out.write(line + '\n')
                    n += 1
        except FileNotFoundError:
            print(f'  [warn] 缺少 {path}，跳过')
print(f'  sft_mix.jsonl: {n} 条')
"
else
  echo "[skip] sft_mix.jsonl 已存在"
fi

# ── 阶段 1: 硬标签 SFT ───────────────────────────────────
if [ ! -f "$CKPT/sft_final.pt" ]; then
  step "阶段1. 硬标签 SFT (2000步≈3-4epoch, lr=2e-5, sft_mix)"
  python -u -m train.train_reflex --mode sft \
    --sft-data $DATA/sft_mix.jsonl \
    --sft-steps 2000 --batch-size $BS --grad-accum 8 \
    --lr 2e-5 --dtype bfloat16 \
    --resume $CKPT/pretrain_final.pt --output-dir $CKPT
else
  echo "[skip] sft_final.pt 已存在"
fi

# ── 阶段 1.5: Consciousness Warmup（SelfModel + Critic + sigma 校准）──
# 2026-08 新增：SelfModel/Critic 随机起步违背"意识流有内容"；
# sigma 校准让主动求证的触发信号在部署前就对齐 CE 不确定性。
if [ ! -f "$CKPT/warmup_final.pt" ]; then
  step "阶段1.5. Consciousness Warmup (1000步)"
  python -u -c "
import torch, sys
sys.path.insert(0, '.')
from config.model_config import ReflexMiniConfig
from core.model import ReflexModel
from train.train_reflex import ReflexTrainer, TrainConfig
from transformers import AutoTokenizer

config = ReflexMiniConfig()
model = ReflexModel(config)
tokenizer = AutoTokenizer.from_pretrained('$DATA/qwen2.5-0.5b', trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model._decode_tokenizer = tokenizer

cfg = TrainConfig(mode='full', device='cuda', dtype='bfloat16',
                  sft_data='$DATA/sft_mix.jsonl',
                  output_dir='$CKPT',
                  resume='$CKPT/sft_final.pt')
model.to('cuda')
trainer = ReflexTrainer(model, tokenizer, cfg)
trainer.load_checkpoint('$CKPT/sft_final.pt')
trainer.train_consciousness_warmup(steps=1000)
"
else
  echo "[skip] warmup_final.pt 已存在"
fi

# ── 阶段 2: 软标签 KD（resume 自 warmup_final.pt，保留预热成果）──
# teacher logits 自动生成（首次运行 ~2-3h，之后存在即跳过）
if [ ! -f "$CKPT/distill_final.pt" ]; then
  step "阶段2. 软标签 KD (4000步, T=4, lr=1e-5)"
  if [ -f "$CKPT/warmup_final.pt" ]; then
    KD_RESUME=$CKPT/warmup_final.pt
  else
    KD_RESUME=$CKPT/sft_final.pt
  fi
  echo "  [info] KD resuming from: $KD_RESUME"
  python -u -m train.train_reflex --mode distill \
    --distill-data $DATA/sft_mix.jsonl --teacher $TEACHER \
    --distill-steps 4000 --batch-size $BS --grad-accum 4 \
    --lr 1e-5 --distill-temperature 4.0 --distill-kd-weight 0.8 \
    --dtype bfloat16 \
    --resume $KD_RESUME --output-dir $CKPT
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
    --distill-data $DATA/sft_mix.jsonl --teacher $TEACHER \
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

# ── 阶段 4: 记忆微调（长程引用多轮数据，可选但推荐）─────
# 需要先有 mt_memory_20k.jsonl（generate_qa.py --mode multi-turn --memory-tune）
if [ -f "$DATA/mt_memory_20k.jsonl" ] && [ ! -f "$CKPT/memory_tuned.pt" ]; then
  step "阶段4. 记忆微调 (3000步, KV 历史真实参与训练)"
  python -u train_memory.py \
    --checkpoint $CKPT/distill_refined.pt \
    --data $DATA/mt_memory_20k.jsonl \
    --steps 3000 --lr 1e-5 --batch-size 4 \
    --tokenizer $DATA/qwen2.5-0.5b \
    --output-dir $CKPT
else
  echo "[skip] 记忆微调（需要 mt_memory_20k.jsonl；可选阶段）"
fi

# ── 完成 ─────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "  全部完成: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================="
echo "最终模型: $CKPT/distill_refined.pt"
echo "高T版本:  $CKPT/distill_highT.pt"
echo "记忆微调: $CKPT/memory_tuned.pt（若执行了阶段4）"
echo ""
echo "部署测试:"
echo "  python run_mini.py --checkpoint $CKPT/distill_refined.pt --tokenizer $DATA/qwen2.5-0.5b"
echo ""
