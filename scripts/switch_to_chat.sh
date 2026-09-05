#!/usr/bin/env bash
# ============================================================
# switch_to_chat.sh — 将嫁接从 DeepSeek-V2-Lite(base) 切换为
#                    DeepSeek-V2-Lite-Chat（对话版）
# 用法: bash scripts/switch_to_chat.sh
#   ① 下载 Chat 权重（31GB，已存在则跳过）
#   ② 生成 Chat 嫁接 checkpoint（reflex_dsv2lite_chat.pt）
#   ③ 数值验证（top-1 应与 base 相当 ≈78%）
#   ④ 打印运行命令
# ============================================================
set -euo pipefail

CHAT_DIR="/root/autodl-tmp/data/DeepSeek-V2-Lite-Chat"
CHAT_CKPT="/root/autodl-tmp/checkpoints/reflex_dsv2lite_chat.pt"
BASE_CKPT="/root/autodl-tmp/checkpoints/reflex_dsv2lite_graft.pt"

echo "==> 磁盘空间检查（需要 ≥70GB 剩余）:"
df -h /root/autodl-tmp | tail -1

# ① 下载 Chat 权重（约 31GB bf16，MIT 许可）
if [ ! -f "$CHAT_DIR/config.json" ]; then
  echo "==> 下载 deepseek-ai/DeepSeek-V2-Lite-Chat ..."
  mkdir -p "$CHAT_DIR"
  modelscope download --model deepseek-ai/DeepSeek-V2-Lite-Chat \
      --local_dir "$CHAT_DIR"
else
  echo "==> Chat 权重已存在: $CHAT_DIR"
fi

# ② 生成 Chat 嫁接 checkpoint（与 base 完全同结构，代码零改动）
echo "==> 生成 Chat 嫁接 checkpoint（约 5-10 分钟）..."
python scripts/load_deepseek_graft.py \
    --model-path "$CHAT_DIR" \
    --output "$CHAT_CKPT" \
    --dtype bf16

# ③ 数值验证（判定同 base：top-1 >95% 最佳；≈78% 为 bf16 内核噪声）
echo "==> 数值验证 ..."
python scripts/verify_deepseek.py \
    --model-path "$CHAT_DIR" \
    --checkpoint "$CHAT_CKPT" \
    --device cuda --dtype bfloat16

cat <<MSG

========================================================
Chat 版嫁接完成！
  checkpoint: $CHAT_CKPT（≈46GB，与 base 版互不影响）

运行对话（Chat 版，eos 立即停 + 512 上限）:
  python run_mini.py --checkpoint $CHAT_CKPT \
      --tokenizer $CHAT_DIR \
      --device cuda --dtype bfloat16 \
      --no-online-ce --hide-think --gen-debug --sigma-cal \
      --eos-grace 0 --max-new-tokens 512

（Hebbian 对照关闭时追加: --hebbian-lr 0）

释放 base 版空间（可选，确认 Chat 跑通后）:
  rm -f $BASE_CKPT
  rm -rf /root/autodl-tmp/data/DeepSeek-V2-Lite
========================================================
MSG
