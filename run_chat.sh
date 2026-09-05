#!/usr/bin/env bash
# ============================================================
# run_chat.sh — 一键运行 DeepSeek-V2-Lite-Chat 嫁接（傻瓜式）
#
# 自动完成: 依赖检查 → Chat 权重下载(31GB) → checkpoint 生成(46GB)
#          → 启动双循环对话。重复运行自动跳过已完成步骤。
#
# 用法: bash run_chat.sh
#       bash run_chat.sh --force      # 强制重建 checkpoint
#       bash run_chat.sh --verify     # 生成后追加数值验证
#       bash run_chat.sh --online-ce  # 打开每轮 CE 在线训练（默认关闭）
#       bash run_chat.sh --hebbian-lr 0   # 透传参数给 run_mini.py
# ============================================================
set -euo pipefail
cd "$(dirname "$0")"

CHAT_DIR="${DSV2_CHAT_DIR:-/root/autodl-tmp/data/DeepSeek-V2-Lite-Chat}"
CHAT_CKPT="${DSV2_CHAT_CKPT:-/root/autodl-tmp/checkpoints/reflex_dsv2lite_chat.pt}"
FORCE=0
VERIFY=0
ONLINE_CE=0
EXTRA=()
for a in "$@"; do
  case "$a" in
    --force) FORCE=1 ;;
    --verify) VERIFY=1 ;;
    --online-ce) ONLINE_CE=1 ;;
    *) EXTRA+=("$a") ;;
  esac
done

echo "=========================================================="
echo " NeuroStream-Reflex × DeepSeek-V2-Lite-Chat 一键运行"
echo "=========================================================="

# 0) 依赖检查（AutoDL 镜像一般自带 torch；pip 需镜像源，逐个装不缺的）
#    清华源（PyPI 直连在 AutoDL 上经常不可达）
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null || true
echo "==> 依赖检查 ..."
python3 -c "import torch" 2>/dev/null || { echo "  装 torch ..."; pip install -q torch; }
python3 -c "import transformers" 2>/dev/null || { echo "  装 transformers ..."; pip install -q transformers; }
python3 -c "import safetensors" 2>/dev/null || { echo "  装 safetensors ..."; pip install -q safetensors; }
python3 -c "import modelscope" 2>/dev/null || { echo "  装 modelscope ..."; pip install -q modelscope; }
echo "   依赖 OK：$(python3 -c "import torch,transformers,safetensors,modelscope; print('torch', torch.__version__, '| transformers', transformers.__version__)")"

# 1) Chat 权重（31GB，MIT）
if [ ! -f "$CHAT_DIR/config.json" ]; then
  echo "==> 下载 deepseek-ai/DeepSeek-V2-Lite-Chat（31GB，需几分钟）..."
  mkdir -p "$CHAT_DIR"
  modelscope download --model deepseek-ai/DeepSeek-V2-Lite-Chat \
      --local_dir "$CHAT_DIR"
else
  echo "==> Chat 权重已就绪: $CHAT_DIR"
fi

# 2) 嫁接 checkpoint（46GB）
if [ ! -f "$CHAT_CKPT" ] || [ "$FORCE" = 1 ]; then
  echo "==> 生成 Chat 嫁接 checkpoint（5-10 分钟）..."
  python scripts/load_deepseek_graft.py \
      --model-path "$CHAT_DIR" \
      --output "$CHAT_CKPT" \
      --dtype bf16
  echo "==> checkpoint 生成完成: $CHAT_CKPT"
else
  echo "==> Chat checkpoint 已就绪: $CHAT_CKPT（--force 可重建）"
fi

# 3) 可选数值验证
if [ "$VERIFY" = 1 ]; then
  echo "==> 数值验证（与官方实现对比）..."
  python scripts/verify_deepseek.py \
      --model-path "$CHAT_DIR" \
      --checkpoint "$CHAT_CKPT" \
      --device cuda --dtype bfloat16
fi

# 4) 启动双循环对话（透传额外参数；默认关闭每轮 CE，--online-ce 打开）
echo ""
echo "==> 启动 Chat 版双循环对话（输入 quit 退出；stats 看内循环）..."
RUN_OPTS=()
[ "$ONLINE_CE" = 0 ] && RUN_OPTS+=(--no-online-ce)
exec python run_mini.py --checkpoint "$CHAT_CKPT" \
    --tokenizer "$CHAT_DIR" \
    --device cuda --dtype bfloat16 \
    ${RUN_OPTS[@]+"${RUN_OPTS[@]}"} \
    --hide-think --gen-debug --sigma-cal \
    --eos-grace 0 --max-new-tokens 512 "${EXTRA[@]}"
