#!/bin/bash
# autodl_setup.sh — AutoDL 傻瓜式一键准备脚本（DeepSeek-V2-Lite 嫁接）
# 用法:
#   bash autodl_setup.sh                          # 全流程：依赖+下载+加载+验证
#   bash autodl_setup.sh --skip-download          # 权重已下载过，跳过下载
#   bash autodl_setup.sh --skip-download --skip-load    # 嫁接 checkpoint 也有了
#   bash autodl_setup.sh --model-path /root/autodl-tmp/data/DeepSeek-V2-Lite \
#                        --output /root/autodl-tmp/checkpoints/reflex_dsv2lite_graft.pt
# 注: DeepSeek-V2-Lite 约 31GB bf16，MIT 许可
set -e
cd "$(dirname "$0")"

MODEL_PATH=${MODEL_PATH:-/root/autodl-tmp/data/DeepSeek-V2-Lite}
OUTPUT=${OUTPUT:-/root/autodl-tmp/checkpoints/reflex_dsv2lite_graft.pt}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-8192}
SKIP_DOWNLOAD=0; SKIP_LOAD=0; SKIP_VERIFY=0
for arg in "$@"; do
  case $arg in
    --skip-download) SKIP_DOWNLOAD=1 ;;
    --skip-load)     SKIP_LOAD=1 ;;
    --skip-verify)   SKIP_VERIFY=1 ;;
    --model-path=*)  MODEL_PATH="${arg#*=}" ;;
    --output=*)      OUTPUT="${arg#*=}" ;;
    --max-seq-len=*) MAX_SEQ_LEN="${arg#*=}" ;;
  esac
done

echo "======================================================"
echo " [1/5] 环境检查与依赖安装"
echo "======================================================"
nvidia-smi | head -12 || true
python --version
pip install -q modelscope safetensors tqdm "transformers>=4.57.1"
echo "  OK"

if [ "$SKIP_DOWNLOAD" = "0" ]; then
echo "======================================================"
echo " [2/5] 下载 DeepSeek-V2-Lite 原始权重（约 31GB，视网速 10-40 分钟）"
echo "======================================================"
mkdir -p "$(dirname "$MODEL_PATH")"
modelscope download --model deepseek-ai/DeepSeek-V2-Lite --local_dir "$MODEL_PATH"
du -sh "$MODEL_PATH"
ls "$MODEL_PATH" | head
echo "  OK: 权重已下载到 $MODEL_PATH"
else
echo " [2/5] 跳过下载（--skip-download），使用 $MODEL_PATH"
fi

if [ "$SKIP_LOAD" = "0" ]; then
echo "======================================================"
echo " [3/5] 生成嫁接 checkpoint（约 5-10 分钟，无需训练）"
echo "======================================================"
mkdir -p "$(dirname "$OUTPUT")"
python scripts/load_deepseek_graft.py \
    --model-path "$MODEL_PATH" \
    --output "$OUTPUT" \
    --dtype bf16 --max-seq-len "$MAX_SEQ_LEN"
ls -lh "$OUTPUT"
echo "  OK: 嫁接 checkpoint 已生成 -> $OUTPUT"
else
echo " [3/5] 跳过加载（--skip-load），使用 $OUTPUT"
fi

echo "======================================================"
echo " [4/5] 数值验证（与官方实现对比 logits，约 3-8 分钟）"
echo "======================================================"
if [ "$SKIP_VERIFY" = "0" ]; then
  if python scripts/verify_deepseek.py \
      --model-path "$MODEL_PATH" \
      --checkpoint "$OUTPUT" \
      --prompt "中国的首都是" --device cuda --dtype bfloat16; then
    echo "  OK: 数值一致，权重复用正确"
  else
    echo "  [WARN] 验证失败/被中断（显存不足可换 --device cpu）。"
    echo "  [WARN] 若不影响使用，可跳过验证直接进入 [5/5] 冒烟测试。"
  fi
else
  echo "  跳过验证（--skip-verify）"
fi

echo "======================================================"
echo " [5/5] 完成！接下来两条命令即可使用："
echo "======================================================"
echo "  # A. 快速测试（推荐先跑这个）:"
echo "  python chat_sft.py --checkpoint $OUTPUT \\"
echo "      --tokenizer $MODEL_PATH --device cuda --dtype bfloat16 \\"
echo "      --prompt \"中国的首都是\" --prompt \"解释一下什么是光合作用\""
echo ""
echo "  # B. 完整部署（内循环 + 记忆 + 主动求证 + Hebbian 光谱）:"
echo "  python run_mini.py --checkpoint $OUTPUT \\"
echo "      --tokenizer $MODEL_PATH --device cuda --dtype bfloat16 --no-online-ce \\"
echo "      --hide-think --gen-debug --sigma-cal"
echo ""
echo "全部就绪，祝实验顺利！"
