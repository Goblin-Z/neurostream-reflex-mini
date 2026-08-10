#!/bin/bash
# Package ReflexMini for SFT inference test on another machine.
# Run from project root: bash pack_for_test.sh
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$ROOT/../reflex_mini_sft_test.tar.gz}"
STAGE="$(mktemp -d)"
NAME="reflex_mini_sft_test"
DEST="$STAGE/$NAME"
mkdir -p "$DEST"

echo "Staging to $DEST"
# runtime code only
for d in config core loop learn interaction improve; do
  cp -a "$ROOT/$d" "$DEST/"
done
# drop caches
find "$DEST" -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
find "$DEST" -name '*.pyc' -delete 2>/dev/null || true

cp -a "$ROOT/chat_sft.py" "$DEST/"
cp -a "$ROOT/__init__.py" "$DEST/" 2>/dev/null || true
cp -a "$ROOT/sft_15B_077B_final.pt" "$DEST/"

cat > "$DEST/README_TEST.md" << 'EOF'
# ReflexMini SFT 测试包

## 内容
- ReflexMini 推理代码（config/core/loop/learn/interaction/improve）
- chat_sft.py：交互 / 单次生成测试
- sft_15B_077B_final.pt：SFT 权重

## 依赖
```bash
pip install torch transformers safetensors tqdm
# 建议与训练侧一致：PyTorch 2.x + CUDA
export HF_ENDPOINT=https://hf-mirror.com   # 国内镜像
```

## Tokenizer
训练时用 Qwen2.5 词表。二选一：
1. 自动从 HF 拉（默认 Qwen/Qwen2.5-0.5B）
2. 本地路径：`--tokenizer /path/to/qwen2.5-0.5b`

## 运行
```bash
cd reflex_mini_sft_test

# 交互
python chat_sft.py --checkpoint sft_15B_077B_final.pt --device cuda

# 单次测试
python chat_sft.py --checkpoint sft_15B_077B_final.pt --device cuda \
  --prompt "中国的首都是" \
  --prompt "解释一下什么是光合作用" \
  --max-new-tokens 80
```
EOF

echo "Packing -> $OUT"
tar -czf "$OUT" -C "$STAGE" "$NAME"
rm -rf "$STAGE"
ls -lh "$OUT"
echo "Done: $OUT"
