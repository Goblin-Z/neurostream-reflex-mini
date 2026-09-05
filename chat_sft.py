"""
SFT checkpoint interactive / one-shot test for ReflexMini.

Usage:
  # interactive
  python chat_sft.py --checkpoint sft_15B_077B_final.pt --device cuda

  # one-shot prompts
  python chat_sft.py --checkpoint sft_15B_077B_final.pt --device cuda \
      --prompt "中国的首都是" --prompt "解释一下什么是光合作用"
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import AutoTokenizer

from config.model_config import ReflexMiniConfig
from core.model import ReflexModel


def main():
    p = argparse.ArgumentParser(description='Test ReflexMini SFT checkpoint')
    p.add_argument('--checkpoint', default='sft_15B_077B_final.pt')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--tokenizer', default=None,
                   help='HF id or local path (default: Qwen/Qwen2.5-0.5B)')
    p.add_argument('--max-new-tokens', type=int, default=900,
                   help='生成上限（用户要求 900；仅为上限，模型输出 '
                        '<|im_end|>/eos 会自动停止，不会强制跑满）')
    p.add_argument('--hide-think', action='store_true',
                   help='显示时剥离 <think>...</think> 思考块（只看最终答案）')
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--top-p', type=float, default=0.9)
    p.add_argument('--prompt', action='append', default=None,
                   help='One-shot prompt (repeatable). If omitted, enter interactive mode.')
    p.add_argument('--dtype', default='bfloat16',
                   choices=['bfloat16', 'float16', 'float32'])
    args = p.parse_args()

    device = args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        print('[WARN] CUDA not available, falling back to cpu')
        device = 'cpu'

    dtype = {
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
        'float32': torch.float32,
    }[args.dtype]
    if device == 'cpu' and dtype != torch.float32:
        dtype = torch.float32

    tok_name = args.tokenizer or os.environ.get(
        'TOKENIZER_PATH', 'Qwen/Qwen2.5-0.5B')
    print(f'Loading tokenizer: {tok_name}')
    tok = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print('Building ReflexModel...')
    ckpt_path = args.checkpoint
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ckpt_path)
    print(f'Loading checkpoint: {ckpt_path}')
    # mmap 加载：不把 54GB checkpoint 全量读进物理内存（需 torch>=2.1）
    try:
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False, mmap=True)
    except TypeError:
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    from config.model_config import config_from_checkpoint
    config = config_from_checkpoint(ck)
    model = ReflexModel(config)
    # 显存/内存优化：加载前先转目标 dtype——
    # fp32 中间态 = 29.45B×4B ≈ 118GB CPU RAM（叠加 54GB checkpoint 会被 OOM Killed）
    load_dtype = dtype if device != 'cpu' else torch.float32
    model.to(dtype=load_dtype)
    sd = ck.get('model_state_dict', ck)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f'  step={ck.get("step", "?")} phase={ck.get("phase", "?")} '
          f'backbone={getattr(config, "backbone", "reflex")}')
    print(f'  missing={len(missing)} unexpected={len(unexpected)}')
    # 释放 checkpoint 引用（mmap 页缓存可回收）
    del sd, ck
    import gc
    gc.collect()
    model._decode_tokenizer = tok
    model.to(device=device, dtype=load_dtype)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Ready | params={n_params:,} device={device} dtype={dtype}')
    print('Type text, or "quit" to exit.\n')

    def generate_once(prompt: str) -> str:
        ids = tok(prompt, return_tensors='pt')['input_ids'].to(device)
        with torch.no_grad():
            out = model.generate(
                ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=1.2,
            )
        text = tok.decode(out[0], skip_special_tokens=True)
        if args.hide_think:
            import re
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.S)
            text = re.sub(r'\s+', ' ', text).strip()
        return text

    if args.prompt:
        for pr in args.prompt:
            print(f'>>> {pr}')
            print(generate_once(pr))
            print()
        return

    while True:
        try:
            user = input('>>> ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nBye!')
            break
        if not user:
            continue
        if user.lower() in ('quit', 'exit', 'q'):
            break
        print(generate_once(user))
        print()


if __name__ == '__main__':
    main()
