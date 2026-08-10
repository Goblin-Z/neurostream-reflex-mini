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
    p.add_argument('--max-new-tokens', type=int, default=80)
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

    print('Building ReflexMini...')
    config = ReflexMiniConfig()
    model = ReflexModel(config)
    ckpt_path = args.checkpoint
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ckpt_path)
    print(f'Loading checkpoint: {ckpt_path}')
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ck.get('model_state_dict', ck)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f'  step={ck.get("step", "?")} phase={ck.get("phase", "?")}')
    print(f'  missing={len(missing)} unexpected={len(unexpected)}')
    model._decode_tokenizer = tok
    model.to(device=device, dtype=dtype if device != 'cpu' else torch.float32)
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
        return tok.decode(out[0], skip_special_tokens=True)

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
