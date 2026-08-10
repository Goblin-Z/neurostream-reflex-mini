#!/usr/bin/env python3
"""
init_from_qwen.py - 从 Qwen2.5-0.5B 提取 token embedding 初始化 ReflexMini

Qwen2.5-0.5B embedding: [151936, 512]
ReflexMini needs:        [151936, 640]
策略: 零填充 (前 512 维用 Qwen 权重, 后 128 维用 N(0, 0.02))
"""
import torch
import sys
import os

def main():
    qwen_path = os.environ.get('QWEN_PATH', '/root/autodl-tmp/data/qwen2.5-0.5b')
    output_path = os.environ.get('OUTPUT_PATH', '/root/autodl-tmp/data/reflex_mini_init.pt')

    if not os.path.exists(qwen_path):
        print(f'ERROR: Qwen path not found: {qwen_path}')
        sys.exit(1)

    print(f'Loading Qwen2.5-0.5B from {qwen_path}...')

    # Method 1: load safetensors
    from safetensors.torch import load_file as load_safetensors
    st_files = sorted(glob.glob(f'{qwen_path}/*.safetensors'))
    if st_files:
        qwen_state = {}
        for f in st_files:
            qwen_state.update(load_safetensors(f))
    else:
        qwen_state = torch.load(f'{qwen_path}/pytorch_model.bin', map_location='cpu')

    # Find embedding
    emb_key = None
    for key in qwen_state:
        if 'embed' in key.lower() and 'weight' in key:
            emb_key = key
            break

    if emb_key is None:
        print('ERROR: Could not find embedding weight in Qwen state dict')
        print('Available keys:', [k for k in qwen_state.keys() if 'embed' in k.lower()])
        sys.exit(1)

    qwen_emb = qwen_state[emb_key]
    print(f'  Qwen embedding: {qwen_emb.shape} (vocab={qwen_emb.shape[0]}, d={qwen_emb.shape[1]})')

    vocab, qwen_d = qwen_emb.shape
    target_d = 640  # ReflexMiniConfig.d_model

    if qwen_d < target_d:
        pad = torch.randn(vocab, target_d - qwen_d, dtype=qwen_emb.dtype) * 0.02
        new_emb = torch.cat([qwen_emb, pad], dim=1)
    elif qwen_d > target_d:
        new_emb = qwen_emb[:, :target_d]
    else:
        new_emb = qwen_emb

    print(f'  Adapted embedding: {new_emb.shape}')

    # Build ReflexMini model and load embedding
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.model_config import ReflexMiniConfig
    from core.model import ReflexModel

    config = ReflexMiniConfig()
    model = ReflexModel(config)

    # Load adapted embedding (tied with lm_head)
    model.token_embedding.weight.data.copy_(new_emb)
    print(f'  Loaded embedding into ReflexModel (d_model={config.d_model})')

    # Save checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'step': 0,
        'config': config.__dict__,
    }
    torch.save(checkpoint, output_path)
    print(f'  Saved init checkpoint -> {output_path}')
    print('Done.')


if __name__ == '__main__':
    import glob
    main()
