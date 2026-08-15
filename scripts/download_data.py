#!/usr/bin/env python3
"""
download_data.py - 从 HF-mirror 下载全部训练数据
统一使用直接 HTTP 下载，绕过 Xet/CAS 存储和 hugggingface_hub 库的兼容性问题。
"""
import os
import sys
import requests
from tqdm import tqdm

MIRROR = 'https://hf-mirror.com'
DATA_DIR = '/root/autodl-tmp/data'


def download_file(url, out_path, desc=''):
    """Download a single file with retry and progress bar."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1e6:
        return  # already exists
    for attempt in range(3):
        try:
            r = requests.get(url, stream=True, timeout=120)
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))
            with open(out_path, 'wb') as f:
                for chunk in tqdm(r.iter_content(chunk_size=8192*8),
                                  total=total//8192//8,
                                  desc=desc or os.path.basename(out_path),
                                  unit='chunk', leave=False):
                    f.write(chunk)
            return
        except requests.RequestException as e:
            if attempt == 2:
                print(f'  失败: {url} - {e}')


# ════════════════════════════════════════════
# 1. Wikipedia (6 个 JSON 文件, ~22.9 GB)
# ════════════════════════════════════════════
def download_wikipedia():
    print('[1/4] Wikipedia 全量快照...')
    files = [
        'wikipedia-zh-cn-20240901.json',
        'wikipedia-zh-cn-20241020.json',
        'wikipedia-zh-cn-20250320.json',
        'wikipedia-zh-cn-20250901.json',
        'wikipedia-zh-cn-20260201.json',
        'wikipedia-zh-cn-20260501.json',
    ]
    dst = f'{DATA_DIR}/wikipedia'
    os.makedirs(dst, exist_ok=True)
    for fname in files:
        url = f'{MIRROR}/datasets/fjcanyue/wikipedia-zh-cn/resolve/main/{fname}'
        out = f'{dst}/{fname}'
        download_file(url, out, desc=f'Wiki {fname[:20]}..')
    print(f'  Wikipedia 完成 -> {dst} ({len(files)} files)')


# ════════════════════════════════════════════
# 2. SkyPile (18 个 JSONL 文件, ~26.4 GB)
# ════════════════════════════════════════════
def download_skypile():
    print('[2/4] SkyPile 最后批次 head...')
    base = f'{MIRROR}/datasets/Skywork/SkyPile-150B/resolve/main/data'
    dst = f'{DATA_DIR}/skypile/data'
    os.makedirs(dst, exist_ok=True)
    # 2023-14: 8 files, 2023-06: 10 files
    files = [f'2023-14_zh_head_{i:04d}.jsonl' for i in range(8)]
    files += [f'2023-06_zh_head_{i:04d}.jsonl' for i in range(10)]
    for fname in files:
        url = f'{base}/{fname}'
        out = f'{dst}/{fname}'
        download_file(url, out, desc=f'SkyPile {fname[:22]}..')
    print(f'  SkyPile 完成 -> {dst} ({len(files)} files)')


# ════════════════════════════════════════════
# 3. ChineseWebText2.0-HQ (30 parquet, ~5.4 GB)
# ════════════════════════════════════════════
def download_cwt():
    print('[3/4] ChineseWebText2.0-HQ (30 shards)...')
    base = f'{MIRROR}/datasets/Morton-Li/ChineseWebText2.0-HighQuality/resolve/main/data'
    dst = f'{DATA_DIR}/cwt-hq/data'
    os.makedirs(dst, exist_ok=True)
    for i in range(30):
        fname = f'CASIA-LM_ChineseWebText2.0_partial-{i:06d}.parquet'
        url = f'{base}/{fname}'
        out = f'{dst}/{fname}'
        download_file(url, out, desc=f'CWT {fname[:30]}..')
    print(f'  CWT 完成 -> {dst} (30 files)')


# ════════════════════════════════════════════
# 4. FineWeb-2-zh (5 parquet, ~7.1 GB)
# ════════════════════════════════════════════
def download_fineweb():
    print('[4/4] FineWeb-2-zh (5 files)...')
    base = f'{MIRROR}/datasets/TiWu-Lab/fineweb-2-zh/resolve/main'
    dst = f'{DATA_DIR}/fineweb-zh'
    os.makedirs(dst, exist_ok=True)
    for i in range(5):
        fname = f'000_{i:05d}.parquet'
        url = f'{base}/{fname}'
        out = f'{dst}/{fname}'
        download_file(url, out, desc=f'FineWeb {fname}')
    print(f'  FineWeb 完成 -> {dst} (5 files)')


# ════════════════════════════════════════════
# 5. SFT 数据
# ════════════════════════════════════════════
def download_sft():
    # 删除旧版 snapshot_download 留下的残片
    import shutil
    for d in ['firefly', 'belle', 'coig']:
        p = f'{DATA_DIR}/{d}'
        if os.path.isdir(p):
            sz = sum(os.path.getsize(os.path.join(dp,f)) for dp,_,fn in os.walk(p) for f in fn) if any(os.listdir(p)) else 0
            if sz < 500_000_000:  # <500MB = 不完整
                shutil.rmtree(p)
                print(f'  [清理] 删除不完整目录: {d} ({sz/1e6:.0f}MB)')

    print('[SFT] Firefly 1.1M...')
    url = f'{MIRROR}/datasets/YeungNLP/firefly-train-1.1M/resolve/main/firefly-train-1.1M.jsonl'
    download_file(url, f'{DATA_DIR}/firefly/firefly-train-1.1M.jsonl', desc='Firefly')

    print('[SFT] BELLE 1M...')
    url = f'{MIRROR}/datasets/BelleGroup/train_1M_CN/resolve/main/Belle_open_source_1M.json'
    download_file(url, f'{DATA_DIR}/belle/Belle_open_source_1M.json', desc='BELLE')

    print('[SFT] ShareGPT...')
    sharegpt_files = [
        'common_en_70k.jsonl', 'common_zh_70k.jsonl',
        'computer_en_26k.jsonl', 'computer_zh_26k.jsonl',
        'computer_cn_26k_continue.jsonl', 'unknow_zh_38k.jsonl',
        'computer_en_26k(fixed).jsonl', 'computer_zh_26k(fixed).jsonl',
        'computer_en_26k_continue.jsonl', 'unknow_zh_38k_continue.jsonl',
    ]
    for fname in sharegpt_files:
        url = f'{MIRROR}/datasets/shareAI/ShareGPT-Chinese-English-90k/resolve/main/sharegpt_jsonl/{fname}'
        dst = f'{DATA_DIR}/sharegpt/sharegpt_jsonl/{fname}'
        download_file(url, dst, desc=f'ShareGPT {fname[:24]}..')

    print('[SFT] COIG...')
    coig_files = [
        'translated_instructions.jsonl', 'exam_instructions.jsonl',
        'human_value_alignment_instructions_part1.json',
        'human_value_alignment_instructions_part2.json',
        'leetcode_instructions.jsonl',
    ]
    for fname in coig_files:
        url = f'{MIRROR}/datasets/BAAI/COIG/resolve/main/{fname}'
        dst = f'{DATA_DIR}/coig/{fname}'
        download_file(url, dst, desc=f'COIG {fname[:24]}..')


# ════════════════════════════════════════════
# 6. 模型权重
# ════════════════════════════════════════════
def download_models():
    print('[Models] Qwen2.5-1.5B-Instruct...')
    base = f'{MIRROR}/Qwen/Qwen2.5-1.5B-Instruct/resolve/main'
    dst = f'{DATA_DIR}/qwen2.5-1.5b-instruct'
    os.makedirs(dst, exist_ok=True)
    # 逐个下载模型文件
    model_files = [
        'config.json', 'generation_config.json',
        'merges.txt', 'vocab.json',
        'tokenizer.json', 'tokenizer_config.json',
    ]
    for fname in model_files:
        download_file(f'{base}/{fname}', f'{dst}/{fname}', desc=f'1.5B {fname}')
    # 模型权重文件 (可能是 shard 或单个)
    download_file(f'{base}/model.safetensors', f'{dst}/model.safetensors', desc='1.5B weights')
    # 如果没有单个 model.safetensors, 尝试分片
    import glob
    if not os.path.exists(f'{dst}/model.safetensors'):
        print('  (没有单文件, 尝试下载分片...)')
        for i in range(10):
            fname = f'model-{i+1:05d}-of-00010.safetensors'
            download_file(f'{base}/{fname}', f'{dst}/{fname}', desc=f'1.5B shard {i+1}')

    print('[Models] Qwen2.5-0.5B...')
    base = f'{MIRROR}/Qwen/Qwen2.5-0.5B/resolve/main'
    dst = f'{DATA_DIR}/qwen2.5-0.5b'
    os.makedirs(dst, exist_ok=True)
    model_files = [
        'config.json', 'generation_config.json',
        'merges.txt', 'vocab.json',
        'tokenizer.json', 'tokenizer_config.json',
    ]
    for fname in model_files:
        download_file(f'{base}/{fname}', f'{dst}/{fname}', desc=f'0.5B {fname}')
    download_file(f'{base}/model.safetensors', f'{dst}/model.safetensors', desc='0.5B weights')


# ════════════════════════════════════════════
# Main
# ════════════════════════════════════════════
if __name__ == '__main__':
    os.makedirs(DATA_DIR, exist_ok=True)

    download_wikipedia()
    print()
    download_skypile()
    print()
    download_cwt()
    print()
    download_fineweb()
    print()
    download_sft()
    print()
    download_models()
    print()

    print('='*60)
    print('下载完成!')
    print('='*60)
    os.system(f'du -sh {DATA_DIR}/*/')
