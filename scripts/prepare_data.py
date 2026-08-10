#!/usr/bin/env python3
"""
prepare_data.py - 将各数据源预处理为统一 JSONL 格式

Pretrain 输出: /root/autodl-tmp/data/pretrain_all.jsonl  {"text": "..."}
SFT 输出:     /root/autodl-tmp/data/sft_all.jsonl        {"instruction":"...", "input":"", "output":"..."}
"""
import json
import glob
import os
import sys
import pandas as pd

DATA = os.environ.get('DATA_DIR', '/root/autodl-tmp/data')


def write_jsonl(path, generator, name):
    count = 0
    with open(path, 'w', encoding='utf-8') as f:
        for item in generator:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            count += 1
            if count % 100000 == 0:
                print(f'  {name}: {count:,} documents...')
    print(f'  {name}: {count:,} documents -> {path}')
    return count


# ── Pretrain ──

def extract_wikipedia():
    for path in sorted(glob.glob(f'{DATA}/wikipedia/wikipedia-zh-cn-*.json')):
        print(f'  Wikipedia: {os.path.basename(path)} ({os.path.getsize(path)/1e9:.1f}GB)')
        # Wikipedia is JSONL: one JSON object per line (not a single array)
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                title = item.get('title', '')
                body = item.get('body', item.get('content', ''))
                text = f'{title}\n\n{body}' if title else body
                if len(text) > 50:
                    yield {'text': text}


def extract_skypile():
    for pattern in [f'{DATA}/skypile/data/2023-14_zh_head_*.jsonl',
                    f'{DATA}/skypile/data/2023-06_zh_head_*.jsonl']:
        for filepath in sorted(glob.glob(pattern)):
            print(f'  SkyPile: {os.path.basename(filepath)}')
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        text = item.get('text', item.get('content', ''))
                        if len(text) > 200:
                            yield {'text': text}
                    except json.JSONDecodeError:
                        continue


def extract_cwt():
    for filepath in sorted(glob.glob(f'{DATA}/cwt-hq/data/*.parquet')):
        print(f'  CWT: {os.path.basename(filepath)}')
        df = pd.read_parquet(filepath)
        col = 'text' if 'text' in df.columns else df.columns[0]
        for text in df[col].dropna():
            if len(str(text)) > 200:
                yield {'text': str(text)}


def extract_fineweb():
    for filepath in sorted(glob.glob(f'{DATA}/fineweb-zh/*.parquet')):
        print(f'  FineWeb: {os.path.basename(filepath)}')
        df = pd.read_parquet(filepath)
        col = 'text' if 'text' in df.columns else df.columns[0]
        for text in df[col].dropna():
            if len(str(text)) > 200:
                yield {'text': str(text)}


def prepare_pretrain():
    print('=== Preparing Pretrain Data ===')
    output = f'{DATA}/pretrain_all.jsonl'
    count = 0
    with open(output, 'w', encoding='utf-8') as f:
        for gen, name in [(extract_wikipedia, 'Wikipedia'),
                          (extract_skypile, 'SkyPile'),
                          (extract_cwt, 'CWT'),
                          (extract_fineweb, 'FineWeb')]:
            for item in gen():
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                count += 1
                if count % 100000 == 0:
                    print(f'  Total: {count:,} documents...')
    print(f'  Pretrain total: {count:,} documents -> {output}')
    return count


# ── SFT ──

def extract_firefly(half=True):
    filepath = f'{DATA}/firefly/firefly-train-1.1M.jsonl'
    if not os.path.exists(filepath):
        for p in glob.glob(f'{DATA}/firefly/**/*.jsonl', recursive=True):
            filepath = p
            break
    count = 0
    limit = 575000 if half else float('inf')  # 115万/2 = 57.5万
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if count >= limit:
                break
            try:
                item = json.loads(line)
                yield {
                    'instruction': item.get('input', ''),
                    'input': '',
                    'output': item.get('target', ''),
                }
                count += 1
            except json.JSONDecodeError:
                continue


def extract_belle(half=True):
    filepath = f'{DATA}/belle/Belle_open_source_1M.json'
    if not os.path.exists(filepath):
        for p in glob.glob(f'{DATA}/belle/**/*.json', recursive=True):
            if 'README' not in p and '.gitattr' not in p:
                filepath = p
                break
    count = 0
    limit = 500000 if half else float('inf')  # 100万/2 = 50万
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if count >= limit:
                break
            try:
                item = json.loads(line)
                inst = item.get('instruction', '')
                inp = item.get('input', '')
                out = item.get('output', '')
                if inst and out:
                    yield {'instruction': inst, 'input': inp, 'output': out}
                    count += 1
            except json.JSONDecodeError:
                continue


def extract_sharegpt():
    for filepath in sorted(glob.glob(f'{DATA}/sharegpt/sharegpt_jsonl/*.jsonl')):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    convs = item.get('conversations', [])
                    if len(convs) >= 2:
                        human = convs[0].get('value', '')
                        gpt = convs[1].get('value', '')
                        if human and gpt:
                            yield {'instruction': human, 'input': '', 'output': gpt}
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue


def extract_coig():
    for filepath in sorted(glob.glob(f'{DATA}/coig/**/*.json*', recursive=True)):
        if 'README' in filepath or '.gitattr' in filepath or '.py' in filepath:
            continue
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                if filepath.endswith('.jsonl'):
                    for line in f:
                        item = json.loads(line)
                        if 'instruction' in item:
                            yield {
                                'instruction': item['instruction'],
                                'input': item.get('input', ''),
                                'output': item.get('output', ''),
                            }
                else:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if 'instruction' in item:
                                yield {
                                    'instruction': item['instruction'],
                                    'input': item.get('input', ''),
                                    'output': item.get('output', ''),
                                }
        except (json.JSONDecodeError, Exception):
            continue


def prepare_sft():
    print('=== Preparing SFT Data ===')
    output = f'{DATA}/sft_all.jsonl'
    count = 0
    with open(output, 'w', encoding='utf-8') as f:
        for gen, name in [(extract_firefly, 'Firefly(50%)'),
                          (extract_belle, 'BELLE(50%)'),
                          (extract_sharegpt, 'ShareGPT'),
                          (extract_coig, 'COIG')]:
            for item in gen():
                if item['instruction'] and item['output']:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    count += 1
                    if count % 50000 == 0:
                        print(f'  Total: {count:,} examples...')
    print(f'  SFT total: {count:,} examples -> {output}')
    return count


if __name__ == '__main__':
    pretrain_out = f'{DATA}/pretrain_all.jsonl'
    sft_out = f'{DATA}/sft_all.jsonl'
    pretrain_done = os.path.isfile(pretrain_out) and os.path.getsize(pretrain_out) > 1000000
    sft_done = os.path.isfile(sft_out) and os.path.getsize(sft_out) > 1000000

    if not pretrain_done:
        prepare_pretrain()
    else:
        print(f'=== Pretrain data exists ({os.path.getsize(pretrain_out)/1e9:.1f}GB), skipping ===')

    if not sft_done:
        print()
        prepare_sft()
    else:
        print(f'=== SFT data exists ({os.path.getsize(sft_out)/1e9:.1f}GB), skipping ===')
    print('\n=== Data Preparation Complete ===')
