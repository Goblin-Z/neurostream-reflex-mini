#!/bin/bash
# ============================================================
# NeuroStream-Reflex 数据准备脚本 v3 (AutoDL)
# 用 ModelScope REST API 直接下载，绕过 library 兼容性问题
# ============================================================
set -e
cd /root

echo "=== Step 1: 清理 ==="
rm -rf /root/autodl-tmp/skypile /root/autodl-tmp/.cache 2>/dev/null
rm -rf /data/wudao /data/firefly /data/sft /data/skypile /tmp/skypile_json 2>/dev/null
rm -f /data/wudao/wudao_10pct.jsonl /data/firefly/firefly_1.6m.jsonl 2>/dev/null
mkdir -p /root/autodl-tmp/skypile /data/wudao /data/firefly /tmp/skypile_json

echo "=== Step 2: 下载 BELLE SFT 数据 ==="
export HF_ENDPOINT=https://hf-mirror.com
python3 << 'PYEOF'
from datasets import load_dataset
ds = load_dataset('BelleGroup/train_3.5M_CN', split='train')
ds.to_json('/root/autodl-tmp/sft_data.jsonl')
print(f'BELLE: {len(ds)} samples')
PYEOF
ln -sf /root/autodl-tmp/sft_data.jsonl /data/firefly/firefly_1.6m.jsonl

echo "=== Step 3: 从 ModelScope 下载预训练数据 ==="
# 需要设置你的 ModelScope Token
if [ -z "$MODELSCOPE_API_TOKEN" ]; then
    echo "请先设置 MODELSCOPE_API_TOKEN："
    echo "  export MODELSCOPE_API_TOKEN=你的token"
    exit 1
fi

python3 << 'PYEOF'
import requests, json, os, time

token = os.environ.get('MODELSCOPE_API_TOKEN', '')
headers = {'Authorization': token}

# 1. 获取数据集文件树
print("Fetching file tree...")
url = 'https://modelscope.cn/api/v1/datasets/modelscope/SkyPile-150B/repo?Revision=master'
r = requests.get(url, headers=headers, timeout=30)
if not r.ok:
    print(f"Failed to get file tree: {r.status_code} {r.text[:200]}")
    exit(1)

data = r.json()
files = data.get('Files', data.get('files', []))
if not files:
    # Try different response format
    if isinstance(data, dict):
        for key in ['tree', 'entries', 'data', 'result']:
            if key in data:
                files = data[key]
                break

print(f"Found {len(files)} files/dirs")

# 2. 找出 train shard 文件
train_files = []
for f in files:
    name = f.get('Name', f.get('name', ''))
    path = f.get('Path', f.get('path', ''))
    if ('train' in name or 'train' in path) and name.endswith('.jsonl'):
        train_files.append(f)

if not train_files:
    # Maybe files are nested in a data/ directory
    for f in files:
        name = f.get('Name', f.get('name', ''))
        if name == 'data':
            print("Found data directory, fetching children...")
            # Fetch data directory
            url2 = 'https://modelscope.cn/api/v1/datasets/modelscope/SkyPile-150B/repo?Revision=master&Path=data'
            r2 = requests.get(url2, headers=headers, timeout=30)
            if r2.ok:
                data2 = r2.json()
                subfiles = data2.get('Files', data2.get('files', []))
                if not subfiles and isinstance(data2, dict):
                    for key in ['tree', 'entries', 'data', 'result']:
                        if key in data2:
                            subfiles = data2[key]
                            break
                for sf in subfiles:
                    sname = sf.get('Name', sf.get('name', ''))
                    if sname.endswith('.jsonl'):
                        sf['Path'] = sf.get('Path', sf.get('path', f'data/{sname}'))
                        train_files.append(sf)

# Print what we found
if train_files:
    train_files = sorted(train_files, key=lambda x: x.get('Name', x.get('name', '')))
    print(f"Train shards: {len(train_files)}")
    for tf in train_files[:5]:
        print(f"  {tf.get('Name', tf.get('name', ''))}")
else:
    print("No train files found. Printing first 5 files:")
    for f in files[:5]:
        print(json.dumps(f, indent=2)[:300])
    exit(1)

# 3. 获取每个文件的下载 URL 并下载
data_dir = '/root/autodl-tmp/skypile'
target_shards = min(len(train_files), 27)
print(f"Downloading {target_shards} shards...")

for i, tf in enumerate(train_files[:target_shards]):
    fname = tf.get('Name', tf.get('name', ''))
    fpath = tf.get('Path', tf.get('path', f'data/{fname}'))
    
    # Get download URL
    dl_url = f'https://modelscope.cn/api/v1/datasets/modelscope/SkyPile-150B/repo?Revision=master&FilePath={fpath}&Download=true'
    r2 = requests.get(dl_url, headers=headers, timeout=30)
    
    if r2.ok:
        dl_info = r2.json()
        # The download URL might be in different fields
        real_url = None
        if isinstance(dl_info, str):
            real_url = dl_info
        elif isinstance(dl_info, dict):
            for key in ['url', 'download_url', 'Url', 'DownloadUrl', 'data', 'result']:
                if key in dl_info and isinstance(dl_info[key], str) and dl_info[key].startswith('http'):
                    real_url = dl_info[key]
                    break
        
        if real_url:
            print(f"  [{i+1}/{target_shards}] {fname} -> downloading...")
            r3 = requests.get(real_url, stream=True, timeout=300)
            if r3.ok:
                local_path = os.path.join(data_dir, fname)
                with open(local_path, 'wb') as f:
                    for chunk in r3.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                size = os.path.getsize(local_path)
                print(f"    OK: {size/1024/1024:.1f} MB")
            else:
                print(f"    FAILED: {r3.status_code}")
        else:
            print(f"  [{i+1}/{target_shards}] {fname}: no download URL in response")
            print(f"    Response: {json.dumps(dl_info, indent=2)[:200]}")
    else:
        print(f"  [{i+1}/{target_shards}] {fname}: failed to get download URL ({r2.status_code})")

# 4. 合并
print("\nMerging...")
import subprocess
result = subprocess.run(
    f'cat {data_dir}/train_*.jsonl > {data_dir}/skypile_20B.jsonl',
    shell=True, capture_output=True, text=True
)
if os.path.exists(f'{data_dir}/skypile_20B.jsonl'):
    total = os.path.getsize(f'{data_dir}/skypile_20B.jsonl')
    print(f"Merged: {total/1024/1024/1024:.2f} GB ({total} bytes)")
else:
    print(f"Merge failed: {result.stderr}")

PYEOF

ln -sf /root/autodl-tmp/skypile/skypile_20B.jsonl /data/wudao/wudao_10pct.jsonl 2>/dev/null

echo "=== Step 4: 验证 ==="
ls -lh /data/wudao/wudao_10pct.jsonl 2>/dev/null || echo "WARNING: pretrain data missing"
ls -lh /data/firefly/firefly_1.6m.jsonl 2>/dev/null || echo "WARNING: SFT data missing"
echo "=== DONE ==="