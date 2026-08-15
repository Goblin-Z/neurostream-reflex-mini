"""
流式去重 sft_all.jsonl（按 instruction+input+output 三字段哈希）。
115 万行内存占用 ~40MB，1-2 分钟完成。

用法:
  python scripts/dedup_sft.py --input /root/autodl-tmp/data/sft_all.jsonl \
                              --output /root/autodl-tmp/data/sft_all_dedup.jsonl
"""
import sys, os, json, hashlib, argparse


def main():
    ap = argparse.ArgumentParser(description='Dedup SFT jsonl (streaming)')
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    seen = set()
    total = 0
    kept = 0
    with open(args.input, 'r', encoding='utf-8') as f, \
         open(args.output, 'w', encoding='utf-8') as out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            key = (item.get('instruction', ''),
                   item.get('input', ''),
                   item.get('output', ''))
            h = hashlib.md5('|'.join(key).encode('utf-8')).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            kept += 1
            out.write(line + '\n')

    dupes = total - kept
    print(f'total={total} kept={kept} dupes={dupes} '
          f'dup_rate={dupes / total:.1%}' if total else 'empty')


if __name__ == '__main__':
    main()
