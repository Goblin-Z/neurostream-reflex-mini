"""
生成后 QA/多轮数据全局清洗：
  1. 长度过滤
  2. 高度重复输出检测（字符/词块/整句重复）
  3. 符号/噪声占比
  4. 完全重复样本去重 (q, a) 哈希
  5. 多轮数据基本校验
输出统计报告。

用法:
  python scripts/filter_qa.py \
      --inputs /root/autodl-tmp/data/qa_tpl_20k.jsonl \
               /root/autodl-tmp/data/qa_si_40k.jsonl \
      --output /root/autodl-tmp/data/sft_kd_clean.jsonl
"""
import os, sys, json, re, hashlib, argparse
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_REPEAT_CHAR = re.compile(r'(.)\1{4,}')       # 同一字符重复 5+（"对对对对对"）
_REPEAT_BLOCK = re.compile(r'(.{2,6})\1{2,}')  # 2-6 字块重复 3+（"中国中国中国中国"）
_SENT_SPLIT = re.compile(r'[。！？!?\n]+')


def is_clean_answer(q: str, a: str) -> bool:
    a = a.strip()
    if not (5 <= len(a) <= 500):
        return False
    if _REPEAT_CHAR.search(a):
        return False
    if _REPEAT_BLOCK.search(a):
        return False
    # 整句重复：分句后同一句出现 2+ 次
    sents = [s.strip() for s in _SENT_SPLIT.split(a) if len(s.strip()) >= 4]
    if len(sents) >= 2:
        c = Counter(sents)
        if max(c.values()) >= 2:
            return False
    # 符号/非文字占比
    nonsym = sum(1 for ch in a
                 if ch.isalnum() or '\u4e00' <= ch <= '\u9fff')
    if nonsym / max(1, len(a)) < 0.75:
        return False
    # 复述问题开头
    if len(q) >= 6 and a.startswith(q[:6]):
        return False
    # 空泛回复
    if re.match(r'^(好的|嗯|哦|知道了|是的|对)$', a):
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', nargs='+', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--multiturn-files', nargs='*', default=[],
                    help='多轮格式文件（conversations），只做基本校验')
    args = ap.parse_args()

    seen = set()
    stats = Counter()
    written = 0

    with open(args.output, 'w', encoding='utf-8') as out:
        for path in args.inputs + args.multiturn_files:
            is_mt = path in args.multiturn_files
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        stats['bad_json'] += 1
                        continue

                    if is_mt:
                        convs = item.get('conversations', [])
                        if len(convs) < 2:
                            stats['mt_too_short'] += 1
                            continue
                        # 基本校验：所有轮非空、总长合理
                        total_len = sum(len(v.get('value', ''))
                                        for v in convs)
                        if not (20 <= total_len <= 2000):
                            stats['mt_bad_len'] += 1
                            continue
                        key = hashlib.md5(
                            json.dumps(convs, ensure_ascii=False)
                            .encode('utf-8')).hexdigest()
                        if key in seen:
                            stats['dup'] += 1
                            continue
                        seen.add(key)
                        out.write(line + '\n')
                        written += 1
                        continue

                    # QA 格式
                    q = item.get('instruction', '') or ''
                    a = item.get('output', '') or ''
                    if not is_clean_answer(q, a):
                        if len(a) < 5 or len(a) > 500:
                            stats['bad_len'] += 1
                        elif _REPEAT_CHAR.search(a) or _REPEAT_BLOCK.search(a):
                            stats['repeat'] += 1
                        else:
                            stats['noise'] += 1
                        continue
                    key = hashlib.md5(
                        (q + '\x00' + a).encode('utf-8')).hexdigest()
                    if key in seen:
                        stats['dup'] += 1
                        continue
                    seen.add(key)
                    out.write(line + '\n')
                    written += 1

    print(f'written={written}')
    for k, v in stats.most_common():
        print(f'  dropped_{k}={v}')


if __name__ == '__main__':
    main()
