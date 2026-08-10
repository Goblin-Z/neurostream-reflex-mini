"""
Data pipeline for NeuroStream-Reflex training.

Three data sources:
  1. Wudao 2.0 / SkyPile — large-scale Chinese web corpus (pretraining)
     Download: https://opendatalab.com  → /data/wudao/wudao_10pct.jsonl
  2. Firefly + BELLE — high-quality Chinese SFT data (instruction tuning)
     Download: https://hf-mirror.com/YangyiYin/Firefly  → /data/firefly/firefly_1.6m.jsonl
     Download: https://hf-mirror.com/BelleGroup  → 合并到 SFT 数据
  3. Self-Instruct / Evol-Instruct — synthetic data from Teacher LLM (distillation)
     Generated locally by train_reflex.py, no download needed.

All data is streamed from disk to avoid GPU memory pressure.
"""
import os, json, random, math, gzip
from typing import List, Dict, Optional, Iterator, Tuple
from dataclasses import dataclass

import torch
from torch.utils.data import IterableDataset, DataLoader


# ── Config ──────────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    max_seq_len: int = 2048
    pretrain_ratio: float = 0.7       # 70% web corpus
    sft_ratio: float = 0.2            # 20% instruction data
    distill_ratio: float = 0.1        # 10% synthetic data
    shuffle_buffer: int = 10000       # shuffle buffer size
    num_workers: int = 4


# ── Data source readers ────────────────────────────────────────────────

class JsonlReader:
    """Reads JSONL files, yields dicts. Supports .gz files.

    Streams line-by-line with a bounded shuffle buffer (memory-safe for
    large files, e.g. multi-hundred-GB distillation outputs).
    """

    def __init__(self, paths: List[str], shuffle: bool = True,
                 shuffle_buffer: int = 10000):
        self.paths = paths
        self.shuffle = shuffle
        self.shuffle_buffer = shuffle_buffer

    def __iter__(self):
        buffer = []
        rng = random.Random()
        for path in self.paths:
            open_fn = gzip.open if path.endswith('.gz') else open
            with open_fn(path, 'rt', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if self.shuffle:
                        buffer.append(item)
                        if len(buffer) >= self.shuffle_buffer:
                            rng.shuffle(buffer)
                            yield buffer.pop()
                    else:
                        yield item
        if self.shuffle:
            rng.shuffle(buffer)
        yield from buffer


class WudaoReader:
    """Reads Wudao / SkyPile web corpus. Yields raw text chunks."""

    def __init__(self, path: str, chunk_size: int = 512):
        self.path = path
        self.chunk_size = chunk_size

    def __iter__(self):
        open_fn = gzip.open if self.path.endswith('.gz') else open
        with open_fn(self.path, 'rt', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = item.get('text', '') or item.get('content', '') or ''
                # Yield chunks matched to token limit (no waste)
                char_chunk = self.chunk_size * 3  # ~3 chars per Chinese token
                char_step = char_chunk            # no overlap
                pos = 0
                while pos < len(text):
                    yield text[pos:pos + char_chunk]
                    pos += char_step


# ── Tokenized dataset ──────────────────────────────────────────────────

class StreamingDataset(IterableDataset):
    """
    Iterable dataset that streams from disk, tokenizes on-the-fly.
    Supports mixing multiple data sources with configurable ratios.
    """

    def __init__(self, tokenizer, config: DataConfig,
                 pretrain_paths: List[str] = None,
                 sft_paths: List[str] = None,
                 distill_paths: List[str] = None):
        self.tokenizer = tokenizer
        self.config = config
        self.pretrain_paths = pretrain_paths or []
        self.sft_paths = sft_paths or []
        self.distill_paths = distill_paths or []

    def _tokenize(self, text: str) -> torch.Tensor:
        ids = self.tokenizer(
            text, max_length=self.config.max_seq_len,
            truncation=True, return_tensors='pt',
        )['input_ids'][0]
        return ids

    def _pretrain_iter(self) -> Iterator[Dict]:
        """Yield pretrain chunks as (input_ids, labels) for CLM."""
        for path in self.pretrain_paths:
            reader = WudaoReader(path, chunk_size=self.config.max_seq_len)
            for chunk in reader:
                ids = self._tokenize(chunk)
                if ids.size(0) < 32:
                    continue
                # CLM: label[t] = input[t+1], 最后一个 token 无 next -> -100
                labels = ids.clone()
                labels[:-1] = ids[1:]
                labels[-1] = -100
                yield {
                    'input_ids': ids,
                    'labels': labels,
                }

    def _sft_iter(self) -> Iterator[Dict]:
        """Yield SFT samples with instruction masking (supports multi-turn)."""
        for path in self.sft_paths:
            reader = JsonlReader([path])
            for item in reader:
                # ── 提取轮次 ──
                turns = []  # list of (role, text), role in {'user','assistant'}
                if 'conversations' in item:
                    for t in item['conversations']:
                        frm = t.get('from', '')
                        val = (t.get('value', '') or '').strip()
                        if not val:
                            continue
                        if frm == 'human':
                            turns.append(('user', val))
                        elif frm in ('assistant', 'gpt'):
                            turns.append(('assistant', val))
                else:
                    inst = item.get('instruction', '') or ''
                    extra = item.get('input', '') or ''
                    if extra:
                        inst = inst + '\n' + extra if inst else extra
                    resp = item.get('output', '') or item.get('response', '') \
                        or item.get('target', '') or ''
                    if inst.strip():
                        turns.append(('user', inst.strip()))
                    if resp.strip():
                        turns.append(('assistant', resp.strip()))

                # 至少一轮 user + 一轮 assistant
                if len(turns) < 2 or turns[-1][0] != 'assistant':
                    continue

                # 截断到 max_seq_len 能容纳的轮数（保首保尾）
                if getattr(self.tokenizer, 'chat_template', None):
                    yield from self._sft_iter_template(turns)
                else:
                    # Fallback: 无 chat_template 用裸拼接（只用最后一轮）
                    inst = turns[-2][1]
                    resp = turns[-1][1]
                    inst_ids = self.tokenizer(
                        inst, add_special_tokens=False,
                        max_length=self.config.max_seq_len // 2,
                        truncation=True)['input_ids']
                    resp_ids = self.tokenizer(
                        resp, add_special_tokens=False,
                        max_length=self.config.max_seq_len // 2,
                        truncation=True)['input_ids']
                    eos = [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id else []
                    input_ids = torch.tensor(inst_ids + resp_ids + eos, dtype=torch.long)
                    labels = torch.full_like(input_ids, -100)
                    mask_end = len(inst_ids)
                    if input_ids.size(0) > mask_end:
                        labels[mask_end - 1:-1] = input_ids[mask_end:]
                    yield {'input_ids': input_ids, 'labels': labels}

    def _sft_iter_template(self, turns):
        """Multi-turn via Qwen chat template. Supervise ALL assistant turns."""
        tok = self.tokenizer
        max_len = self.config.max_seq_len

        def _build(turn_list):
            return tok.apply_chat_template(turn_list, tokenize=False,
                                           add_generation_prompt=False)

        def _encode(s):
            return tok(s, add_special_tokens=False,
                       max_length=max_len, truncation=True)['input_ids']

        # 渐进构造：每次加一轮，记录每个 assistant 内容起止（用于 mask）
        msgs = []
        full_ids = []
        # system 前缀（apply_chat_template 会自动加；这里手动拼以精确控制边界）
        sys_str = tok.apply_chat_template([{'role': 'system',
                                            'content': 'You are a helpful assistant.'}],
                                          tokenize=False) if tok.chat_template else ''
        if sys_str:
            full_ids = _encode(sys_str)
        else:
            full_ids = []

        boundaries = []  # (start, end) of each assistant CONTENT in full_ids
        for role, text in turns:
            msgs.append({'role': role, 'content': text})
            new_str = _build(msgs)
            new_ids = _encode(new_str)
            if len(new_ids) > max_len:
                break
            if role == 'assistant':
                # 新加的 assistant 内容 = 新串尾部（去掉结束标记 <|im_end|>/eos）
                # 简单起见：从上一轮末尾到本串末尾之间都是本轮的 assistant 内容
                if len(full_ids) < len(new_ids):
                    start = len(full_ids)
                    end = len(new_ids)
                    boundaries.append((start, end))
            full_ids = new_ids

        if not boundaries:
            return
        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels = torch.full_like(input_ids, -100)
        # labels[t] = input_ids[t+1]；对每个 assistant 内容区间监督（不含最后一个 token 的 next）
        for s, e in boundaries:
            if e > s:
                # 区间内预测下一个 token
                labels[s:e - 1] = input_ids[s + 1:e]
        # 截断保护
        if input_ids.size(0) > max_len:
            input_ids = input_ids[:max_len]
            labels = labels[:max_len]
            labels[-1] = -100
        yield {'input_ids': input_ids, 'labels': labels}

    def _distill_iter(self) -> Iterator[Dict]:
        """Yield distillation samples (top-k sparse teacher log-probs)."""
        for path in self.distill_paths:
            # Teacher samples are ~1MB each (top-256 x T); a small bounded
            # shuffle buffer keeps per-worker RAM low (each DataLoader worker
            # owns its own JsonlReader instance).
            reader = JsonlReader([path],
                                 shuffle_buffer=max(256, min(1024, self.config.shuffle_buffer)))
            for item in reader:
                input_ids = torch.tensor(item['input_ids'], dtype=torch.long)
                labels = torch.tensor(item['labels'], dtype=torch.long)
                tk_idx = torch.tensor(item['teacher_topk_indices'], dtype=torch.long)
                tk_logp = torch.tensor(item['teacher_topk_logp'], dtype=torch.float16)
                if input_ids.size(0) > self.config.max_seq_len:
                    input_ids = input_ids[:self.config.max_seq_len]
                    labels = labels[:self.config.max_seq_len]
                    labels[-1] = -100  # 截断点无 next token
                    tk_idx = tk_idx[:self.config.max_seq_len]
                    tk_logp = tk_logp[:self.config.max_seq_len]
                yield {
                    'input_ids': input_ids,
                    'labels': labels,
                    'teacher_topk_indices': tk_idx,
                    'teacher_topk_logp': tk_logp,
                }

    def __len__(self):
        """D1 fix: IterableDataset 无真实长度；返回近似大数，
        使 DistributedSampler 可构造（DDP 分支），epoch 轮换对
        流式数据本身无意义但无害。"""
        return 1 << 30

    def __iter__(self) -> Iterator[Dict]:
        """Interleave data sources according to configured ratios."""
        iters = []
        ratios = []
        if self.pretrain_paths:
            iters.append(self._pretrain_iter())
            ratios.append(self.config.pretrain_ratio)
        if self.sft_paths:
            iters.append(self._sft_iter())
            ratios.append(self.config.sft_ratio)
        if self.distill_paths:
            iters.append(self._distill_iter())
            ratios.append(self.config.distill_ratio)

        if not iters:
            return

        # Round-robin with weighted probabilities
        total = sum(ratios)
        probs = [r / total for r in ratios]
        buffers = [[] for _ in iters]
        buffer_sizes = [self.config.shuffle_buffer // len(iters)] * len(iters)

        while True:
            # Pick a source
            src_idx = random.choices(range(len(iters)), weights=probs, k=1)[0]
            # Refill buffer if empty
            if not buffers[src_idx]:
                try:
                    for _ in range(buffer_sizes[src_idx]):
                        buffers[src_idx].append(next(iters[src_idx]))
                except StopIteration:
                    buffers[src_idx] = buffers[src_idx] or None
                    if all(not b for b in buffers):
                        return
                    continue
                random.shuffle(buffers[src_idx])
            yield buffers[src_idx].pop()


# ── Collation ──────────────────────────────────────────────────────────

_PAD_ID = 0


def set_pad_id(tokenizer):
    """Set the pad token id from the tokenizer. Call once before training."""
    global _PAD_ID
    _PAD_ID = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0


def collate_fn(batch: List[Dict]) -> Dict:
    """Pad sequences to max length in batch."""
    max_len = max(x['input_ids'].size(0) for x in batch)
    pad_id = _PAD_ID
    input_ids, labels = [], []
    has_teacher = 'teacher_topk_indices' in batch[0]
    k_size = None

    for x in batch:
        pad = max_len - x['input_ids'].size(0)
        input_ids.append(torch.cat([x['input_ids'], torch.full((pad,), pad_id, dtype=torch.long)]))
        labels.append(torch.cat([x['labels'], torch.full((pad,), -100, dtype=torch.long)]))
        if has_teacher:
            tk_i = x.get('teacher_topk_indices', None)
            tk_lp = x.get('teacher_topk_logp', None)
            if tk_i is None or tk_lp is None:
                # Missing teacher data: zero-fill (logp=0 -> prob=1, degenerate;
                # masked by labels anyway)
                if k_size is None:
                    for y in batch:
                        if 'teacher_topk_indices' in y:
                            k_size = y['teacher_topk_indices'].size(-1)
                            break
                    k_size = k_size or 1
                tk_i = torch.zeros(x['input_ids'].size(0), k_size, dtype=torch.long)
                tk_lp = torch.zeros(x['input_ids'].size(0), k_size, dtype=torch.float16)
            else:
                k_size = tk_i.size(-1)
            x['teacher_topk_indices'] = torch.cat([
                tk_i, torch.zeros(pad, k_size, dtype=tk_i.dtype)
            ], dim=0)
            x['teacher_topk_logp'] = torch.cat([
                tk_lp, torch.zeros(pad, k_size, dtype=tk_lp.dtype)
            ], dim=0)

    result = {
        'input_ids': torch.stack(input_ids),
        'labels': torch.stack(labels),
    }
    if has_teacher:
        result['teacher_topk_indices'] = torch.stack(
            [x['teacher_topk_indices'] for x in batch]
        )
        result['teacher_topk_logp'] = torch.stack(
            [x['teacher_topk_logp'] for x in batch]
        )
    return result


# ── Teacher logits generator (offline) ─────────────────────────────────

@torch.no_grad()
def generate_teacher_logits(
    teacher_model,
    tokenizer,
    input_path: str,
    output_path: str,
    batch_size: int = 4,
    max_seq_len: int = 2048,
    device: str = 'cuda',
    max_samples: Optional[int] = None,
    temperature: float = 2.0,
):
    """
    Generate teacher logits for distillation data (top-k sparse storage).
    Run this ONCE before training to avoid holding teacher in GPU memory.

    Full vocab=151936 logits per sample would be ~1.2GB in JSON (infeasible
    for 1M+ samples).  Only top-256 log-PROBABILITIES are saved:
      - teacher_logp_k = log_softmax(full_logits)[k] for top-k tokens
      - p_rest = 1 - sum(exp(logp_k))  (probability mass outside top-k)
    This enables an exact (up to teacher-entropy constant) sparse CE for KD.

    max_samples: when set, the input corpus is first uniformly sub-sampled
    via streaming reservoir sampling (O(max_samples) memory, no full-file
    shuffle), so a limited sample is representative of the WHOLE corpus
    rather than just its head (sft_all.jsonl is source-ordered, e.g.
    Firefly -> BELLE -> ShareGPT -> COIG).
    """
    TOP_K = 64
    teacher_model.to(device).eval()

    # ── Stage 1: stream the corpus once, reservoir-sample if capped ──
    items = []
    rng = random.Random(0)
    n_seen = 0
    for item in JsonlReader([input_path]):
        inst = item.get('instruction', '') or item.get('input', '') or ''
        resp = item.get('output', '') or item.get('response', '') or ''
        if not inst.strip() or not resp.strip():
            continue
        if max_samples is None:
            items.append(item)
        else:
            n_seen += 1
            if len(items) < max_samples:
                items.append(item)
            else:
                j = rng.randint(0, n_seen - 1)
                if j < max_samples:
                    items[j] = item
    if max_samples is not None:
        print(f"  [INFO] Reservoir-sampled {len(items)}/{n_seen} samples "
              f"(uniform over whole corpus)")

    batch_ids, batch_labels = [], []
    n_written = 0

    # gzip output: the JSON text of per-token top-k lists is large
    # (~1-2.5MB/sample uncompressed); compression cuts disk usage ~2.5x.
    # JsonlReader transparently reads .gz files.
    # NOTE: the caller may pass a '.partial' atomic-write path like
    # 'foo.jsonl.gz.partial'; accept both so gzip still applies.
    open_fn = gzip.open if (output_path.endswith('.gz')
                            or output_path.endswith('.gz.partial')) else open
    out_file = open_fn(output_path, 'wt', encoding='utf-8')

    def _write_item(out, item):
        out.write(json.dumps(item, ensure_ascii=False) + '\n')

    with out_file as f:
        for item in items:
            inst = item.get('instruction', '') or item.get('input', '') or ''
            resp = item.get('output', '') or item.get('response', '') or ''
            inst_ids = tokenizer(
                inst, add_special_tokens=False,
                max_length=max_seq_len // 2, truncation=True,
            )['input_ids']
            resp_ids = tokenizer(
                resp, add_special_tokens=False,
                max_length=max_seq_len // 2, truncation=True,
            )['input_ids']
            if getattr(tokenizer, 'chat_template', None):
                # Qwen chat template: teacher 在正确的对话输入下产生目标分布，
                # 否则 KD 蒸馏的是"无模板续写"而非"对话格式"（BUG_AUDIT C7）
                inst = tokenizer.decode(inst_ids)
                resp = tokenizer.decode(resp_ids)
                prompt_str = tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': inst}],
                    tokenize=False, add_generation_prompt=True)
                full_str = prompt_str + resp + (tokenizer.eos_token or '')
                input_ids = torch.tensor(
                    tokenizer(full_str, add_special_tokens=False,
                              max_length=max_seq_len, truncation=True)['input_ids'],
                    dtype=torch.long)
                prompt_ids = torch.tensor(
                    tokenizer(prompt_str, add_special_tokens=False)['input_ids'],
                    dtype=torch.long)
                mask_end = prompt_ids.size(0)
            else:
                eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id else []
                input_ids = torch.tensor(inst_ids + resp_ids + eos, dtype=torch.long)
                mask_end = len(inst_ids)
            # CLM shift: labels[t] = input_ids[t+1] (只监督回复部分)
            labels = torch.full_like(input_ids, -100)
            if input_ids.size(0) > mask_end:
                labels[mask_end - 1:-1] = input_ids[mask_end:]
            if input_ids.size(0) > max_seq_len:
                input_ids = input_ids[:max_seq_len]
                labels = labels[:max_seq_len]
                labels[-1] = -100  # 截断点无 next token
            batch_ids.append(input_ids)
            batch_labels.append(labels)

            if len(batch_ids) == batch_size:
                ids, lbls = _pad_batch(batch_ids, batch_labels, tokenizer.pad_token_id or 0)
                ids, lbls = ids.to(device), lbls.to(device)
                logits = teacher_model(ids).logits.detach().cpu().to(torch.float32)
                log_probs = torch.log_softmax(logits / temperature, dim=-1).to(torch.float16)
                for i in range(batch_size):
                    T = len(batch_ids[i])
                    tk = min(TOP_K, logits.size(-1))
                    topk_logp, topk_i = torch.topk(log_probs[i, :T], k=tk, dim=-1)
                    # round log-probs to 4 decimals: shrinks JSON text ~30%
                    # (p_rest covers the mass outside top-K, so 4-decimal
                    # precision is far below the KD signal noise floor)
                    logp_l = [[round(float(v), 4) for v in row]
                              for row in topk_logp.tolist()]
                    out = {
                        'input_ids': batch_ids[i].tolist(),
                        'labels': batch_labels[i].tolist(),
                        'teacher_topk_indices': topk_i.tolist(),   # [T, K]
                        'teacher_topk_logp': logp_l,               # [T, K] full-dist log-probs
                    }
                    _write_item(f, out)
                batch_ids, batch_labels = [], []
                n_written += batch_size

        # Flush remaining
        if batch_ids:
            ids, lbls = _pad_batch(batch_ids, batch_labels, tokenizer.pad_token_id or 0)
            ids, lbls = ids.to(device), lbls.to(device)
            logits = teacher_model(ids).logits.detach().cpu().to(torch.float32)
            log_probs = torch.log_softmax(logits / temperature, dim=-1).to(torch.float16)
            for i in range(len(batch_ids)):
                T = len(batch_ids[i])
                tk = min(TOP_K, logits.size(-1))
                topk_logp, topk_i = torch.topk(log_probs[i, :T], k=tk, dim=-1)
                logp_l = [[round(float(v), 4) for v in row]
                          for row in topk_logp.tolist()]
                out = {
                    'input_ids': batch_ids[i].tolist(),
                    'labels': batch_labels[i].tolist(),
                    'teacher_topk_indices': topk_i.tolist(),
                    'teacher_topk_logp': logp_l,
                }
                _write_item(f, out)

    teacher_model.to('cpu')
    print(f"Teacher logits saved to {output_path}")


def _pad_batch(ids_list, lbls_list, pad_id):
    max_len = max(x.size(0) for x in ids_list)
    ids = torch.stack([
        torch.cat([x, torch.full((max_len - x.size(0),), pad_id, dtype=torch.long)])
        for x in ids_list
    ])
    lbls = torch.stack([
        torch.cat([x, torch.full((max_len - x.size(0),), -100, dtype=torch.long)])
        for x in lbls_list
    ])
    return ids, lbls
