"""
记忆微调（Memory Fine-tune）——教模型"利用 KV 历史"。

核心：训练时 KV 记忆真实参与——
  1. 历史轮（前 n-1 轮）→ eval forward → 各层 KV 缓存入 MemoryBank
  2. 当前轮 → train forward（带 mem_kv）→ CE 监督当前轮回复
  3. 梯度经记忆注意力路径回传 → 模型学会"何时回忆、回忆什么"

数据：generate_qa.py --mode multi-turn --memory-tune（7 轮 + 长程引用）

用法:
  python train_memory.py --checkpoint sft_kd_150k_final.pt \
      --data /root/autodl-tmp/data/mt_memory_20k.jsonl \
      --steps 3000 --lr 1e-5 --batch-size 4
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from config.model_config import ReflexMiniConfig
from core.model import ReflexModel


def build_conv_inputs(tok, conv, device):
    """把 conversations 拆成 历史轮(前 n-1) 和 当前轮(最后 user+assistant)。

    返回 (hist_ids, cur_ids, reply_start)：
      hist_ids: 历史轮模板（不含最后 assistant）
      cur_ids:  历史 + 最后 user + 最后 assistant（监督最后回复）
      reply_start: cur_ids 中最后 assistant 内容的起点
    """
    msgs = []
    for t in conv:
        role = 'user' if t.get('from') == 'human' else 'assistant'
        msgs.append({'role': role, 'content': t.get('value', '')})

    last_user = msgs[-2]['content'] if len(msgs) >= 2 else ''
    last_assistant = msgs[-1]['content']
    hist_msgs = msgs[:-2]  # 不含最后两轮（最后 user+assistant 是当前轮）

    # 历史轮（含倒数第二轮 user 作为 prompt 尾）
    hist_prompt = tok.apply_chat_template(
        hist_msgs + [{'role': 'user', 'content': last_user}],
        tokenize=False, add_generation_prompt=True)
    hist_ids = tok(hist_prompt, return_tensors='pt')['input_ids'].to(device)

    # 当前轮完整序列（历史 + 最后 user + 最后 assistant）
    cur_str = (hist_prompt + last_assistant + (tok.eos_token or ''))
    cur_ids = tok(cur_str, return_tensors='pt')['input_ids'].to(device)
    reply_start = hist_ids.size(1)
    return hist_ids, cur_ids, reply_start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--tokenizer', default='/root/autodl-tmp/data/qwen2.5-0.5b')
    ap.add_argument('--output-dir', default='/root/autodl-tmp/checkpoints/reflex-mini')
    ap.add_argument('--log-every', type=int, default=50)
    ap.add_argument('--save-every', type=int, default=500)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Loading checkpoint: {args.checkpoint}')
    config = ReflexMiniConfig()
    model = ReflexModel(config)
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model_state_dict'], strict=False)
    model.to(device)

    # AttnRes post_norm 放大（记忆 source 真实影响输出）
    pn = getattr(config, 'attnres_postnorm_init', 0.1)
    for m in model.attn_res.modules_list:
        m.post_norm.weight.data.fill_(pn)

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))

    print(f'Data: {args.data}, steps: {args.steps}, lr: {args.lr}, '
          f'batch: {args.batch_size}')
    print('开始记忆微调（KV 历史真实参与训练）...')

    step = 0
    total_loss = 0.0
    mb = model.memory_bank
    while step < args.steps:
        with open(args.data, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                conv = item.get('conversations')
                if not conv or len(conv) < 6:
                    continue  # 需要 ≥6 轮（长程引用训练）

                try:
                    hist_ids, cur_ids, reply_start = build_conv_inputs(
                        tok, conv, device)
                except Exception:
                    continue

                # ── 1. 历史轮 KV 缓存（eval, no_grad）──
                model.eval()
                for layer in model.layers:
                    layer.attention._kv_cache_enabled = True
                with torch.no_grad():
                    model(hist_ids)
                layer_kvs = []
                for layer in model.layers:
                    k, v = layer.attention._last_kv
                    layer_kvs.append((k[0].float().cpu(),
                                      v[0].float().cpu()))
                mb.add_round_kv(layer_kvs, text='hist')

                # ── 2. 当前轮训练（train, 带 mem_kv）──
                mem_kv = {}
                dev = next(model.parameters()).device
                for i, layer in enumerate(model.layers):
                    kv = mb.get_kv(i)
                    if kv is None:
                        break
                    mem_kv[i] = (kv[0].unsqueeze(0).to(dev),
                                 kv[1].unsqueeze(0).to(dev))
                model.train()
                logits = model(cur_ids, mem_kv=mem_kv if mem_kv else None)
                # 监督最后 assistant 回复（CLM shift + mask）
                shift = cur_ids[:, 1:]
                labels = shift.clone()
                if reply_start > 1:
                    labels[:, :reply_start - 1] = -100
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    labels.reshape(-1), ignore_index=-100)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.grad is not None], 1.0)
                opt.step()

                total_loss += loss.item()
                mb.clear_kv()  # 每样本用完即清（下一样本独立记忆）
                step += 1

                if step % args.log_every == 0:
                    print(f'  step {step}/{args.steps}: loss={total_loss/args.log_every:.4f}')
                    total_loss = 0.0
                if step % args.save_every == 0:
                    path = os.path.join(args.output_dir, f'memory_tune_{step}.pt')
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'config': model.config,
                        'step': step, 'phase': 'memory_tune',
                    }, path)
                    print(f'  [save] {path}')
                if step >= args.steps:
                    break

    final = os.path.join(args.output_dir, 'memory_tuned.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model.config,
        'step': step, 'phase': 'memory_tune',
    }, final)
    print(f'记忆微调完成 -> {final}')


if __name__ == '__main__':
    main()
