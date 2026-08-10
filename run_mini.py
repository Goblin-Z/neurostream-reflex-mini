"""
ReflexMini 部署运行脚本（完整模式：内部循环全开）。

所有训练阶段关闭的模块都会激活：
  - InternalLoop（SelfModel 状态前向 + Hebbian 更新 + Critic + 辩证缓冲 + 巩固）
  - 架构自修改默认关闭（已知 bug：split/prune 后 forward 崩溃），
    可用 --arch-self-mod 显式开启（风险自负）。

用法:
  python run_mini.py --checkpoint /root/autodl-tmp/checkpoints/reflex-mini/sft_final.pt
  python run_mini.py --checkpoint ... --device cuda --tokenizer /root/autodl-tmp/data/qwen2.5-0.5b
  python run_mini.py --checkpoint ... --no-internal-loop   # 关闭内循环（纯推理对照）
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import AutoTokenizer

from config.model_config import ReflexMiniConfig
from core.model import ReflexModel
from loop.internal_loop import InternalLoop
from interaction.pipeline import ReflexPipeline
from interaction.manager import InteractionManager


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--tokenizer', default='/root/autodl-tmp/data/qwen2.5-0.5b')
    p.add_argument('--no-internal-loop', action='store_true',
                   help='关闭内循环（对照：纯生成）')
    p.add_argument('--no-arch-self-mod', action='store_true',
                   help='关闭架构自修改（C2 已修复，默认开启）')
    p.add_argument('--verify-threshold', type=float, default=None,
                   help='主动求证 sigma 阈值（默认 config 0.5；调低更易触发）')
    args = p.parse_args()

    device = args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'

    config = ReflexMiniConfig()
    if args.no_arch_self_mod:
        config.arch_self_mod_enabled = False
        print('[INFO] 架构自修改已关闭 (--no-arch-self-mod)')
    else:
        # C2 已修复（专家增减/加层不再崩溃）——默认开启架构自修改
        print('[INFO] 架构自修改已开启（专家分裂/裁剪/加层，C2 修复后安全）')
    if args.verify_threshold is not None:
        config.verify_threshold = args.verify_threshold
        print(f'[INFO] verify_threshold 调为 {args.verify_threshold} '
              f'(默认 0.5，调低更易触发主动求证)')

    print(f'Loading tokenizer: {args.tokenizer}')
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print('Building ReflexMini...')
    model = ReflexModel(config)
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model_state_dict'], strict=False)
    # post_norm 放大：现有 checkpoint 的 AttnRes 权重（~0.005）几乎无声，
    # 统一设为目标尺度让记忆 source 真实影响输出（记忆微调前提）
    if model.attn_res is not None:
        pn_init = getattr(config, 'attnres_postnorm_init', 0.1)
        for m in model.attn_res.modules_list:
            m.post_norm.weight.data.fill_(pn_init)
        print(f'[INFO] AttnRes post_norm 已放大至 {pn_init}（记忆真实影响输出）')
    model._decode_tokenizer = tok
    model.to(device)
    model.eval()
    print(f'Checkpoint: step={ck.get("step")} phase={ck.get("phase")} '
          f'params={sum(p.numel() for p in model.parameters())/1e6:.0f}M')

    mgr = InteractionManager(config)
    pipeline = ReflexPipeline(model, config)
    pipeline.interaction = mgr
    pipeline.set_tokenizer(tok)

    internal_loop = None
    if not args.no_internal_loop:
        internal_loop = InternalLoop(model, config, interaction_mgr=mgr)
        internal_loop.start()
        print('[INFO] InternalLoop 已启动（Hebbian/Critic/SelfModel 在线运行）')
    else:
        print('[INFO] 纯生成模式（内循环关闭）')

    print('\n输入问题，"quit" 退出，"stats" 查看内循环状态\n')
    while True:
        try:
            user_input = input('>>> ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nBye!')
            break
        if not user_input:
            continue
        if user_input.lower() in ('quit', 'exit', 'q'):
            break
        if user_input.lower() in ('clear', 'reset'):
            # 清空会话记忆（L0 历史 + L4 KV）+ 重置状态，防止垃圾累积污染
            pipeline._chat_history = []
            mb = getattr(model, 'memory_bank', None)
            if mb is not None:
                mb.clear_kv()
                print('  [MEM] 已清空对话历史 + KV 内容记忆')
            if internal_loop is not None:
                internal_loop._mem_kv = None
                internal_loop._mem_kv_n = -1
                internal_loop._critic_pending = None
                print('  [MEM] 内循环记忆缓存已重置')
            print('  [MEM] 会话记忆已清空（L0 历史 / L4 KV）')
            continue
        if user_input.lower() == 'stats' and internal_loop is not None:
            mgr_stats = mgr.get_stats()
            print(f'  internal steps: {internal_loop._total_steps}, '
                  f'loss_int: {internal_loop._loss_int.item() if internal_loop._loss_int is not None else "N/A"}')
            print(f'  sigma(last): {internal_loop._last_sigma:.4f}, '
                  f'mgr sigma: {mgr_stats.get("sigma", 0):.4f}, '
                  f'threshold: {mgr_stats.get("sigma_threshold", 0.5)}, '
                  f'state: {mgr_stats.get("state")}, '
                  f'can_ask: {mgr_stats.get("can_ask")}, '
                  f'asked: {mgr_stats.get("total_asked", 0)}')
            # 显示内循环最近被吞掉的异常（_loop 的 except 只记录不打印）
            errs = [h.get('error') for h in internal_loop._loss_history
                    if 'error' in h][-3:]
            if errs:
                print('  [ERRORS] 最近内循环异常（静默吞掉）:')
                for e in errs:
                    print(f'    {str(e)[:200]}')
            continue
        if internal_loop is not None:
            internal_loop.pause()
        try:
            resp = pipeline.process_text(user_input)
        finally:
            if internal_loop is not None:
                internal_loop.resume()
        print(f'{resp}\n')

    if internal_loop is not None:
        internal_loop.stop()
    print('Done.')


if __name__ == '__main__':
    main()
