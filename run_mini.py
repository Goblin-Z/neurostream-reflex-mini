"""
ReflexMini 部署运行脚本（完整模式：内部循环全开）。

所有训练阶段关闭的模块都会激活：
  - InternalLoop（SelfModel 状态前向 + Hebbian 更新 + Critic + 辩证缓冲 + 巩固）
  - 架构自修改默认关闭（无对照实验前不改写已训练权重），
    可用 --arch-self-mod 显式开启（风险自负）。

用法:
  python run_mini.py --checkpoint /root/autodl-tmp/checkpoints/reflex-mini/sft_final.pt
  python run_mini.py --checkpoint ... --device cuda --tokenizer /root/autodl-tmp/data/qwen2.5-0.5b
  python run_mini.py --checkpoint ... --no-internal-loop   # 关闭内循环（纯推理对照）
  python run_mini.py --checkpoint ... --arch-self-mod      # 显式开启架构自修改

Qwen3.8-27B 嫁接模式（checkpoint 由 scripts/load_qwen3_graft.py 生成）:
  python run_mini.py --checkpoint .../reflex_qwen3_27b_graft.pt \
      --tokenizer /root/autodl-tmp/data/Qwen3.8-27B --device cuda --dtype bfloat16
  # 配置自动从 checkpoint 恢复（backbone=deepseek_v2 → DeepSeekV2GraftConfig）
  # --full-graft: 关闭 graft_lite（Hebbian 全 64 层 + 巩固全开，算力/显存代价高）
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import AutoTokenizer

from config.model_config import ReflexMiniConfig, config_from_checkpoint
from core.model import ReflexModel
from loop.internal_loop import InternalLoop
from interaction.pipeline import ReflexPipeline
from interaction.manager import InteractionManager


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--tokenizer', default=None,
                   help='HF id 或本地路径（默认 Qwen/Qwen2.5-0.5B；嫁接模式建议传 Qwen3.8-27B 目录）')
    p.add_argument('--dtype', default='bfloat16',
                   choices=['bfloat16', 'float16', 'float32'])
    p.add_argument('--config', default=None, choices=['mini', 'qwen3-27b'],
                   help='显式指定配置（默认从 checkpoint 自动恢复）')
    p.add_argument('--no-internal-loop', action='store_true',
                   help='关闭内循环（对照：纯生成）')
    p.add_argument('--arch-self-mod', action='store_true',
                   help='显式开启架构自修改（默认关闭，风险自负）')
    p.add_argument('--verify-threshold', type=float, default=None,
                   help='主动求证 sigma 阈值（默认 config 0.5；调低更易触发）')
    p.add_argument('--graft-lite', dest='graft_lite', action='store_true', default=None,
                   help='嫁接轻量模式（冻结主干 + Hebbian 尾层，默认随 config）')
    p.add_argument('--full-graft', dest='graft_lite', action='store_false',
                   help='嫁接完整模式（Hebbian 全层 + 巩固全开，显存/算力代价高）')
    p.add_argument('--no-online-ce', action='store_true',
                   help='嫁接模式：关闭每轮全量 CE 在线训练（只留 Hebbian/内循环，首轮试验推荐）')
    p.add_argument('--max-new-tokens', type=int, default=None,
                   help='对话回答生成上限（默认 config.max_new_tokens=192；'
                        'Qwen3 带 <think> 块建议 >=512）')
    p.add_argument('--hide-think', action='store_true',
                   help='显示时剥离 <think>...</think> 思考块（只看最终答案）')
    p.add_argument('--gen-debug', action='store_true',
                   help='打印生成停止原因（终止符/重复/think 预算/上限），排查截断用')
    p.add_argument('--eos-grace', type=int, default=None,
                   help='think 未闭合时终止符宽容次数（默认 2；base 模型思考起步'
                        '常误输出 eos 导致空回复；设 0 = 立即停）')
    p.add_argument('--sigma-cal', action='store_true',
                   help='sigma 在线校准：尾层 uncertainty_head 学习 tanh(loss_int)，'
                        '让 sigma 反映真实不确定度（主动求证触发的基础）')
    p.add_argument('--max-asks', type=int, default=None,
                   help='会话提问上限（默认 50；0 = 无限；你遇到的 can_ask=False '
                        '是旧 checkpoint 的 5 次上限，用此参数覆盖）')
    p.add_argument('--hebbian-lr', type=float, default=None,
                   help='Hebbian 学习率（checkpoint 默认 1e-6；建议 1e-5~1e-4，'
                        '改的是 Qwen 原权重，越高越快但也越有破坏风险）')
    p.add_argument('--hebbian-layers', type=int, default=None,
                   help='Hebbian 覆盖尾层数（默认 20；16/24 学习面更大，算力更高）')
    p.add_argument('--memory-write-lr', type=float, default=None,
                   help='L3 语义槽写入门强度（checkpoint 默认 0.05；更稳建议 0.01）')
    args = p.parse_args()

    device = args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'

    dtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16,
             'float32': torch.float32}[args.dtype]
    if device == 'cpu' and dtype != torch.float32:
        print(f'[WARN] CPU 仅支持 float32，dtype 强制为 float32')
        dtype = torch.float32

    # mmap 加载：不把 54GB checkpoint 全量读进物理内存（需 torch>=2.1）
    try:
        ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    except TypeError:
        ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if args.config == 'qwen3-27b':
        from config.model_config import DeepSeekV2GraftConfig
        config = DeepSeekV2GraftConfig()
    elif args.config == 'mini':
        config = ReflexMiniConfig()
    else:
        config = config_from_checkpoint(ck)
    if args.arch_self_mod:
        if getattr(config, 'backbone', '') != 'reflex':
            print('[WARN] 嫁接模式不支持架构自修改（Qwen3GraftLayer 无专家分裂接口），已强制关闭')
            config.arch_self_mod_enabled = False
        else:
            config.arch_self_mod_enabled = True
            print('[INFO] 架构自修改已开启（专家分裂/裁剪/加层，风险自负）')
    else:
        config.arch_self_mod_enabled = False
        print('[INFO] 架构自修改默认关闭（可用 --arch-self-mod 开启）')
    if args.verify_threshold is not None:
        config.verify_threshold = args.verify_threshold
        print(f'[INFO] verify_threshold 调为 {args.verify_threshold} '
              f'(默认 0.5，调低更易触发主动求证)')
    if args.graft_lite is not None and getattr(config, 'backbone', '') != 'reflex':
        config.graft_lite = args.graft_lite
        print(f'[INFO] graft_lite = {config.graft_lite}')
    if args.no_online_ce and getattr(config, 'backbone', '') != 'reflex':
        config.graft_online_ce = False
        print('[INFO] 每轮全量 CE 在线训练已关闭（graft_online_ce=False）')
    if args.max_new_tokens is not None:
        config.max_new_tokens = args.max_new_tokens
        print(f'[INFO] max_new_tokens = {args.max_new_tokens}')
    if args.gen_debug and getattr(config, 'backbone', '') != 'reflex':
        config.graft_gen_debug = True
        print('[INFO] 生成停止原因诊断已开启（--gen-debug）')
    if args.eos_grace is not None and getattr(config, 'backbone', '') != 'reflex':
        config.graft_think_eos_grace = args.eos_grace
        print(f'[INFO] think 未闭合终止符宽容次数 = {args.eos_grace}')
    if args.sigma_cal and getattr(config, 'backbone', '') != 'reflex':
        config.graft_sigma_cal = True
        print('[INFO] sigma 在线校准已开启（--sigma-cal）')
    if args.max_asks is not None:
        config.max_questions_per_session = args.max_asks
        print(f'[INFO] 会话提问上限 = {args.max_asks}'
              f'（0=无限）' if args.max_asks == 0
              else f'[INFO] 会话提问上限 = {args.max_asks}')
    # ── 学习强度覆盖（已有 checkpoint 的 config 存旧值，用 CLI 覆盖生效）──
    # ── 学习率光谱（v3.9 整体升一量级；v3.10/v3.11 可塑档两连升；
    #    v3.12 可塑档 ×8 = 8e-2（极端档）+ 其余各升一量级）──
    # 覆盖 checkpoint 内旧光谱，无需重新生成 checkpoint。
    # 注意：--hebbian-lr 在下方处理，仍可最后覆盖（如 0 关闭）。
    if getattr(config, 'backbone', '') != 'reflex':
        config.expert_baseline_lrs = (
            (1e-5,) * 16 + (1e-4,) * 32 + (8e-2,) * 16)
        config.shared_expert_lr = 1e-4
        config.dense_expert_lr = 1e-5
        print('[INFO] 学习率光谱（路由 16×1e-5 / 32×1e-4 / 16×8e-2；'
              '共享 1e-4；dense 1e-5）——覆盖 checkpoint 旧光谱')
    if args.hebbian_lr is not None and getattr(config, 'backbone', '') != 'reflex':
        config.expert_baseline_lrs = (args.hebbian_lr,)
        # --hebbian-lr 0：同时关闭共享/密集专家（否则共享 1e-6 仍会更新，
        # hebbian_drift 缓慢增长，对照不彻底）
        config.shared_expert_lr = args.hebbian_lr
        config.dense_expert_lr = args.hebbian_lr
        print(f'[INFO] Hebbian 学习率 = {args.hebbian_lr} '
              f'（覆盖 checkpoint 默认；含路由/共享/密集专家）')
    if args.hebbian_layers is not None and getattr(config, 'backbone', '') != 'reflex':
        config.graft_hebbian_layers = args.hebbian_layers
        print(f'[INFO] Hebbian 覆盖尾层数 = {args.hebbian_layers}')
    if args.memory_write_lr is not None:
        config.memory_write_lr = args.memory_write_lr
        print(f'[INFO] L3 语义槽写入门 = {args.memory_write_lr}')

    # ── DeepSeek Chat 官方采样参数（关键修复）──
    # Qwen 时代默认（temp 0.8 / top_k 40 / rep 1.5）对 Chat 模型是灾难：
    # repetition_penalty=1.5 强惩罚常用词 → 模型绕向 emoji/符号/生僻字符
    # 退化；官方 generation_config 为 temp=0.3 / top_p=0.95 / 无 top_k /
    # 无重复惩罚。对 graft 主干强制官方参数（旧 checkpoint 无需重生成）。
    if getattr(config, 'backbone', '') != 'reflex':
        config.sampling_temperature = 0.3
        config.sampling_top_k = 0
        config.sampling_top_p = 0.95
        config.sampling_repetition_penalty = 1.0
        print('[INFO] DeepSeek Chat 官方采样参数启用（temp=0.3, top_p=0.95, '
              'top_k=0, 无重复惩罚）')

    tok_name = args.tokenizer or os.environ.get(
        'TOKENIZER_PATH',
        'Qwen/Qwen2.5-0.5B' if getattr(config, 'backbone', '') == 'reflex'
        else '/root/autodl-tmp/data/Qwen3.8-27B')
    print(f'Loading tokenizer: {tok_name}')
    tok = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f'Building ReflexModel (backbone={getattr(config, "backbone", "reflex")})...')
    model = ReflexModel(config)
    # 内存优化：加载前先转目标 dtype（fp32 中间态 29.45B×4B≈118GB 会被 OOM Killed）
    model.to(dtype=dtype)
    ck_step = ck.get('step')
    ck_phase = ck.get('phase')
    model.load_state_dict(ck['model_state_dict'], strict=False)
    # 释放 checkpoint 引用（mmap 页缓存可回收）
    del ck
    import gc
    gc.collect()
    # AttnRes post_norm：不再强制覆写——部署与 checkpoint 训练行为保持一致。
    # （记忆微调 train_memory.py 会单独处理放大，作为记忆利用训练的前提）
    if model.attn_res is not None:
        pn = [m.post_norm.weight.mean().item() for m in model.attn_res.modules_list]
        print(f'[INFO] AttnRes post_norm 保持 checkpoint 尺度 (mean={sum(pn)/len(pn):.4f})')
    model._decode_tokenizer = tok
    model.to(device=device, dtype=dtype)
    model.eval()
    print(f'Checkpoint: step={ck_step} phase={ck_phase} '
          f'params={sum(p.numel() for p in model.parameters())/1e6:.0f}M '
          f'dtype={args.dtype}')

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
            # 状态门控生效观测（v4 §七·八：从"无影响"逐步到"带想法"）
            print(f'  h_to_bias(max): {internal_loop.h_to_bias_drift:.4f}, '
                  f'global_drift: {internal_loop.global_drift:.4f}, '
                  f'hebbian_drift(max): {internal_loop.hebbian_drift:.4f}')
            # 显示内循环最近被吞掉的异常（_loop 的 except 只记录不打印）
            with internal_loop._stats_lock:
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
        if args.hide_think:
            import re
            resp = re.sub(r'<think>.*?</think>', '', resp, flags=re.S)
            resp = re.sub(r'\s+', ' ', resp).strip()
        print(f'{resp}\n')

    if internal_loop is not None:
        internal_loop.stop()
    print('Done.')


if __name__ == '__main__':
    main()
