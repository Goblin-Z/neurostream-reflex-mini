#!/usr/bin/env python3
"""
load_deepseek_graft.py — 将 DeepSeek-V2-Lite 原始权重映射进 ReflexModel
（嫁接 checkpoint 生成器）。

在 AutoDL 下载 DeepSeek-V2-Lite 之后运行（无需联网；权重约 31GB bf16）：

  python scripts/load_deepseek_graft.py \
      --model-path /root/autodl-tmp/data/DeepSeek-V2-Lite \
      --output /root/autodl-tmp/checkpoints/reflex_dsv2lite_graft.pt \
      --dtype bf16

关键映射（详见 docs/DSV2_GRAFT_DESIGN.md）：
  - embed_tokens / lm_head / RMSNorm: 直接拷贝（DeepSeek RMSNorm 与项目同构，
    无需 Qwen3 的 1+w 变换）
  - MLA 投影: q_proj / kv_a_proj_with_mqa / kv_a_layernorm / kv_b_proj / o_proj
    直拷（Lite 无 q_lora_rank，无 q_a/q_b）
  - dense 层（层 0）: mlp.gate_proj/up_proj/down_proj → all_experts.0.w_*
  - MoE 层（层 1-26）:
      mlp.gate.weight        → router.gate_weight（布局 [64, 2048] 直拷）
      mlp.experts.{j}.*      → routed_experts.{j}.w_*（直拷）
      mlp.shared_experts.*   → 合并 MLP 直拷为单个 Expert（intermediate ×
                              n_shared=2816，1:1 不拆分；shared_expert_lr 单学习率）

内存策略：逐 safetensors 分片读取 → 即时映射 → 增量 load_state_dict
（峰值 ≈ 模型 46GB(bf16) + 单分片 ~8GB）。

数值验证见 scripts/verify_deepseek.py。
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from config.model_config import DeepSeekV2GraftConfig
from core.model import ReflexModel

_TEXT_PREFIX = 'model'


def map_deepseek_state_dict(qwen_sd: dict, n_layers: int, layer_types: list):
    """把 DeepSeek 原始 state dict 映射为 ReflexModel key（宽容版，缺 key 跳过）。

    返回 (mapped, 本分片出现的 shared_experts 权重是否完整拆分)。
    """
    mapped = {}

    def put(key, tensor):
        if tensor is not None:
            mapped[key] = tensor

    put('token_embedding.weight', qwen_sd.get(f'{_TEXT_PREFIX}.embed_tokens.weight'))
    put('ln_f.weight', qwen_sd.get(f'{_TEXT_PREFIX}.norm.weight'))
    put('lm_head.weight', qwen_sd.get('lm_head.weight'))

    for i in range(n_layers):
        pre = f'{_TEXT_PREFIX}.layers.{i}'
        put(f'layers.{i}.ln1.weight',
            qwen_sd.get(f'{pre}.input_layernorm.weight'))
        put(f'layers.{i}.ln2.weight',
            qwen_sd.get(f'{pre}.post_attention_layernorm.weight'))

        # MLA 注意力（直拷）
        a = f'{pre}.self_attn'
        for key in ('q_proj', 'kv_a_proj_with_mqa', 'kv_b_proj', 'o_proj'):
            put(f'layers.{i}.attention.{key}.weight',
                qwen_sd.get(f'{a}.{key}.weight'))
        put(f'layers.{i}.attention.kv_a_layernorm.weight',
            qwen_sd.get(f'{a}.kv_a_layernorm.weight'))

        mlp = f'{pre}.mlp'
        if layer_types[i] == 'dense':
            # dense 层：单专家
            put(f'layers.{i}.all_experts.0.w_gate.weight',
                qwen_sd.get(f'{mlp}.gate_proj.weight'))
            put(f'layers.{i}.all_experts.0.w_up.weight',
                qwen_sd.get(f'{mlp}.up_proj.weight'))
            put(f'layers.{i}.all_experts.0.w_down.weight',
                qwen_sd.get(f'{mlp}.down_proj.weight'))
        else:
            # MoE 层：router + 64 路由专家
            put(f'layers.{i}.router.gate_weight',
                qwen_sd.get(f'{mlp}.gate.weight'))
            n_routed = 64
            for j in range(n_routed):
                e = f'{mlp}.experts.{j}'
                put(f'layers.{i}.routed_experts.{j}.w_gate.weight',
                    qwen_sd.get(f'{e}.gate_proj.weight'))
                put(f'layers.{i}.routed_experts.{j}.w_up.weight',
                    qwen_sd.get(f'{e}.up_proj.weight'))
                put(f'layers.{i}.routed_experts.{j}.w_down.weight',
                    qwen_sd.get(f'{e}.down_proj.weight'))
            # 共享专家（合并 MLP intermediate=1408×2=2816）→ 单个 Expert 直拷，
            # 保持原权重完整性（不拆分、零浮点舍入）
            put(f'layers.{i}.shared_experts.0.w_gate.weight',
                qwen_sd.get(f'{mlp}.shared_experts.gate_proj.weight'))
            put(f'layers.{i}.shared_experts.0.w_up.weight',
                qwen_sd.get(f'{mlp}.shared_experts.up_proj.weight'))
            put(f'layers.{i}.shared_experts.0.w_down.weight',
                qwen_sd.get(f'{mlp}.shared_experts.down_proj.weight'))

    return mapped


def build_graft_config(model_path: str, max_seq_len: int = 8192) -> DeepSeekV2GraftConfig:
    cfg_path = os.path.join(model_path, 'config.json')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f'{cfg_path} 不存在（请确认下载了完整模型目录）')
    with open(cfg_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    kwargs = {}
    for reflex_field, qwen_field in {
        'd_model': 'hidden_size',
        'n_layers': 'num_hidden_layers',
        'n_heads': 'num_attention_heads',
        'qk_nope_head_dim': 'qk_nope_head_dim',
        'qk_rope_head_dim': 'qk_rope_head_dim',
        'kv_lora_rank': 'kv_lora_rank',
        'v_head_dim': 'v_head_dim',
        'intermediate_size': 'intermediate_size',
        'moe_intermediate_size': 'moe_intermediate_size',
        'vocab_size': 'vocab_size',
        'n_routed_experts': 'n_routed_experts',
        'n_shared_experts': 'n_shared_experts',
        'num_experts_per_tok': 'num_experts_per_tok',
        'first_k_dense_replace': 'first_k_dense_replace',
        'moe_layer_freq': 'moe_layer_freq',
        'norm_topk_prob': 'norm_topk_prob',
        'topk_method': 'topk_method',
        'n_group': 'n_group',
        'topk_group': 'topk_group',
        'scoring_func': 'scoring_func',
        'aux_loss_alpha': 'aux_loss_alpha',
        'routed_scaling_factor': 'routed_scaling_factor',
        'seq_aux': 'seq_aux',
        'rope_theta': 'rope_theta',
        'rms_norm_eps': 'rms_norm_eps',
    }.items():
        if qwen_field in raw:
            kwargs[reflex_field] = raw[qwen_field]
    if 'rope_scaling' in raw:
        kwargs['rope_scaling'] = raw['rope_scaling']
    kwargs['max_seq_len'] = max_seq_len

    config = DeepSeekV2GraftConfig(**kwargs)
    # 解析 layer_types（与 ReflexModel 一致）：前 first_k_dense_replace 层 dense
    if not getattr(config, 'layer_types', None):
        first_k = getattr(config, 'first_k_dense_replace', 1)
        config.layer_types = (
            ('dense',) * first_k + ('moe',) * (config.n_layers - first_k))
    expect = {
        'd_model': 2048, 'n_layers': 27, 'n_heads': 16,
        'n_routed_experts': 64, 'n_shared_experts': 2,
        'num_experts_per_tok': 6, 'vocab_size': 102400,
    }
    for k, v in expect.items():
        if getattr(config, k) != v:
            print(f'[WARN] config.json {k}={getattr(config, k)} ≠ 预期 {v} '
                  f'（若非 DeepSeek-V2-Lite 请确认模型路径）')
    return config


def load_safetensors_iter(model_path: str):
    files = sorted(glob.glob(os.path.join(model_path, '*.safetensors')))
    if not files:
        raise FileNotFoundError(f'{model_path} 下没有 *.safetensors 分片')
    for f in files:
        print(f'  读取 {os.path.basename(f)} ...', flush=True)
        from safetensors.torch import load_file
        yield f, load_file(f, device='cpu')


def build_graft_checkpoint(model_path: str, output_path: str,
                           dtype: str = 'bf16', max_seq_len: int = 8192):
    t0 = time.time()
    torch_dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16,
                   'fp32': torch.float32}[dtype]

    config = build_graft_config(model_path, max_seq_len=max_seq_len)
    print(f'[CONFIG] d_model={config.d_model} n_layers={config.n_layers} '
          f'heads={config.n_heads} MLA(kv_lora={config.kv_lora_rank}, '
          f'qk_rope={config.qk_rope_head_dim}) '
          f'routed={config.n_routed_experts} shared={config.n_shared_experts} '
          f'top-{config.num_experts_per_tok} vocab={config.vocab_size}')
    print(f'[CONFIG] layer_types[0:4]={list(config.layer_types[:4])} '
          f'dense_layers={sum(1 for t in config.layer_types if t=="dense")} '
          f'moe_layers={sum(1 for t in config.layer_types if t=="moe")}')

    print('构建 ReflexModel (DeepSeekV2GraftConfig)...')
    model = ReflexModel(config)
    model.to(dtype=torch_dtype)
    n_backbone = sum(p.numel() for p in model.parameters())
    print(f'  模型总参数: {n_backbone/1e9:.2f}B（含 Reflex 附加件，dtype={dtype}）')
    # 期望值: ≈23.0B = 15.7B 主干 + ~7.3B Reflex 附加件（各专家 query_proj/
    # uncertainty_head + SelfModel/AttnRes/MemoryBank 等）——23.0B 是正确值，
    # 不是旧包特征。若明显偏离（<18B 或 >28B）才需核对 VERSION.txt。

    # 去重硬校验：同一 Expert 被注册多次 → 参数对象 id 重复。修复版中
    # all_experts/stable_experts/plastic_experts 均为普通 list（不注册）。
    # （旧版 ModuleList 重复注册会造成 checkpoint key 重复与
    #   run_mini global_drift 快照错位崩溃。）
    param_ids = [id(p) for p in model.parameters()]
    if len(param_ids) != len(set(param_ids)):
        raise RuntimeError(
            f'检测到 {len(param_ids) - len(set(param_ids))} 个重复注册的参数对象'
            f'（总参数 {n_backbone/1e9:.2f}B）—— 专家被重复注册（ModuleList 与'
            f'普通 list 混用），请使用含去重修复的最新代码包（核对 VERSION.txt）。')
    print(f'  [OK] 参数无重复注册（{len(set(param_ids))} 个参数张量，'
          f'总参数 {n_backbone/1e9:.2f}B ≈ 23.0B 为正确值）')

    from collections import Counter
    seen_prefix = Counter()
    mapped_keys = set()
    total_keys = 0
    for fname, shard in load_safetensors_iter(model_path):
        for k in shard:
            parts = k.split('.')
            seen_prefix['.'.join(parts[:2])] += 1
        mapped = map_deepseek_state_dict(
            shard, config.n_layers, list(config.layer_types))
        missing, unexpected = model.load_state_dict(mapped, strict=False)
        mapped_keys.update(mapped.keys())
        total_keys += len(mapped)
        print(f'  -> 映射 {len(mapped)} 个 key（missing={len(missing)} '
              f'unexpected={len(unexpected)}）')
        del shard, mapped
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 终检：主干 key 必须全部就位
    layer_types = list(config.layer_types)

    def _is_backbone_key(k):
        if k in ('token_embedding.weight', 'lm_head.weight', 'ln_f.weight'):
            return True
        parts = k.split('.')
        if len(parts) >= 3 and parts[0] == 'layers' and parts[1].isdigit():
            li = int(parts[1])
            if parts[2] in ('ln1', 'ln2', 'attention'):
                return True
            # router: 仅 gate_weight 是 DeepSeek 权重（moe 层）
            if (parts[2] == 'router' and len(parts) >= 4
                    and parts[3] == 'gate_weight' and layer_types[li] == 'moe'):
                return True
            if parts[2] == 'all_experts' and layer_types[li] == 'dense':
                if len(parts) >= 5 and parts[4] in ('w_gate', 'w_up', 'w_down'):
                    return True
            if len(parts) >= 5 and parts[2] in ('routed_experts',
                                                'shared_experts'):
                if parts[4] in ('w_gate', 'w_up', 'w_down'):
                    return True
        return False

    model_keys = set(model.state_dict().keys())
    backbone_missing = sorted(
        k for k in model_keys if _is_backbone_key(k) and k not in mapped_keys)
    addon_count = len(model_keys) - len(mapped_keys)
    print(f'\n[CHECK] 主干 key 总数='
          f'{sum(1 for k in model_keys if _is_backbone_key(k))} '
          f'已映射={len(mapped_keys)} 主干缺失={len(backbone_missing)} '
          f'Reflex 附加件（预期随机初始化）={addon_count}')
    if backbone_missing:
        print('  主干 missing keys:')
        for k in backbone_missing[:10]:
            print(f'    {k}')
        print('  —— 实际检测到的 key 前缀分布:')
        for prefix, cnt in seen_prefix.most_common(20):
            print(f'    {prefix}: {cnt}')
        raise RuntimeError('主干权重映射不完整：请检查模型版本/路径')

    # 形状断言
    l1 = model.layers[1]
    assert model.token_embedding.weight.shape == (config.vocab_size, config.d_model)
    assert model.lm_head.weight.shape == (config.vocab_size, config.d_model)
    assert l1.attention.q_proj.weight.shape[0] == config.n_heads * (
        config.qk_nope_head_dim + config.qk_rope_head_dim)
    assert l1.attention.kv_b_proj.weight.shape[0] == config.n_heads * (
        config.qk_nope_head_dim + config.v_head_dim)
    assert len(l1.routed_experts) == config.n_routed_experts
    # 共享专家：DeepSeek 语义 n_shared=2，实现为 1 个合并 MLP Expert（保持完整性）
    assert len(l1.shared_experts) == 1
    assert l1.routed_experts[0].w_gate.weight.shape == (
        config.moe_intermediate_size, config.d_model)
    # 共享专家为合并 MLP（1:1 直拷，完整性保持）
    assert l1.shared_experts[0].w_gate.weight.shape == (
        config.moe_intermediate_size * config.n_shared_experts, config.d_model)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    ck = {
        'model_state_dict': model.state_dict(),
        'config': config.__dict__,
        'step': 0,
        'phase': 'graft_deepseek-v2-lite',
        'dtype': dtype,
        'grafted_from': os.path.basename(model_path.rstrip('/')),
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    torch.save(ck, output_path)
    size_gb = os.path.getsize(output_path) / 1e9
    print(f'\n[OK] 嫁接 checkpoint 已保存: {output_path} '
          f'({size_gb:.1f} GB, {total_keys} 个权重 key, '
          f'用时 {time.time()-t0:.0f}s)')
    print('\n下一步（AutoDL）:')
    print(f'  1) 数值验证: python scripts/verify_deepseek.py '
          f'--model-path {model_path} --checkpoint {output_path}')
    print(f'  2) 冒烟测试: python chat_sft.py --checkpoint {output_path} '
          f'--tokenizer {model_path} --device cuda --prompt "中国的首都是"')
    print(f'  3) 完整部署: python run_mini.py --checkpoint {output_path} '
          f'--tokenizer {model_path} --device cuda --dtype bf16')
    return output_path


def main():
    p = argparse.ArgumentParser(
        description='DeepSeek-V2-Lite → Reflex 嫁接 checkpoint 生成器')
    p.add_argument('--model-path', required=True,
                   help='DeepSeek-V2-Lite 本地目录（含 config.json 与 *.safetensors）')
    p.add_argument('--output', required=True, help='输出 checkpoint 路径 (.pt)')
    p.add_argument('--dtype', default='bf16', choices=['bf16', 'fp16', 'fp32'])
    p.add_argument('--max-seq-len', type=int, default=8192,
                   help='部署序列上限（原生 163840，按显存/算力调小）')
    args = p.parse_args()
    for name, path in (('--model-path', args.model_path),
                       ('--output', args.output)):
        if '...' in path or path.strip() in ('', '.'):
            p.error(f'{name} 无效: {path!r} —— 请替换为真实路径（不要保留占位符 ...）')
    if not os.path.isdir(args.model_path):
        p.error(f'--model-path 目录不存在: {args.model_path}')
    if not os.path.exists(os.path.join(args.model_path, 'config.json')):
        p.error(f'--model-path 下没有 config.json（请确认完整模型目录）: {args.model_path}')

    build_graft_checkpoint(args.model_path, args.output, args.dtype,
                           args.max_seq_len)


if __name__ == '__main__':
    main()
