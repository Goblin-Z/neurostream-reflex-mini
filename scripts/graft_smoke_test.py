#!/usr/bin/env python3
"""
graft_smoke_test.py — DeepSeek-V2-Lite 嫁接代码路径的本地冒烟测试（CPU 小尺寸）。

覆盖：
  1. 构建 + 前向 + 反向（dense + MoE 层，训练路径）
  2. MLA 增量解码（past KV）与全量前向 logits 一致性
  3. 共享专家拆分等价性（合并 MLP ≡ 两个拆分专家之和）
  4. 路由语义（softmax 全分布 top-k、不归一化权重，与官方公式一致）
  5. 权重映射函数（合成 key + 共享拆分）
  6. checkpoint 存取 + config_from_checkpoint 恢复
  7. bf16 E2E（pipeline 对话 + InternalLoop 内循环）

用法: python scripts/graft_smoke_test.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from config.model_config import DeepSeekV2GraftConfig, config_from_checkpoint
from core.model import ReflexModel
from core.deepseek_router import DeepSeekRouter

PASS = 0
FAIL = 0


def check(name, cond, detail=''):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f'  [PASS] {name}')
    else:
        FAIL += 1
        print(f'  [FAIL] {name} {detail}')


def make_tiny_config():
    """小尺寸嫁接配置：3 层 = 1 dense + 2 moe；MLA/路由/共享专家全部覆盖。"""
    return DeepSeekV2GraftConfig(
        d_model=64,
        n_layers=3,
        n_heads=4,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        kv_lora_rank=16,
        v_head_dim=8,
        intermediate_size=96,       # dense 层 FFN
        moe_intermediate_size=32,   # 路由/共享专家 FFN
        vocab_size=512,
        max_seq_len=128,
        n_routed_experts=8,
        n_shared_experts=2,
        num_experts_per_tok=3,
        first_k_dense_replace=1,
        layer_types=('dense', 'moe', 'moe'),
        expert_baseline_lrs=(
            (1e-7,) * 2 + (1e-6,) * 4 + (1e-5,) * 2),
        shared_expert_lr=1e-6,
        dense_expert_lr=1e-7,
        attnres_enabled=False,
        rope_scaling={
            'type': 'yarn', 'factor': 2.0,
            'original_max_position_embeddings': 64,
            'beta_fast': 8, 'beta_slow': 1,
            'mscale': 0.707, 'mscale_all_dim': 0.707,
        },
        self_model_z_dim=8,
        self_model_hidden_dim=32,
        critic_hidden_dim=16,
        memory_bank_capacity=8,
        endosphere_capacity=16,
        graft_hebbian_layers=2,
        internal_step_delay_ms=0.0,
        verify_warmup_steps=10 ** 9,
        max_new_tokens=64,
    )


def test_build_and_forward():
    print('\n[1] 构建 + 前向 + 反向（dense + MoE）')
    cfg = make_tiny_config()
    model = ReflexModel(cfg)
    model.train()
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    logits = model(ids)
    check('logits shape', logits.shape == (2, 8, cfg.vocab_size),
          f'{logits.shape}')
    loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), ids.view(-1))
    loss.backward()
    check('backward 无异常', torch.isfinite(loss).item())
    # 层类型与专家数
    check('层类型 dense+moe+moe',
          [l.is_moe for l in model.layers] == [False, True, True])
    l1 = model.layers[1]
    check('moe 层专家数 = 8 路由 + 1 合并共享',
          len(l1.routed_experts) == 8 and len(l1.shared_experts) == 1)
    # 无重复注册：state_dict 每个 key 唯一（all_experts 为普通 list）
    keys = list(model.state_dict().keys())
    dup = [k for k in keys if keys.count(k) > 1]
    check('state_dict 无重复 key', len(dup) == 0, f'{dup[:3]}')
    # 附加件参数规模合理性（moe 层专家只注册一次）
    check('moe 层无 all_experts 重复 key',
          not any(k.startswith('layers.1.all_experts.') for k in keys))
    check('dense 层 1 专家', len(model.layers[0].all_experts) == 1)
    check('路由 top-3', l1.router.top_k == 3)
    check('光谱划分稳定/可塑',
          len(l1.stable_experts) > 0 and len(l1.plastic_experts) > 0)
    check('sigma 聚合（路由+共享混合）',
          isinstance(model._last_sigma_aggregate, float))
    return model, cfg


def test_mla_decode_consistency():
    print('\n[2] MLA 增量解码 vs 全量前向 logits 一致性')
    cfg = make_tiny_config()
    model = ReflexModel(cfg)
    model.eval()
    torch.manual_seed(1)
    ids = torch.randint(0, cfg.vocab_size, (1, 6))

    with torch.no_grad():
        ref = model(ids)                       # 全量 [1,6,V]
        logits_prefill, past = model.forward_graft(ids)
        ext = torch.cat([ids, ids[:, -1:]], dim=-1)
        ref_ext = model(ext)                   # [1,7,V]
        logits_decode, _ = model.forward_graft(ids[:, -1:], past=past)

    d1 = (logits_prefill - ref).abs().max().item()
    d2 = (logits_decode - ref_ext[:, -1:, :]).abs().max().item()
    check('prefill 与全量一致', d1 < 1e-4, f'max|Δ|={d1:.3e}')
    check('decode 与全量一致', d2 < 1e-4, f'max|Δ|={d2:.3e}')

    with torch.no_grad():
        out = model.generate(ids.clone(), max_new_tokens=4,
                             temperature=1.0, top_k=0, top_p=1.0)
    check('generate 增量路径可运行', out.size(1) == 10, f'{out.size(1)}')

    # rope interleaved ≡ transformers 5.x built-in 复数语义：
    # 配对 (x[2d], x[2d+1]) 用频率 f_d 的 cos/sin（缓存前 D//2 列）。
    # 曾用 0::2 列（差 5.5）与 half-split（差 6.1）导致 verify top-1 64%/14%。
    rope = model.layers[1].attention.rope
    torch.manual_seed(7)
    xr = torch.randn(1, 4, 5, rope.dim)
    with torch.no_grad():
        ours = rope(xr)
    cos = rope._cached_cos[:5].unsqueeze(0).unsqueeze(0)
    sin = rope._cached_sin[:5].unsqueeze(0).unsqueeze(0)
    a, b = xr[..., 0::2], xr[..., 1::2]
    c, s = cos[..., : rope.dim // 2], sin[..., : rope.dim // 2]
    ref = torch.stack([a * c - b * s, a * s + b * c], dim=-1).reshape(*xr.shape)
    check('rope interleaved ≡ built-in 复数语义',
          torch.allclose(ours, ref, atol=1e-6),
          f'max|Δ|={(ours - ref).abs().max().item():.3e}')
    return model, cfg


def test_shared_merge_equivalence():
    print('\n[3] 共享专家合并 MLP 完整直拷（不拆分、零舍入）')
    cfg = make_tiny_config()
    d, inter, n_shared = cfg.d_model, cfg.moe_intermediate_size, cfg.n_shared_experts
    torch.manual_seed(0)
    gate = torch.randn(inter * n_shared, d)
    up = torch.randn(inter * n_shared, d)
    down = torch.randn(d, inter * n_shared)
    x = torch.randn(4, d)

    from core.expert import Expert
    e = Expert(d, inter * n_shared, 0.0)
    with torch.no_grad():
        e.w_gate.weight.copy_(gate)
        e.w_up.weight.copy_(up)
        e.w_down.weight.copy_(down)
    with torch.no_grad():
        out = e(x)[0]
    ref = (F.silu(x @ gate.T) * (x @ up.T)) @ down.T
    check('合并 Expert ≡ 官方 MLP（逐位一致）',
          torch.equal(out, ref), f'max|Δ|={(out - ref).abs().max().item():.3e}')
    check('共享专家单一学习率（不拆分）',
          cfg.shared_expert_lr == 1e-6)


def test_router_semantics():
    print('\n[4] 路由语义（softmax 全分布 top-k、不归一化）')
    cfg = make_tiny_config()
    torch.manual_seed(2)
    router = DeepSeekRouter(cfg)
    x = torch.randn(6, cfg.d_model)
    with torch.no_grad():
        top_w, top_idx, logits = router(x)
    # 官方公式对照
    scores = logits.softmax(dim=-1)
    ref_w, ref_idx = torch.topk(scores, cfg.num_experts_per_tok, dim=-1, sorted=False)
    check('top-k 选择与官方一致',
          torch.all(top_idx == ref_idx) and torch.allclose(top_w, ref_w, atol=1e-6))
    check('权重不归一化（Lite norm_topk_prob=False）',
          not torch.allclose(top_w.sum(-1), torch.ones(6), atol=1e-4))
    # dense 层恒选
    r1 = DeepSeekRouter(cfg, n_routed=1, top_k=1)
    with torch.no_grad():
        w1, i1, _ = r1(x)
    check('dense 单列恒选 0 且权重 1.0',
          torch.all(i1 == 0) and torch.allclose(w1, torch.ones_like(w1)))


def test_loader_mapping():
    print('\n[5] 权重映射（合成 DeepSeek key + 共享拆分）')
    from scripts.load_deepseek_graft import map_deepseek_state_dict
    cfg = make_tiny_config()
    n = cfg.n_layers
    types = list(cfg.layer_types)

    qwen = {}
    for i in range(n):
        pre = f'model.layers.{i}'
        qwen[f'{pre}.input_layernorm.weight'] = torch.randn(cfg.d_model)
        qwen[f'{pre}.post_attention_layernorm.weight'] = torch.randn(cfg.d_model)
        a = f'{pre}.self_attn'
        qwen[f'{a}.q_proj.weight'] = torch.randn(
            cfg.n_heads * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim), cfg.d_model)
        qwen[f'{a}.kv_a_proj_with_mqa.weight'] = torch.randn(
            cfg.kv_lora_rank + cfg.qk_rope_head_dim, cfg.d_model)
        qwen[f'{a}.kv_a_layernorm.weight'] = torch.randn(cfg.kv_lora_rank)
        qwen[f'{a}.kv_b_proj.weight'] = torch.randn(
            cfg.n_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim), cfg.kv_lora_rank)
        qwen[f'{a}.o_proj.weight'] = torch.randn(
            cfg.d_model, cfg.n_heads * cfg.v_head_dim)
        mlp = f'{pre}.mlp'
        if types[i] == 'dense':
            qwen[f'{mlp}.gate_proj.weight'] = torch.randn(cfg.intermediate_size, cfg.d_model)
            qwen[f'{mlp}.up_proj.weight'] = torch.randn(cfg.intermediate_size, cfg.d_model)
            qwen[f'{mlp}.down_proj.weight'] = torch.randn(cfg.d_model, cfg.intermediate_size)
        else:
            qwen[f'{mlp}.gate.weight'] = torch.randn(cfg.n_routed_experts, cfg.d_model)
            for j in range(cfg.n_routed_experts):
                e = f'{mlp}.experts.{j}'
                qwen[f'{e}.gate_proj.weight'] = torch.randn(cfg.moe_intermediate_size, cfg.d_model)
                qwen[f'{e}.up_proj.weight'] = torch.randn(cfg.moe_intermediate_size, cfg.d_model)
                qwen[f'{e}.down_proj.weight'] = torch.randn(cfg.d_model, cfg.moe_intermediate_size)
            sg = torch.randn(cfg.moe_intermediate_size * 2, cfg.d_model)
            su = torch.randn(cfg.moe_intermediate_size * 2, cfg.d_model)
            sd = torch.randn(cfg.d_model, cfg.moe_intermediate_size * 2)
            qwen[f'{mlp}.shared_experts.gate_proj.weight'] = sg
            qwen[f'{mlp}.shared_experts.up_proj.weight'] = su
            qwen[f'{mlp}.shared_experts.down_proj.weight'] = sd
            qwen[f'_shared_ref_{i}'] = (sg, su, sd)
    qwen['model.embed_tokens.weight'] = torch.randn(cfg.vocab_size, cfg.d_model)
    qwen['model.norm.weight'] = torch.randn(cfg.d_model)
    qwen['lm_head.weight'] = torch.randn(cfg.vocab_size, cfg.d_model)

    mapped = map_deepseek_state_dict(qwen, n, types)
    model = ReflexModel(cfg)
    missing, unexpected = model.load_state_dict(mapped, strict=False)

    def _is_backbone(k):
        if k in ('token_embedding.weight', 'lm_head.weight', 'ln_f.weight'):
            return True
        parts = k.split('.')
        if len(parts) >= 3 and parts[0] == 'layers' and parts[1].isdigit():
            li = int(parts[1])
            if parts[2] in ('ln1', 'ln2', 'attention'):
                return True
            # router: 仅 gate_weight 是 DeepSeek 权重（moe 层）
            if (parts[2] == 'router' and len(parts) >= 4
                    and parts[3] == 'gate_weight' and types[li] == 'moe'):
                return True
            if parts[2] == 'all_experts' and types[li] == 'dense':
                if len(parts) >= 5 and parts[4] in ('w_gate', 'w_up', 'w_down'):
                    return True
            if len(parts) >= 5 and parts[2] in ('routed_experts',
                                                'shared_experts'):
                if parts[4] in ('w_gate', 'w_up', 'w_down'):
                    return True
        return False

    bm = [k for k in missing if _is_backbone(k)]
    check('主干 key 全部映射', len(bm) == 0, f'{bm[:5]}')
    check('unexpected 为空', len(unexpected) == 0, f'{unexpected[:5]}')
    # 共享专家合并直拷验证（1:1，不拆分）
    l1 = model.layers[1]
    sg, su, sd = qwen['_shared_ref_1']
    check('共享 gate 直拷（torch.equal）',
          torch.equal(l1.shared_experts[0].w_gate.weight, sg))
    check('共享 down 直拷（torch.equal）',
          torch.equal(l1.shared_experts[0].w_down.weight, sd))
    check('router gate_weight 直拷',
          torch.equal(l1.router.gate_weight, qwen['model.layers.1.mlp.gate.weight']))
    check('RMSNorm 直拷（无 1+w）',
          torch.equal(model.layers[0].ln1.weight,
                      qwen['model.layers.0.input_layernorm.weight']))


def test_checkpoint_roundtrip():
    print('\n[6] checkpoint 存取 + config 恢复')
    cfg = make_tiny_config()
    model = ReflexModel(cfg)
    ck = {'model_state_dict': model.state_dict(), 'config': cfg.__dict__,
          'step': 0, 'phase': 'graft_deepseek-v2-lite'}
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        path = f.name
    torch.save(ck, path)
    ck2 = torch.load(path, map_location='cpu', weights_only=False)
    cfg2 = config_from_checkpoint(ck2)
    check('config 恢复 backbone', cfg2.backbone == 'deepseek_v2')
    check('config 恢复 MoE 参数',
          cfg2.n_routed_experts == cfg.n_routed_experts and
          cfg2.n_shared_experts == cfg.n_shared_experts)
    model2 = ReflexModel(cfg2)
    miss, unexp = model2.load_state_dict(ck2['model_state_dict'], strict=False)
    check('strict 加载通过', len(miss) == 0 and len(unexp) == 0,
          f'missing={len(miss)} unexpected={len(unexp)}')
    os.unlink(path)


def test_bf16_e2e():
    print('\n[7] bf16 E2E（pipeline 对话 + InternalLoop）')
    from loop.internal_loop import InternalLoop
    from interaction.pipeline import ReflexPipeline
    from interaction.manager import InteractionManager

    cfg = make_tiny_config()
    model = ReflexModel(cfg)
    model.to(dtype=torch.bfloat16)
    model.eval()

    class FakeTok:
        eos_token_id = 0
        pad_token = None
        def encode(self, s, add_special_tokens=True, max_length=None, truncation=False):
            return [ord(c) % cfg.vocab_size for c in s][: (max_length or 100000)]
        def __call__(self, s, return_tensors='pt', max_length=None, truncation=False):
            ids = self.encode(s, add_special_tokens=False, max_length=max_length,
                              truncation=truncation)
            return {'input_ids': torch.tensor([ids], dtype=torch.long)}
        def decode(self, ids, skip_special_tokens=True):
            return ''.join(chr(int(i)) for i in ids)
        def apply_chat_template(self, hist, tokenize=False, add_generation_prompt=True):
            return ' '.join(h['content'] for h in hist)
    tok = FakeTok()

    mgr = InteractionManager(cfg)
    pipe = ReflexPipeline(model, cfg)
    pipe.set_tokenizer(tok)
    resp = pipe.process_text('abc')
    print(f'  pipeline 对话 OK, resp len={len(resp)}')
    resp2 = pipe.process_text('defg')
    print(f'  pipeline 对话 2 OK, resp len={len(resp2)}')

    loop = InternalLoop(model, cfg, interaction_mgr=mgr)
    loop._step_count = 0
    for _ in range(3):
        loop._execute_step()
    check('内循环 3 步（loss 有限）',
          loop._loss_int is not None and torch.isfinite(loop._loss_int))
    check('Hebbian 光谱生效（路由三档 + 共享单一 lr）',
          model.layers[1].routed_experts[0].baseline_lr == 1e-7 and
          model.layers[1].routed_experts[-1].baseline_lr == 1e-5 and
          model.layers[1].shared_experts[0].baseline_lr == cfg.shared_expert_lr)


def main():
    torch.manual_seed(42)
    print('=== DeepSeek-V2-Lite 嫁接冒烟测试（CPU 小尺寸） ===')
    test_build_and_forward()
    test_mla_decode_consistency()
    test_shared_merge_equivalence()
    test_router_semantics()
    test_loader_mapping()
    test_checkpoint_roundtrip()
    test_bf16_e2e()
    print(f'\n=== 结果: {PASS} 通过, {FAIL} 失败 ===')
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
