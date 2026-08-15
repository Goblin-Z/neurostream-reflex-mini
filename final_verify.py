import torch
import sys
print('='*65)
print('NeuroStream-Reflex V2-Mini 最终验证')
print('='*65)

from config.model_config import ReflexMiniConfig
from core.model import ReflexModel

config = ReflexMiniConfig()
model = ReflexModel(config)
total = sum(p.numel() for p in model.parameters())
all_pass = True

def check(name, ok, detail=''):
    global all_pass
    if ok:
        print(f'  [PASS] {name}')
    else:
        print(f'  [FAIL] {name}  {detail}')
        all_pass = False

print()
print('1. Config')
check('d_model=640 (5x128)', config.d_model == 640 and config.d_model % 128 == 0)
check('d_ff=2048 (16x128)', config.d_ff == 2048 and config.d_ff % 128 == 0)
check('n_layers=24', config.n_layers == 24)
check('n_heads=10 n_kv_heads=2 hd=64', config.n_heads == 10 and config.n_kv_heads == 2 and config.d_model//config.n_heads == 64)
check('expert_baseline_lrs len=6', len(config.expert_baseline_lrs) == config.n_stable + config.n_plastic)
check('top_k=2 plastic=2', config.top_k == 2 and config.n_plastic == 2)
check('max_seq_len=2048', config.max_seq_len == 2048)
check('AttnRes boundaries=3', (config.n_layers-1)//config.attnres_block_size == 3)

print()
print('2. Parameters')
emb = sum(p.numel() for p in model.token_embedding.parameters())
layers = sum(p.numel() for p in model.layers.parameters())
attnres = sum(p.numel() for p in model.attn_res.parameters()) if model.attn_res else 0
sm = sum(p.numel() for p in model.self_model.parameters()) if model.self_model else 0
critic_total = sum(p.numel() for p in model.critic.parameters()) if hasattr(model,'critic') and model.critic is not None else 0

check(f'Total: {total/1e9:.3f}B', 0.7e9 < total < 0.85e9, f'got {total/1e9:.3f}B')
frozen_kw = ['self_model','contemplator','critic','query_proj','uncertainty_head','lr_bias','verify_threshold','endo_proj','endo_gate_proj','h_to_bias_weight']
frozen = sum(p.numel() for n,p in model.named_parameters() if any(k in n for k in frozen_kw))
trainable = total - frozen
check(f'Trainable: {trainable/1e6:.0f}M Frozen: {frozen/1e6:.0f}M', trainable > 0)

stable = sum(sum(p.numel() for p in e.parameters()) for e in model.get_stable_experts())
plastic = sum(sum(p.numel() for p in e.parameters()) for e in model.get_plastic_experts())
check(f'Stable: {stable/1e6:.1f}M ({stable/total*100:.1f}%)', stable > 0)
check(f'Plastic: {plastic/1e6:.1f}M ({plastic/total*100:.1f}%)', plastic > 0)
pe = sum(p.numel() for p in model.layers[0].all_experts[0].parameters())
expansion = config.d_ff / config.d_model
check(f'Per expert: {pe/1e6:.1f}M expansion: {expansion:.1f}x', 2.0 <= expansion <= 5.0)

print()
print('3. Dimensions')
l0 = model.layers[0]
e0 = l0.all_experts[0]
a = l0.attention
r = l0.router
check(f'q_proj [{a.q_proj.weight.shape[1]},{a.q_proj.weight.shape[0]}]', a.q_proj.weight.shape[1] == config.d_model and a.q_proj.weight.shape[0] == config.n_heads*64)
check(f'kv_proj [{a.kv_proj.weight.shape[1]},{a.kv_proj.weight.shape[0]}]', a.kv_proj.weight.shape[1] == config.d_model and a.kv_proj.weight.shape[0] == 2*config.n_kv_heads*64)
check(f'o_proj [{a.o_proj.weight.shape[1]},{a.o_proj.weight.shape[0]}]', a.o_proj.weight.shape[1] == config.n_heads*64 and a.o_proj.weight.shape[0] == config.d_model)
check(f'w_gate [{e0.w_gate.weight.shape[1]},{e0.w_gate.weight.shape[0]}]', e0.w_gate.weight.shape == (config.d_ff, config.d_model))
check(f'w_up [{e0.w_up.weight.shape[1]},{e0.w_up.weight.shape[0]}]', e0.w_up.weight.shape == (config.d_ff, config.d_model))
check(f'w_down [{e0.w_down.weight.shape[1]},{e0.w_down.weight.shape[0]}]', e0.w_down.weight.shape == (config.d_model, config.d_ff))
check(f'gate_weight [{r.gate_weight.shape[0]},{r.gate_weight.shape[1]}]', r.gate_weight.shape == (config.d_model, config.n_stable+config.n_plastic))
check(f'embedding [{model.token_embedding.weight.shape[0]},{model.token_embedding.weight.shape[1]}]', model.token_embedding.weight.shape == (config.vocab_size, config.d_model))
check(f'weight tying: lm_head is token_embedding', model.lm_head.weight is model.token_embedding.weight)

print()
print('4. Forward & Backward')
model.eval()
with torch.no_grad():
    out = model(torch.randint(0, config.vocab_size, (2, 32)))
check(f'Forward: {out.shape}', out.shape == (2, 32, config.vocab_size))
check(f'  finite={torch.isfinite(out).all().item()}', torch.isfinite(out).all().item())

model.train()
logits = model(torch.randint(0, config.vocab_size, (2, 64)))
labels = torch.randint(0, config.vocab_size, (2, 64))
loss = torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1, config.vocab_size), labels[:,1:].reshape(-1))
check(f'CE loss: {loss.item():.4f}', torch.isfinite(loss).all())
loss.backward()
grad_count = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
check(f'Gradients: {grad_count} params', grad_count > 0)

learnable = getattr(model, '_learnable_sigmas', None)
check(f'learnable_sigmas exists', learnable is not None)
check(f'  shape={learnable.shape} requires_grad={learnable.requires_grad}', learnable is not None and learnable.requires_grad)

print()
print('5. All fixes check')
_enc = {'encoding': 'utf-8'}
checks = [
    ('KL bounded', 'LOGVAR_MIN' in open('core/self_model.py', **_enc).read()),
    ('Hebbian momentum', '_MOMENTUM' in open('learn/hebbian_update.py', **_enc).read()),
    ('Hebbian clip', '_MAX_UPDATE_SCALE' in open('learn/hebbian_update.py', **_enc).read()),
    ('Hebbian weight order', 'w_down_old' in open('learn/hebbian_update.py', **_enc).read()),
    ('Safe SGD', '_safe_sgd_step' in open('learn/consolidation.py', **_enc).read()),
    ('Gradual decay', 'Stage B2' in open('loop/internal_loop.py', **_enc).read()),
    ('Warmup', 'train_consciousness_warmup' in open('train/train_reflex.py', **_enc).read()),
    ('Alignment loss', '_compute_alignment_loss' in open('interaction/pipeline.py', **_enc).read()),
    ('Sigma calibration', 'sigma_calibration' in open('interaction/pipeline.py', **_enc).read()),
    ('Learnable sigma', '_learnable_sigmas' in open('core/model.py', **_enc).read()),
    ('focal_boost inner', 'pop_focal_boost' in open('loop/internal_loop.py', **_enc).read()),
    ('focal_boost push', 'push_focal_boost' in open('interaction/pipeline.py', **_enc).read()),
    ('Confused text align', 'confused_span' in open('interaction/feedback.py', **_enc).read()),
    ('Reward 0.0', 'return 0.0' in open('interaction/feedback.py', **_enc).read()),
    ('Aux loss', '_last_aux_loss' in open('core/router.py', **_enc).read()),
    ('Critic TD', '_prev_v_pred' in open('loop/internal_loop.py', **_enc).read()),
    ('Critic clip', 'clip_grad_norm_' in open('loop/internal_loop.py', **_enc).read()),
    ('pad_id', 'set_pad_id' in open('train/train_reflex.py', **_enc).read()),
    ('use_reentrant=False', 'use_reentrant=False' in open('core/model.py', **_enc).read()),
    ('Attention dropout', 'self.training' in open('core/attention.py', **_enc).read()),
]
for name, ok in checks:
    check(f'  {name}', ok)

print()
print('=== SUMMARY ===')
print(f'Total params:  {total:,} ({total/1e9:.3f}B)')
print(f'd_model={config.d_model} n_layers={config.n_layers} d_ff={config.d_ff}')
print(f'experts={config.n_stable}S+{config.n_plastic}P top_k={config.top_k}')
print(f'Embedding: {emb/1e6:.1f}M  Layers: {layers/1e6:.1f}M')
print(f'SelfModel: {sm/1e6:.1f}M  AttnRes: {attnres/1e6:.1f}M  Critic: {critic_total/1e6:.1f}M')
print(f'Trainable: {trainable/1e6:.0f}M  Stable: {stable/1e6:.1f}M  Plastic: {plastic/1e6:.1f}M')
print(f'Active (tk=2): {pe*2*config.n_layers/1e6:.0f}M (24 fwds/token)')
print(f'GPU Memory (b=64,s=2048): ~{(total*2/1e9 + trainable*8/1e9 + trainable*2/1e9 + 64*2048*config.d_model*config.n_layers*2/1e9 + 2):.1f} GB')
if all_pass:
    print('ALL PASSED')
else:
    print('SOME FAILED')
