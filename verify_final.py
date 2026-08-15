import torch
from config.model_config import ReflexMiniConfig
from core.model import ReflexModel
from train.train_reflex import TrainConfig

config = ReflexMiniConfig()
model = ReflexModel(config)
total = sum(p.numel() for p in model.parameters())
tc = TrainConfig(mode='full')

frozen_kw = ['self_model','contemplator','critic','query_proj','uncertainty_head','lr_bias','verify_threshold','endo_proj','endo_gate_proj','h_to_bias_weight']
frozen = sum(p.numel() for n,p in model.named_parameters() if any(k in n for k in frozen_kw))
trainable = total - frozen

print('='*60)
print('1. Architecture Compatibility')
print('='*60)
checks = [
    ('d_model % 128 == 0', config.d_model % 128 == 0),
    ('d_ff % 128 == 0', config.d_ff % 128 == 0),
    ('n_heads * head_dim == d_model', config.n_heads * (config.d_model//config.n_heads) == config.d_model),
    ('n_heads % n_kv_heads == 0', config.n_heads % config.n_kv_heads == 0),
    ('expert_baseline_lrs length', len(config.expert_baseline_lrs) == config.n_stable + config.n_plastic),
    ('top_k <= n_experts', config.top_k <= config.n_stable + config.n_plastic),
    ('max_seq_len == 2048', config.max_seq_len == 2048),
    ('AttnRes boundaries == 3', (config.n_layers-1)//config.attnres_block_size == 3),
]
all_pass = True
for name, ok in checks:
    status = 'PASS' if ok else 'FAIL'
    if not ok: all_pass = False
    print(f'  [{status}] {name}')

dm = config.d_model
print(f'  Attention: q[{dm},{dm}] kv[{dm},{dm//5*2}] o[{dm},{dm}] head_dim={dm//config.n_heads}')
print(f'  Expert: gate[{dm},{config.d_ff}] up[{dm},{config.d_ff}] down[{config.d_ff},{dm}]')
print(f'  SelfModel GRU: input={2*dm+128} hidden={dm}')
print(f'  Embedding: [{config.vocab_size},{dm}]')

print()
print('='*60)
print('2. Parameters & Memory (32GB GPU)')
print('='*60)
w = total * 2 / 1e9
opt = trainable * 8 / 1e9
g = trainable * 2 / 1e9
oh = 2.0
fixed = w + opt + g + oh

b, s = tc.pretrain_batch_size, tc.max_seq_len
li = b * s * dm * config.n_layers * 2 / 1e9
pr = b * s * (dm + config.top_k * config.d_ff) * 2 * 3 / 1e9
fa = b * config.n_layers * s * dm * 2 / 1e9 * 0.1
act = li + pr + fa
total_mem = fixed + act

print(f'  Total params: {total/1e9:.3f}B (trainable={trainable/1e6:.0f}M, frozen={frozen/1e6:.0f}M)')
print(f'  Weights (bf16):    {w:.2f} GB')
print(f'  AdamW (fp32):      {opt:.2f} GB')
print(f'  Gradients (bf16):  {g:.2f} GB')
print(f'  Overhead:          {oh:.2f} GB')
print(f'  Fixed:             {fixed:.2f} GB')
print(f'  Activations (b={b},s={s}): {act:.2f} GB')
print(f'  Total:             {total_mem:.2f} GB / 32 GB (free={32-total_mem:.2f} GB)')

print()
print('='*60)
print('3. Training Config')
print('='*60)
pt_tokens = tc.pretrain_steps * tc.pretrain_batch_size * tc.pretrain_grad_accum * tc.max_seq_len
print(f'  Pretrain: {tc.pretrain_steps} steps, b={tc.pretrain_batch_size}x{tc.pretrain_grad_accum}, s={tc.max_seq_len}')
print(f'    tokens/step={tc.pretrain_batch_size*tc.pretrain_grad_accum*tc.max_seq_len}')
print(f'    total={pt_tokens/1e9:.1f}B, Chinchilla={pt_tokens/(20*total)*100:.0f}%')
sft_ep = tc.sft_steps * tc.sft_batch_size * tc.sft_grad_accum / 1225000
print(f'  SFT: {tc.sft_steps} steps, b={tc.sft_batch_size}x{tc.sft_grad_accum}, epochs={sft_ep:.1f}')
print(f'  Distill: {tc.distill_steps} steps, b={tc.distill_batch_size}x{tc.distill_grad_accum}')
print()
if all_pass:
    print('ALL CHECKS PASSED')
else:
    print('SOME CHECKS FAILED')
