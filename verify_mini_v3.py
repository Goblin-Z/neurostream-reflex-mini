"""
Mini V3 全面架构适配度验证
检查所有核心模块在 d_model=768, d_ff=2304, n_layers=24, 6 experts, top_k=2 下的兼容性
"""
import torch
import sys
import traceback

from config.model_config import ReflexMiniConfig
from core.model import ReflexModel, ReflexMoELayer
from core.expert import Expert
from core.router import Router
from core.attention import MultiHeadAttention
from core.attn_res import AttnResStack
from core.self_model import SelfModel
from core.rmsnorm import RMSNorm
from core.rope import RoPE

config = ReflexMiniConfig()
passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  [PASS] {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name}  {detail}")
        failed += 1

print("="*65)
print("Mini V3 架构适配度验证")
print(f"d_model={config.d_model} d_ff={config.d_ff} n_layers={config.n_layers}")
print(f"n_heads={config.n_heads} n_kv_heads={config.n_kv_heads} head_dim={config.d_model//config.n_heads}")
print(f"experts={config.n_stable}S+{config.n_plastic}P top_k={config.top_k}")
print("="*65)

# ── 1. Config consistency ──
print("\n[1] Config 一致性")
check("d_model % 128 == 0", config.d_model % 128 == 0, f"d_model={config.d_model}")
check("d_ff % 128 == 0", config.d_ff % 128 == 0, f"d_ff={config.d_ff}")
check("n_heads * head_dim == d_model",
      config.n_heads * (config.d_model // config.n_heads) == config.d_model,
      f"{config.n_heads} * {config.d_model//config.n_heads} != {config.d_model}")
check("n_heads % n_kv_heads == 0",
      config.n_heads % config.n_kv_heads == 0,
      f"{config.n_heads} % {config.n_kv_heads}")
check("n_stable + n_plastic == len(expert_baseline_lrs)",
      config.n_stable + config.n_plastic == len(config.expert_baseline_lrs),
      f"{config.n_stable}+{config.n_plastic} != {len(config.expert_baseline_lrs)}")
check("top_k <= n_experts",
      config.top_k <= config.n_stable + config.n_plastic)
check("AttnRes boundaries > 0",
      (config.n_layers - 1) // config.attnres_block_size > 0)
n_boundaries = (config.n_layers - 1) // config.attnres_block_size
check(f"AttnRes boundaries = {n_boundaries} (expect 3)", n_boundaries == 3)

# ── 2. Model build & param count ──
print("\n[2] 模型构建")
model = ReflexModel(config)
total = sum(p.numel() for p in model.parameters())
check(f"Total params ~0.8B (got {total/1e9:.3f}B)", 0.7e9 < total < 0.85e9)
check("n_layers correct", len(model.layers) == config.n_layers)
check("experts per layer correct",
      all(len(layer.all_experts) == config.n_stable + config.n_plastic for layer in model.layers))
check("SelfModel exists", model.self_model is not None)
check("Critic exists", hasattr(model, 'critic') and model.critic is not None)
check("AttnRes exists", model.attn_res is not None)
check("AttnRes n_boundaries", model.attn_res.n_boundaries == n_boundaries)
check("Weight tying", model.lm_head.weight is model.token_embedding.weight)

# ── 3. Attention / GQA ──
print("\n[3] Attention / GQA")
attn = model.layers[0].attention
check("n_heads", attn.n_heads == config.n_heads)
check("n_kv_heads", attn.n_kv_heads == config.n_kv_heads)
check("head_dim", attn.head_dim == config.d_model // config.n_heads)
check("n_rep", attn.n_rep == config.n_heads // config.n_kv_heads)
B, T = 2, 32
x = torch.randn(B, T, config.d_model)
mask = torch.ones(B, T)
attn.train()
out = attn(x, mask)
check("Attention forward shape", out.shape == (B, T, config.d_model), f"got {out.shape}")
check("Attention forward finite", torch.isfinite(out).all().item())
attn.eval()
out2 = attn(x, mask)
check("Attention eval forward", out2.shape == (B, T, config.d_model))

# ── 4. RoPE ──
print("\n[4] RoPE")
check("RoPE head_dim", model.layers[0].attention.rope.head_dim == config.d_model // config.n_heads)
check("RoPE theta", model.layers[0].attention.rope.theta == config.rope_theta)

# ── 5. Expert / SwiGLU ──
print("\n[5] Expert / SwiGLU")
expert = model.layers[0].all_experts[0]
check("w_gate shape", expert.w_gate.weight.shape == (config.d_ff, config.d_model))
check("w_up shape", expert.w_up.weight.shape == (config.d_ff, config.d_model))
check("w_down shape", expert.w_down.weight.shape == (config.d_model, config.d_ff))
check("w_gate bias=False", expert.w_gate.bias is None)
check("w_up bias=False", expert.w_up.bias is None)
check("w_down bias=False", expert.w_down.bias is None)
check("uncertainty_head exists", hasattr(expert, 'uncertainty_head'))
check("query_proj shape", expert.query_proj.weight.shape == (config.d_model, config.d_model))
check("expert.id is string", isinstance(expert.id, str))
# Forward with hebbian buffers
x_exp = torch.randn(4, config.d_model)
out, sigma = expert(x_exp, save_hebbian_buffers=True)
check("Expert forward shape", out.shape == (4, config.d_model))
check("Expert sigma shape", sigma.shape == (4, 1))
check("Expert buffers saved", expert._hidden is not None and expert._input is not None)
check("Expert _hidden dim", expert._hidden.size(-1) == config.d_ff)

# ── 6. Hebbian update ──
print("\n[6] Hebbian update")
from learn.hebbian_update import focal_update
expert2 = model.layers[0].all_experts[0]
expert2(x_exp, save_hebbian_buffers=True)
grad_y = torch.randn(4, config.d_model) * 0.1
w_before = expert2.w_down.weight.data.clone()
focal_update(grad_y, expert2, 1e-4, 1.0, focus_boost=1.0)
w_change = (expert2.w_down.weight.data - w_before).abs().max().item()
check("Hebbian update applied", w_change > 0, f"change={w_change}")
check("Hebbian weights finite", torch.isfinite(expert2.w_down.weight.data).all().item())
# Check grad_h uses pre-update w_down (P1-1 fix)
expert2(x_exp, save_hebbian_buffers=True)
w_gate_before = expert2.w_gate.weight.data.clone()
focal_update(grad_y, expert2, 1e-4, 1.0, focus_boost=1.0)
gate_change = (expert2.w_gate.weight.data - w_gate_before).abs().max().item()
check("Hebbian w_gate updated", gate_change > 0, "w_gate not updated")

# ── 7. Router ──
print("\n[7] Router")
router = model.layers[0].router
check("top_k", router.top_k == config.top_k)
check("n_experts", router.gate_weight.size(1) == config.n_stable + config.n_plastic)
check("gate_weight shape", router.gate_weight.shape == (config.d_model, config.n_stable + config.n_plastic))
check("h_to_bias_weight shape",
      router.h_to_bias_weight.shape == (config.d_model, config.n_stable + config.n_plastic))
x_flat = torch.randn(8, config.d_model)
top_w, top_idx, logits = router(x_flat)
check("Router top_w shape", top_w.shape == (8, config.top_k))
check("Router top_idx shape", top_idx.shape == (8, config.top_k))
check("Router logits shape", logits.shape == (8, config.n_stable + config.n_plastic))
# With h_state
h_state = torch.randn(1, config.d_model)
top_w2, top_idx2, logits2 = router(x_flat, h_state=h_state, is_internal=True)
check("Router with h_state", top_w2.shape == (8, config.top_k))
# Sigma aggregation
sigmas = torch.randn(config.n_stable + config.n_plastic)
sigma_agg = router.aggregate_sigma(sigmas, top_w, top_idx)
check("Router aggregate_sigma returns float", isinstance(sigma_agg, float))

# ── 8. ReflexMoELayer ──
print("\n[8] ReflexMoELayer")
layer = model.layers[0]
check("ln1 dim", layer.ln1.weight.size(0) == config.d_model)
check("ln2 dim", layer.ln2.weight.size(0) == config.d_model)
check("n_stable_experts", len(layer.stable_experts) == config.n_stable)
check("n_plastic_experts", len(layer.plastic_experts) == config.n_plastic)
x_layer = torch.randn(B, T, config.d_model)
mask_layer = torch.ones(B, T)
model.train()
result = layer(x_layer, mask_layer, save_hebbian_buffers=True)
output = result[0]
check("Layer forward shape", output.shape == (B, T, config.d_model))
check("Layer forward finite", torch.isfinite(output).all().item())
# Internal mode
result_int = layer(x_layer, mask_layer, h_state=h_state, is_internal=True, save_hebbian_buffers=True)
check("Layer internal forward", result_int[0].shape == (B, T, config.d_model))

# ── 9. AttnRes ──
print("\n[9] AttnRes")
attnres = model.attn_res
check("n_boundaries", attnres.n_boundaries == n_boundaries)
for i in range(n_boundaries):
    mod = attnres.modules_list[i]
    check(f"Boundary {i} q_proj shape", mod.q_proj.weight.shape == (config.attnres_rank, config.d_model))
    check(f"Boundary {i} k_proj shape", mod.k_proj.weight.shape == (config.attnres_rank, config.d_model))
    check(f"Boundary {i} v_proj shape", mod.v_proj.weight.shape == (config.d_model, config.d_model))
    check(f"Boundary {i} out_proj shape", mod.out_proj.weight.shape == (config.d_model, config.d_model))
    check(f"Boundary {i} post_norm", mod.post_norm.weight.size(0) == config.d_model)

# ── 10. SelfModel ──
print("\n[10] SelfModel")
sm = model.self_model
check("d_model", sm.d_model == config.d_model)
check("z_dim", sm.z_dim == config.self_model_z_dim)
check("GRU input size",
      sm.gru.input_size == config.d_model * 2 + config.self_model_z_dim,
      f"got {sm.gru.input_size}, expect {config.d_model * 2 + config.self_model_z_dim}")
check("GRU hidden size", sm.gru.hidden_size == config.d_model)
# Forward
h = torch.randn(1, config.d_model)
z = torch.randn(1, config.self_model_z_dim)
action = torch.randn(1, config.d_model)
h_next, z_mean, z_logvar = sm(h, z, action)
check("SelfModel h_next shape", h_next.shape == (1, config.d_model))
check("SelfModel z_mean shape", z_mean.shape == (1, config.self_model_z_dim))
check("SelfModel z_logvar shape", z_logvar.shape == (1, config.self_model_z_dim))
check("SelfModel z_logvar finite", torch.isfinite(z_logvar).all().item())
# KL divergence
z_post_mean, z_post_logvar = sm.observe_and_correct(h, torch.randn(1, config.d_model))
kl = sm.kl_divergence(z_post_mean, z_post_logvar, z_mean, z_logvar)
check("KL divergence finite", torch.isfinite(kl).item(), f"kl={kl.item()}")
check("KL divergence >= 0", kl.item() >= 0, f"kl={kl.item()}")
# sample_z
z_sample = sm.sample_z(z_mean, z_logvar)
check("sample_z shape", z_sample.shape == (1, config.self_model_z_dim))
check("sample_z finite", torch.isfinite(z_sample).all().item())
# decode
decoded = sm.decode(z_sample, h_next)
check("decode shape", decoded.shape == (1, config.d_model))
# init_state
h_init, z_init = sm.init_state(device=torch.device('cpu'))
check("init_state h shape", h_init.shape == (1, config.d_model))
check("init_state z shape", z_init.shape == (1, config.self_model_z_dim))

# ── 11. Critic ──
print("\n[11] Critic")
critic = model.critic
v = critic(torch.randn(config.d_model))
check("Critic output scalar", v.shape[-1] == 1, f"got {v.shape}")
v_norm = critic.get_normalized_v(torch.randn(config.d_model))
check("Critic normalized_v is float", isinstance(v_norm, float))

# ── 12. Full forward ──
print("\n[12] Full model forward")
input_ids = torch.randint(0, config.vocab_size, (2, 64))
model.eval()
with torch.no_grad():
    logits = model(input_ids)
check("Forward logits shape", logits.shape == (2, 64, config.vocab_size))
check("Forward finite", torch.isfinite(logits).all().item())
# forward_internal
emb = torch.randn(1, 1, config.d_model)
with torch.no_grad():
    out_int = model.forward_internal(emb.squeeze(0), h_state=h)
check("forward_internal shape", out_int.shape[-1] == config.d_model)

# ── 13. Generate ──
print("\n[13] Generate")
model.eval()
try:
    with torch.no_grad():
        gen = model.generate(input_ids[:1, :8], max_new_tokens=10, temperature=0.8,
                             top_k=config.sampling_top_k, top_p=config.sampling_top_p)
    check("Generate shape", gen.shape[0] == 1)
    check("Generate length >= input", gen.size(1) >= 8)
except Exception as e:
    check("Generate", False, str(e))

# ── 14. Expert helpers ──
print("\n[14] Expert helpers")
all_e = model.get_all_experts()
check("get_all_experts count", len(all_e) == (config.n_stable + config.n_plastic) * config.n_layers)
stable_e = model.get_stable_experts()
check("get_stable_experts count", len(stable_e) == config.n_stable * config.n_layers)
plastic_e = model.get_plastic_experts()
check("get_plastic_experts count", len(plastic_e) == config.n_plastic * config.n_layers)
# focal boost
model.push_focal_boost("test_id", 2.5)
boost = model.pop_focal_boost("test_id")
check("push/pop focal_boost", boost == 2.5)
boost2 = model.pop_focal_boost("test_id")
check("pop cleared focal_boost", boost2 == 1.0)
# aux loss
model.train()
_ = model(input_ids)
aux = model.get_aux_loss()
check("get_aux_loss returns tensor or None", aux is None or isinstance(aux, torch.Tensor))

# ── 15. Loss & backward ──
print("\n[15] Loss & backward")
model.train()
logits = model(input_ids)
labels = torch.randint(0, config.vocab_size, (2, 64))
loss = torch.nn.functional.cross_entropy(
    logits[:, :-1].contiguous().view(-1, config.vocab_size),
    labels[:, 1:].contiguous().view(-1),
)
check("Loss finite", torch.isfinite(loss).item(), f"loss={loss.item()}")
loss.backward()
# Check gradients exist for key params
has_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
check("Gradients exist", has_grad > 0, f"only {has_grad} params have grad")

# ── 16. GradientManager ──
print("\n[16] GradientManager")
from loop.gradient_manager import GradientManager
gm = GradientManager(config.n_layers)
check("GM num_layers", gm.num_layers == config.n_layers)
layers_list = list(gm.iterate_layers(loss, model.layers))
check("GM iterate_layers count", len(layers_list) == config.n_layers)
# Check retain_graph pattern: last layer should have retain=False
check("GM last layer retain=False", layers_list[-1][2] == False)
check("GM first layer retain=True (if >1 layer)", layers_list[0][2] == True if config.n_layers > 1 else True)

# ── 17. Consolidation compatibility ──
print("\n[17] Consolidation")
from learn.fisher import _get_stable_param_names, estimate_fisher, ewc_penalty
stable_names = _get_stable_param_names(model)
check("Fisher stable_names non-empty", len(stable_names) > 0)
check("Fisher stable_names count", len(stable_names) > 0,
      f"got {len(stable_names)} names")
# Quick Fisher estimate (tiny batch)
emb_batch = torch.randn(2, 4, config.d_model)
fisher = estimate_fisher(model, emb_batch, num_samples=2)
check("Fisher estimate non-empty", len(fisher) > 0)
check("Fisher values finite", all(torch.isfinite(f).all().item() for f in fisher.values()))
# EWC penalty
old_params = {name: p.data.clone() for name, p in model.named_parameters() if name in stable_names}
named_stable = [(n, p) for n, p in model.named_parameters() if n in stable_names]
penalty = ewc_penalty(named_stable, fisher, old_params, lambda_ewc=40.0)
check("EWC penalty finite", torch.isfinite(penalty).item() if isinstance(penalty, torch.Tensor) else True)

# ── 18. Internal loop compatibility ──
print("\n[18] InternalLoop compatibility")
from loop.internal_loop import InternalLoop
from interaction.manager import InteractionManager
model._h_state = None
model._z_state = None
interaction_mgr = InteractionManager(config)
try:
    loop = InternalLoop(model, config, interaction_mgr)
    check("InternalLoop init", True)
    check("Loop gradient_mgr num_layers", loop._gradient_mgr.num_layers == config.n_layers)
    check("Loop self_model_optimizer exists", loop._self_model_optimizer is not None)
    check("Loop global_optimizer exists", loop._global_optimizer is not None)
    check("Loop replay_buffer capacity", loop._replay_buffer.capacity == config.replay_capacity)
    check("Loop noise_scheduler exists", loop._noise_scheduler is not None)
except Exception as e:
    check("InternalLoop init", False, str(e))
    traceback.print_exc()

# ── 19. Feedback pipeline ──
print("\n[19] Feedback pipeline")
from interaction.feedback import StructuredFeedback
sf = StructuredFeedback(config)
check("StructuredFeedback init", sf is not None)
check("feedback_alignment_weight loaded", hasattr(sf, 'alignment_weight'))
check("feedback_strength loaded", hasattr(sf, 'strength'))

# ── 20. DialecticalBuffer ──
print("\n[20] DialecticalBuffer")
from loop.dialectical_buffer import DialecticalBuffer
db = DialecticalBuffer(config.d_model, config.endosphere_capacity, config.verify_threshold)
check("DB capacity", db.capacity == config.endosphere_capacity)
v_test = torch.randn(config.d_model)
db.push(v_test, sigma=0.7)
latest = db.get_latest()
check("DB push/get_latest", latest is not None)
check("DB get_latest dim", latest.size(-1) == config.d_model if latest is not None else False)

# ── 21. CriticalNoiseScheduler ──
print("\n[21] CriticalNoiseScheduler")
from learn.critical_noise import CriticalNoiseScheduler
ns = CriticalNoiseScheduler(
    target_sigma=config.internal_entropy_threshold,
    noise_min=config.noise_min, noise_max=config.noise_max,
)
noise = ns.get_noise(0.5)
check("Noise in [min, max]", config.noise_min <= noise <= config.noise_max, f"noise={noise}")

# ── 22. EndoSphereBuffer ──
print("\n[22] EndoSphereBuffer")
from loop.endosphere import EndoSphereBuffer
es = EndoSphereBuffer(config.d_model, config.endosphere_capacity)
check("ES capacity", es.capacity == config.endosphere_capacity)
es.push(v_test)
latest_es = es.get_latest()
check("ES push/get_latest", latest_es is not None)

# ── Summary ──
print("\n" + "="*65)
print(f"RESULTS: {passed} passed, {failed} failed, {passed+failed} total")
if failed == 0:
    print("*** ALL CHECKS PASSED ***")
else:
    print(f"*** {failed} CHECKS FAILED ***")
print("="*65)
