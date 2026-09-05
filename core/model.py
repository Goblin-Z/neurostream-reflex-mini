import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from config.model_config import ReflexConfig
from core.expert import Expert
from core.router import Router
from core.deepseek_router import DeepSeekRouter
from core.attention import MultiHeadAttention
from core.rmsnorm import RMSNorm
from core.attn_res import AttnResStack
from core.self_model import SelfModel
from core.memory_bank import MemoryBank
from core.mla import DeepseekMLA
from loop.endosphere import EndoSphereBuffer

import threading


class ReflexMoELayer(nn.Module):
    """
    Single MoE layer: Attention -> Router -> Experts (SwiGLU FFN) -> Residual.

    Uses RMSNorm (not LayerNorm) for Pre-Norm residual connections.
    Attention uses GQA + RoPE + QK-Norm internally.
    Experts use SwiGLU FFN.
    """

    def __init__(self, config: ReflexConfig):
        super().__init__()
        self.config = config
        self.attention = MultiHeadAttention(config)
        self.router = Router(config)
        self.ln1 = RMSNorm(config.d_model)
        self.ln2 = RMSNorm(config.d_model)

        spectrum = list(config.expert_baseline_lrs)
        n_total = config.n_stable + config.n_plastic
        if len(spectrum) < n_total:
            spectrum += [spectrum[-1]] * (n_total - len(spectrum))
        else:
            spectrum = spectrum[:n_total]

        self.all_experts = nn.ModuleList([
            Expert(config.d_model, config.d_ff, config.dropout,
                   baseline_lr=spectrum[i])
            for i in range(n_total)
        ])
        self.stable_experts = nn.ModuleList(
            self.all_experts[:config.n_stable]
        )
        self.plastic_experts = nn.ModuleList(
            self.all_experts[config.n_stable:]
        )

    def forward(self, x, attention_mask=None, h_state=None,
                is_internal=False, save_hebbian_buffers=True,
                mem_kv=None):
        attn_out = self.attention(self.ln1(x), attention_mask, mem_kv=mem_kv)
        x = x + attn_out

        x_norm = self.ln2(x)
        batch, seq, d_model = x_norm.shape
        x_flat = x_norm.view(-1, d_model)

        top_w, top_idx, logits = self.router(
            x_flat, h_state=h_state, is_internal=is_internal
        )

        output = torch.zeros_like(x_flat)
        n_active = len(self.all_experts)
        expert_sigmas = torch.zeros(n_active, device=x_flat.device)
        per_token_sigma = torch.zeros(x_flat.size(0), device=x_flat.device)
        learnable_sigma_list = []

        for i, expert in enumerate(self.all_experts):
            rows, cols = (top_idx == i).nonzero(as_tuple=True)
            if rows.numel() == 0:
                if save_hebbian_buffers:
                    expert.clear_buffers()
                continue

            token_input = x_flat[rows]
            weight = top_w[rows, cols].unsqueeze(-1)
            expert_out, expert_sigma = expert(token_input, save_hebbian_buffers)
            expert_sigmas[i] = expert_sigma.mean().detach()
            learnable_sigma_list.append(expert_sigma.mean())  # differentiable
            per_token_sigma[rows] += (
                weight.squeeze(-1) * expert_sigma.squeeze(-1)
            ).detach()
            output.index_add_(0, rows, weight * expert_out)

        output = output.view(batch, seq, d_model)
        output = x + output

        sigma_agg = self.router.aggregate_sigma(
            expert_sigmas, top_w, top_idx
        )
        per_token_sigma = per_token_sigma.view(batch, seq)

        if learnable_sigma_list:
            self._learnable_sigmas = torch.stack(learnable_sigma_list).mean()
        else:
            self._learnable_sigmas = None

        return (output, top_w, top_idx, logits,
                expert_sigmas, sigma_agg, per_token_sigma)

    def add_expert(self, baseline_lr=None):
        exp = Expert(self.config.d_model, self.config.d_ff,
                     self.config.dropout,
                     baseline_lr=baseline_lr or 1e-5)
        self.plastic_experts.append(exp)
        self.all_experts.append(exp)
        self.router.add_column()
        return exp

    def remove_expert_by_idx(self, idx):
        exp = self.all_experts.pop(idx)
        self._del_from(self.stable_experts, exp)
        self._del_from(self.plastic_experts, exp)
        self.router.remove_column(idx)
        return exp

    def _del_from(self, mod_list, expert):
        for i, e in enumerate(mod_list):
            if e is expert:
                mod_list.pop(i)
                return

    def get_expert_by_id(self, eid):
        for e in self.all_experts:
            if e.id == eid:
                return e
        return None

    def get_plastic_experts(self):
        return list(self.plastic_experts)

    def get_stable_experts(self):
        return list(self.stable_experts)


class DeepSeekGraftLayer(nn.Module):
    """
    DeepSeek-V2-Lite 嫁接层 —— 与 ReflexMoELayer 相同 forward 契约。

    专家数量与 DeepSeek-V2-Lite 对齐：
      - dense 层（层 0）: 1 个专家（FFN intermediate=10944），router 单列恒选
      - moe 层（层 1-26）: 64 个路由专家（top-6）+ 2 个共享专家（拆分自
        DeepSeek 合并共享 MLP intermediate=2816，数学等价；学习率一高一低）

    学习率光谱（用户方案）：64 路由专家三阶梯（16×1e-7 稳定 / 32×1e-6 中间 /
    16×1e-5 可塑）+ 2 共享专家（1e-7 / 1e-5）。光谱仅影响 Hebbian 强度分层，
    不影响前向（前向完全复用 DeepSeek 原权重语义）。

    注意力为 MLA（DeepseekMLA）；sigma/Hebbian/记忆接口与原生层一致。
    """

    def __init__(self, config: ReflexConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        layer_types = getattr(config, 'layer_types', None) or []
        self.layer_type = (layer_types[layer_idx]
                           if layer_idx < len(layer_types) else 'moe')
        self.is_moe = (self.layer_type == 'moe')
        self.is_full_attention = True   # MLA 层可缓存 KV（L4 记忆/增量解码）

        self.ln1 = RMSNorm(config.d_model)
        self.ln2 = RMSNorm(config.d_model)
        self.attention = DeepseekMLA(config)

        if self.is_moe:
            n_routed = config.n_routed_experts
            n_shared = config.n_shared_experts
            spectrum = list(config.expert_baseline_lrs)
            if len(spectrum) < n_routed:
                spectrum += [spectrum[-1]] * (n_routed - len(spectrum))
            self.routed_experts = nn.ModuleList([
                Expert(config.d_model, config.moe_intermediate_size,
                       config.dropout, baseline_lr=spectrum[i])
                for i in range(n_routed)
            ])
            # 共享专家：保持 DeepSeek 合并 MLP 完整性（intermediate × n_shared），
            # 1:1 直拷不拆分；单一学习率（shared_expert_lr）
            self.shared_experts = nn.ModuleList([
                Expert(config.d_model,
                       config.moe_intermediate_size * n_shared,
                       config.dropout,
                       baseline_lr=getattr(config, 'shared_expert_lr', 1e-6))
            ])
            self.router = DeepSeekRouter(config)
            # all_experts 用普通 list（不注册为 ModuleList）：避免与
            # routed_experts/shared_experts 共享对象重复注册——否则
            # state_dict/named_parameters 会出现重复 key（参数计数翻倍、
            # checkpoint 膨胀、global_drift 快照错位）。下游仅遍历使用。
            self.all_experts = (
                list(self.routed_experts) + list(self.shared_experts))
        else:
            self.router = DeepSeekRouter(config, n_routed=1, top_k=1)
            self.all_experts = nn.ModuleList([
                Expert(config.d_model, config.intermediate_size,
                       config.dropout,
                       baseline_lr=getattr(config, 'dense_expert_lr', 1e-7))
            ])
            self.routed_experts = nn.ModuleList()
            self.shared_experts = nn.ModuleList()

        # 稳定/可塑视图（按光谱划分：lr<=1e-6 稳定，>=1e-5 可塑）
        # 用普通 list：避免共享对象重复注册进 ModuleList
        stable, plastic = [], []
        for e in self.all_experts:
            if e.baseline_lr >= 1e-5:
                plastic.append(e)
            else:
                stable.append(e)
        self.stable_experts = stable
        self.plastic_experts = plastic

        self._new_past = None

    def forward(self, x, attention_mask=None, h_state=None,
                is_internal=False, save_hebbian_buffers=True,
                mem_kv=None, past=None, rope_offset=0):
        attn_out = self.attention(self.ln1(x), attention_mask,
                                  mem_kv=mem_kv, rope_offset=rope_offset)
        x = x + attn_out

        x_norm = self.ln2(x)
        batch, seq, d_model = x_norm.shape
        x_flat = x_norm.view(-1, d_model)

        top_w, top_idx, logits = self.router(
            x_flat, bsz=batch, seq_len=seq,
            h_state=h_state, is_internal=is_internal)

        n_total = len(self.all_experts)
        n_routed = len(self.routed_experts)
        output = torch.zeros_like(x_flat)
        expert_sigmas = torch.zeros(n_total, device=x_flat.device)
        per_token_sigma = torch.zeros(x_flat.size(0), device=x_flat.device)
        learnable_sigma_list = []

        if self.is_moe:
            # ── 路由专家（top-k 激活）──
            for i, expert in enumerate(self.routed_experts):
                rows, cols = (top_idx == i).nonzero(as_tuple=True)
                if rows.numel() == 0:
                    if save_hebbian_buffers:
                        expert.clear_buffers()
                    continue
                token_input = x_flat[rows]
                # 路由权重保持 fp32（与 transformers built-in 相同路径：
                # bf16 输出 × fp32 权重 → fp32 → 转回 bf16 再 index_add；
                # 此前先 .to(bf16) 再乘会引入每层 ~1e-3 精度损失，27 层
                # 累积 + 路由分叉 → verify top-1 被拖到 ~78%）
                weight = top_w[rows, cols].unsqueeze(-1)
                expert_out, expert_sigma = expert(
                    token_input, save_hebbian_buffers)
                expert_sigmas[i] = expert_sigma.mean().detach()
                learnable_sigma_list.append(expert_sigma.mean())
                per_token_sigma[rows] += (
                    weight.squeeze(-1).to(x_flat.dtype)
                    * expert_sigma.squeeze(-1)).detach()
                weighted = (expert_out.float() * weight.float()).to(x_flat.dtype)
                output.index_add_(0, rows, weighted)

            # ── 共享专家（全部 token 激活，权重 1.0）──
            for j, expert in enumerate(self.shared_experts):
                s_out, s_sigma = expert(x_flat, save_hebbian_buffers)
                idx = n_routed + j
                expert_sigmas[idx] = s_sigma.mean().detach()
                learnable_sigma_list.append(s_sigma.mean())
                per_token_sigma += s_sigma.squeeze(-1).detach()
                output += s_out
        else:
            # ── dense 层：单专家恒选 ──
            expert = self.all_experts[0]
            e_out, e_sigma = expert(x_flat, save_hebbian_buffers)
            expert_sigmas[0] = e_sigma.mean().detach()
            learnable_sigma_list.append(e_sigma.mean())
            per_token_sigma += e_sigma.squeeze(-1).detach()
            output += e_out

        output = output.view(batch, seq, d_model)
        x = x + output

        # sigma 聚合：路由 top-k 加权 + 共享均值混合（各半）
        routed_agg = self.router.aggregate_sigma(
            expert_sigmas, top_w, top_idx)
        if self.is_moe and n_routed < n_total:
            shared_sig = expert_sigmas[n_routed:].mean().item()
            sigma_agg = 0.5 * routed_agg + 0.5 * shared_sig
        else:
            sigma_agg = routed_agg
        per_token_sigma = per_token_sigma.view(batch, seq)

        if learnable_sigma_list:
            self._learnable_sigmas = torch.stack(learnable_sigma_list).mean()
        else:
            self._learnable_sigmas = None

        return (x, top_w, top_idx, logits,
                expert_sigmas, sigma_agg, per_token_sigma)

    # ── Expert helpers（与 ReflexMoELayer 接口对齐）──

    def get_expert_by_id(self, eid):
        for e in self.all_experts:
            if e.id == eid:
                return e
        return None

    def get_plastic_experts(self):
        return list(self.plastic_experts)

    def get_stable_experts(self):
        return list(self.stable_experts)


class ReflexModel(nn.Module):
    """
    NeuroStream-Reflex: main model with modern Transformer architecture.
    backbone='reflex': 原生 MoE 主干；backbone='deepseek_v2': DeepSeek-V2-Lite 嫁接。
    """

    def __init__(self, config: ReflexConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        self.dropout = nn.Dropout(config.dropout)

        # ── Backbone 分支 ──
        self.backbone = getattr(config, 'backbone', 'reflex')
        if self.backbone == 'deepseek_v2':
            # DeepSeek-V2-Lite：第 0 层 dense，第 1+ 层 MoE
            layer_types = list(getattr(config, 'layer_types', None) or [])
            if len(layer_types) != config.n_layers:
                first_k = getattr(config, 'first_k_dense_replace', 1)
                layer_types = (
                    ['dense'] * first_k
                    + ['moe'] * (config.n_layers - first_k))
            config.layer_types = tuple(layer_types)
            self.layers = nn.ModuleList([
                DeepSeekGraftLayer(config, i) for i in range(config.n_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                ReflexMoELayer(config) for _ in range(config.n_layers)
            ])

        # AttnRes: Block Delta Attention Residuals
        self.attnres_enabled = getattr(config, 'attnres_enabled', True)
        if self.attnres_enabled:
            self.attn_res = AttnResStack(
                n_layers=config.n_layers,
                block_size=getattr(config, 'attnres_block_size', 4),
                d_model=config.d_model,
                rank=getattr(config, 'attnres_rank', 128),
            )
            pn_init = getattr(config, 'attnres_postnorm_init', 0.1)
            for m in self.attn_res.modules_list:
                m.post_norm.weight.data.fill_(pn_init)
        else:
            self.attn_res = None

        self.ln_f = RMSNorm(config.d_model)

        # Weight tying
        self.tie_weights = getattr(config, 'tie_word_embeddings', True)
        if self.tie_weights:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
            self.lm_head.weight = self.token_embedding.weight  # tie
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if getattr(config, 'self_model_enabled', True):
            self.self_model = SelfModel(
                d_model=config.d_model,
                z_dim=config.self_model_z_dim,
                hidden_dim=config.self_model_hidden_dim,
                n_prior_experts=config.self_model_n_prior_experts,
                n_post_experts=config.self_model_n_post_experts,
            )
        else:
            self.self_model = None

        self._h_state = None
        self._z_state = None

        self.endosphere = EndoSphereBuffer(config.d_model, config.endosphere_capacity)
        self.replay_buffer = None  # managed by InternalLoop

        # ── 记忆系统 v4 (L1-L4) ──
        if getattr(config, 'memory_enabled', True):
            head_dim = getattr(config, 'head_dim', 0) or (
                config.d_model // config.n_heads)
            self.memory_bank = MemoryBank(
                d_model=config.d_model,
                capacity=getattr(config, 'memory_bank_capacity', 128),
                top_k=getattr(config, 'memory_context_top_k', 8),
                write_lr=getattr(config, 'memory_write_lr', 0.05),
                kv_rounds=getattr(config, 'kv_cache_rounds', 4),
                num_layers=config.n_layers,
                n_heads=config.n_heads,
                head_dim=head_dim,
            )
        else:
            self.memory_bank = None

        # Forward state (reset each forward)
        self._last_expert_sigmas = None
        self._last_sigma_aggregate = 0.0
        self._last_token_sigmas = None
        self._last_layer_outputs = None
        self._last_input_ids = None

        # Interaction tracking
        self._internal_step_count = 0
        self._lock = threading.RLock()

        self._pending_focal_boosts = {}
        self._focal_boost_lock = threading.Lock()

        # Critic
        if getattr(config, 'critic_enabled', True):
            from learn.critic import ReflexCritic
            self.critic = ReflexCritic(
                config.d_model,
                getattr(config, 'critic_hidden_dim', 256),
            )
            self.critic_optimizer = torch.optim.Adam(
                self.critic.parameters(),
                lr=getattr(config, 'critic_lr', 1e-3),
            )

        self._decode_tokenizer = None

    # ── Forward ──

    def _apply_layers_with_attnres(self, x, attention_mask=None,
                                   h_state=None,
                                   save_hebbian_buffers=True,
                                   is_internal=False, mem_kv=None):
        import torch.utils.checkpoint as chk

        block_size = getattr(self.config, 'attnres_block_size', 4)
        block_outputs = [x]
        boundary_idx = 0
        expert_sigmas = None
        sigma_agg = 0.0
        per_token_sigma = None

        for i, layer in enumerate(self.layers):
            layer_mem = None
            if mem_kv is not None and i in mem_kv:
                layer_mem = mem_kv[i]
            if self.training and torch.is_grad_enabled() and not is_internal:
                def _fn(x, attention_mask, layer=layer,
                        hs=h_state, ii=is_internal, sb=save_hebbian_buffers,
                        mk=layer_mem):
                    return layer(x, attention_mask, h_state=hs,
                                 is_internal=ii, save_hebbian_buffers=sb,
                                 mem_kv=mk)
                x, _, _, _, expert_sigmas, sigma_agg, per_token_sigma = \
                    chk.checkpoint(_fn, x, attention_mask, use_reentrant=False)
            else:
                x, _, _, _, expert_sigmas, sigma_agg, per_token_sigma = \
                    layer(x, attention_mask, h_state=h_state,
                          is_internal=is_internal,
                          save_hebbian_buffers=save_hebbian_buffers,
                          mem_kv=layer_mem)

            if (self.attn_res is not None
                    and (i + 1) % block_size == 0
                    and (i + 1) < len(self.layers)):
                block_outputs.append(x)
                x = self.attn_res.apply(
                    boundary_idx, x, block_outputs,
                    memory_bank=getattr(self, 'memory_bank', None),
                )
                block_outputs = [block_outputs[0], x]
                boundary_idx += 1

        return x, expert_sigmas, sigma_agg, per_token_sigma

    def forward(self, input_ids, attention_mask=None,
                save_hebbian_buffers=True, return_hidden=False,
                mem_kv=None, h_state=None):
        batch, seq = input_ids.shape

        x = self.token_embedding(input_ids)
        x = self.dropout(x)

        x, expert_sigmas, sigma_agg, per_token_sigma = \
            self._apply_layers_with_attnres(
                x, attention_mask, h_state=h_state,
                save_hebbian_buffers=save_hebbian_buffers,
                mem_kv=mem_kv,
            )

        self._last_expert_sigmas = expert_sigmas
        self._last_sigma_aggregate = sigma_agg
        self._last_token_sigmas = per_token_sigma
        self._last_input_ids = input_ids

        learnable_list = [
            getattr(layer, '_learnable_sigmas', None)
            for layer in self.layers
        ]
        learnable_list = [s for s in learnable_list if s is not None]
        if learnable_list:
            self._learnable_sigmas = torch.stack(
                [s.mean() for s in learnable_list]).mean()
        else:
            self._learnable_sigmas = None

        x = self.ln_f(x)
        self._last_layer_outputs = {'hidden_states': x}
        if return_hidden:
            return x
        logits = self.lm_head(x)
        self._last_layer_outputs['logits'] = logits
        return logits

    def forward_embeddings(self, embeddings, attention_mask=None,
                           save_hebbian_buffers=True):
        batch, seq, d = embeddings.shape
        if seq > self.config.max_seq_len:
            embeddings = embeddings[:, -self.config.max_seq_len:, :]
            seq = embeddings.size(1)
        x = self.dropout(embeddings)

        x, _, _, _ = self._apply_layers_with_attnres(
            x, attention_mask,
            save_hebbian_buffers=save_hebbian_buffers,
        )

        x = self.ln_f(x)
        return self.lm_head(x)

    def forward_internal(self, v_t, h_state=None, mem_kv=None):
        if v_t.dim() == 1:
            v_t = v_t.unsqueeze(0).unsqueeze(0)
        elif v_t.dim() == 2:
            v_t = v_t.unsqueeze(1)

        x = v_t

        x, expert_sigmas, sigma_agg, per_token_sigma = \
            self._apply_layers_with_attnres(
                x, None, h_state=h_state,
                is_internal=True,
                mem_kv=mem_kv,
            )

        self._last_expert_sigmas = expert_sigmas
        self._last_sigma_aggregate = sigma_agg
        self._last_token_sigmas = per_token_sigma

        return self.ln_f(x)

    # ── 嫁接专用（backbone='deepseek_v2'，与 Qwen 嫁接同框架）──

    @property
    def kv_layers(self):
        """L4 KV 记忆覆盖的层索引（嫁接层均有 KV；原生 reflex 全部）。"""
        if self.backbone != 'reflex':
            return [i for i, l in enumerate(self.layers)
                    if getattr(l, 'is_full_attention', True)]
        return list(range(len(self.layers)))

    def forward_graft(self, input_ids, attention_mask=None, mem_kv=None,
                      h_state=None, past=None):
        """
        嫁接主干前向（含增量解码 past）。
        past: dict {layer_idx: (k, v)}（展开的 key/value）
        返回 (logits [B, T, V], new_past)。
        """
        x = self.token_embedding(input_ids)
        x = self.dropout(x)

        block_size = getattr(self.config, 'attnres_block_size', 4)
        use_attnres = (self.attn_res is not None
                       and getattr(self.config, 'graft_decode_attnres', False))
        block_outputs = [x]
        boundary_idx = 0
        new_past = {}
        n_layers = len(self.layers)

        for i, layer in enumerate(self.layers):
            if getattr(layer, 'is_full_attention', True):
                kv = mem_kv.get(i) if mem_kv is not None else None
                layer_past = past.get(i) if past is not None else None
                past_len = 0
                if layer_past is not None:
                    past_k, past_v = layer_past
                    past_len = past_k.size(2)
                    if kv is not None:
                        past_k = torch.cat([kv[0], past_k], dim=2)
                        past_v = torch.cat([kv[1], past_v], dim=2)
                    kv = (past_k, past_v)
                layer.attention._kv_cache_enabled = True
                x, _, _, _, _, _, _ = layer(
                    x, attention_mask, h_state=h_state,
                    is_internal=False, save_hebbian_buffers=False,
                    mem_kv=kv, rope_offset=past_len)
                cur_k, cur_v = layer.attention._last_kv
                if layer_past is not None:
                    new_k = torch.cat([layer_past[0], cur_k], dim=2)
                    new_v = torch.cat([layer_past[1], cur_v], dim=2)
                else:
                    new_k, new_v = cur_k, cur_v
                if new_k.size(2) > self.config.max_seq_len:
                    new_k = new_k[:, :, -self.config.max_seq_len:, :]
                    new_v = new_v[:, :, -self.config.max_seq_len:, :]
                new_past[i] = (new_k.detach(), new_v.detach())
            else:
                x, _, _, _, _, _, _ = layer(
                    x, attention_mask, h_state=h_state,
                    is_internal=False, save_hebbian_buffers=False)
                if layer._new_past is not None:
                    new_past[i] = layer._new_past

            if use_attnres and (i + 1) % block_size == 0 and (i + 1) < n_layers:
                block_outputs.append(x)
                x = self.attn_res.apply(
                    boundary_idx, x, block_outputs,
                    memory_bank=getattr(self, 'memory_bank', None))
                block_outputs = [block_outputs[0], x]
                boundary_idx += 1

        x = self.ln_f(x)
        self._last_layer_outputs = {'hidden_states': x}
        logits = self.lm_head(x)
        self._last_layer_outputs['logits'] = logits
        return logits, new_past

    def forward_internal_tail(self, v_t, tail_start, h_state=None, mem_kv=None):
        """嫁接轻量模式：头段 no_grad，尾段建图（Hebbian 只覆盖尾段）。"""
        if v_t.dim() == 1:
            v_t = v_t.unsqueeze(0).unsqueeze(0)
        elif v_t.dim() == 2:
            v_t = v_t.unsqueeze(1)

        x = v_t
        block_size = getattr(self.config, 'attnres_block_size', 4)
        block_outputs = [x]
        boundary_idx = 0
        mem = getattr(self, 'memory_bank', None)
        n_layers = len(self.layers)

        def _run(start, end, grad_enabled, hs, sb):
            nonlocal x, block_outputs, boundary_idx
            with torch.set_grad_enabled(grad_enabled):
                for i in range(start, end):
                    layer = self.layers[i]
                    if getattr(layer, 'is_full_attention', True):
                        x, *_ = layer(x, None, h_state=hs, is_internal=True,
                                      save_hebbian_buffers=sb,
                                      mem_kv=(mem_kv or {}).get(i))
                    else:
                        x, *_ = layer(x, None, h_state=hs, is_internal=True,
                                      save_hebbian_buffers=sb)
                    if (self.attn_res is not None
                            and (i + 1) % block_size == 0
                            and (i + 1) < n_layers):
                        block_outputs.append(x)
                        x = self.attn_res.apply(
                            boundary_idx, x, block_outputs, memory_bank=mem)
                        block_outputs = [block_outputs[0], x]
                        boundary_idx += 1

        _run(0, tail_start, grad_enabled=False, hs=None, sb=False)
        _run(tail_start, n_layers, grad_enabled=True, hs=h_state, sb=True)
        return self.ln_f(x)

    # ── Generation ──

    @torch.no_grad()
    def _sample_next(self, logits, temperature, repetition_penalty,
                     top_k, top_p, input_ids):
        next_logits = logits[:, -1, :] / temperature
        if repetition_penalty != 1.0:
            for i in range(input_ids.size(0)):
                for tid in input_ids[i].unique():
                    v = next_logits[i, tid]
                    next_logits[i, tid] = (v / repetition_penalty
                                           if v >= 0 else v * repetition_penalty)
        if top_k > 0:
            k = min(top_k, next_logits.size(-1))
            kth = torch.topk(next_logits, k, dim=-1).values[:, -1:]
            next_logits[next_logits < kth] = float('-inf')
        if top_p < 1.0:
            sorted_l, sorted_i = torch.sort(next_logits, descending=True)
            cum = torch.cumsum(F.softmax(sorted_l, dim=-1), dim=-1)
            mask = cum > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            for i in range(next_logits.size(0)):
                next_logits[i, sorted_i[i][mask[i]]] = float('-inf')

        probs = F.softmax(next_logits, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0)
        if probs.sum() == 0:
            return logits[:, -1, :].argmax(dim=-1, keepdim=True)
        return torch.multinomial(probs / probs.sum(), 1)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0,
                 attention_mask=None, repetition_penalty=1.5,
                 top_k=40, top_p=0.9, mem_kv=None, h_state=None):
        was_training = self.training
        self.eval()

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        if (getattr(self.config, 'graft_use_past', False)
                and self.backbone != 'reflex'):
            out = self._generate_graft(
                input_ids, max_new_tokens, temperature, attention_mask,
                repetition_penalty, top_k, top_p, mem_kv, h_state)
            if was_training:
                self.train()
            return out

        stop_ids = self._get_stop_ids()
        recent_tokens = []

        for _ in range(max_new_tokens):
            if input_ids.size(1) > self.config.max_seq_len:
                input_ids = input_ids[:, -self.config.max_seq_len:]
                if attention_mask is not None:
                    attention_mask = attention_mask[:, -self.config.max_seq_len:]

            logits = self.forward(input_ids, attention_mask,
                                   save_hebbian_buffers=False,
                                   mem_kv=mem_kv, h_state=h_state)
            if not isinstance(logits, torch.Tensor) or logits.size(1) == 0:
                break

            next_token = self._sample_next(
                logits, temperature, repetition_penalty, top_k, top_p,
                input_ids)

            next_id = next_token.item()
            if next_id in stop_ids:
                break

            recent_tokens.append(next_id)
            if len(recent_tokens) > 40:
                recent_tokens.pop(0)
            if len(recent_tokens) >= 9:
                counts = {}
                for i in range(len(recent_tokens) - 2):
                    k = tuple(recent_tokens[i:i + 3])
                    counts[k] = counts.get(k, 0) + 1
                if any(v >= 4 for v in counts.values()):
                    break

            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, torch.ones(input_ids.size(0), 1,
                                                device=input_ids.device)],
                    dim=-1)

        if was_training:
            self.train()
        return input_ids

    @torch.no_grad()
    def _generate_graft(self, input_ids, max_new_tokens, temperature,
                        attention_mask, repetition_penalty, top_k, top_p,
                        mem_kv, h_state):
        """嫁接增量解码：首步 prefill，之后每步只前向最后 1 token。

        适时结束机制：
          - stop_ids（终止符）→ 立即停止；
          - 三连重复检测（think 闭合后）→ 停止；
          - think 预算：</think> 闭合后最多再生成 max(64, 上限/3)；
          - think 未闭合时终止符宽容 graft_think_eos_grace 次。
        """
        stop_ids = self._get_stop_ids()
        recent_tokens = []
        past = None
        think_closed_step = None
        think_budget = max(64, max_new_tokens // 3)
        tok = self._decode_tokenizer
        gen_debug = getattr(self.config, 'graft_gen_debug', False)
        eos_grace = max(0, int(getattr(self.config, 'graft_think_eos_grace', 0)))
        eos_ignored = 0

        for step in range(max_new_tokens):
            if past is None:
                step_ids = input_ids[:, -self.config.max_seq_len:]
            else:
                step_ids = input_ids[:, -1:]
            logits, past = self.forward_graft(
                step_ids, attention_mask=None, mem_kv=mem_kv,
                h_state=h_state, past=past)
            if not isinstance(logits, torch.Tensor) or logits.size(1) == 0:
                if gen_debug:
                    print(f'[GEN] stop@step{step}: logits 异常', file=sys.stderr)
                break

            next_token = self._sample_next(
                logits, temperature, repetition_penalty, top_k, top_p,
                input_ids)

            next_id = next_token.item()
            if next_id in stop_ids:
                if (think_closed_step is None and eos_ignored < eos_grace):
                    eos_ignored += 1
                    if gen_debug:
                        print(f'[GEN] 忽略 think 未闭合时的终止符 token={next_id}'
                              f'（宽容 {eos_ignored}/{eos_grace}）', file=sys.stderr)
                else:
                    if gen_debug:
                        print(f'[GEN] stop@step{step}: 终止符 token={next_id} '
                              f'(think 未闭合={think_closed_step is None})',
                              file=sys.stderr)
                    break

            recent_tokens.append(next_id)
            if len(recent_tokens) > 40:
                recent_tokens.pop(0)

            if think_closed_step is None and tok is not None:
                try:
                    txt = tok.decode(recent_tokens, skip_special_tokens=False)
                    if '</think>' in txt:
                        think_closed_step = step
                except Exception:
                    pass
            elif think_closed_step is not None and \
                    step - think_closed_step >= think_budget:
                if gen_debug:
                    print(f'[GEN] stop@step{step}: think 预算用尽 '
                          f'({think_budget})', file=sys.stderr)
                break

            if think_closed_step is not None and len(recent_tokens) >= 9:
                counts = {}
                for i in range(len(recent_tokens) - 2):
                    k = tuple(recent_tokens[i:i + 3])
                    counts[k] = counts.get(k, 0) + 1
                if any(v >= 4 for v in counts.values()):
                    if gen_debug:
                        print(f'[GEN] stop@step{step}: 三连重复检测', file=sys.stderr)
                    break

            input_ids = torch.cat([input_ids, next_token], dim=-1)

        return input_ids

    # ── Expert helpers ──

    def push_focal_boost(self, expert_id, boost):
        with self._focal_boost_lock:
            self._pending_focal_boosts[expert_id] = boost

    def pop_focal_boost(self, expert_id, default=1.0):
        with self._focal_boost_lock:
            return self._pending_focal_boosts.pop(expert_id, default)

    def get_all_experts(self):
        exps = []
        for layer in self.layers:
            exps.extend(layer.all_experts)
        return exps

    def get_plastic_experts(self):
        exps = []
        for layer in self.layers:
            exps.extend(layer.get_plastic_experts())
        return exps

    def get_stable_experts(self):
        exps = []
        for layer in self.layers:
            exps.extend(layer.get_stable_experts())
        return exps

    def get_expert_by_id(self, eid):
        for layer in self.layers:
            e = layer.get_expert_by_id(eid)
            if e is not None:
                return e

    def get_aux_loss(self):
        total = None
        for layer in self.layers:
            aux = getattr(layer.router, '_last_aux_loss', None)
            if aux is not None:
                total = aux if total is None else total + aux
        return total

    def set_stable_requires_grad(self, rg):
        for e in self.get_stable_experts():
            for p in e.parameters():
                p.requires_grad = rg

    def set_plastic_requires_grad(self, rg):
        for e in self.get_plastic_experts():
            for p in e.parameters():
                p.requires_grad = rg

    def soft_reset_plastic_experts(self, keep_ratio=0.1):
        for e in self.get_plastic_experts():
            with torch.no_grad():
                for name, param in e.named_parameters():
                    fresh = torch.empty_like(param.data)
                    if param.dim() >= 2:
                        nn.init.xavier_uniform_(fresh)
                    else:
                        nn.init.zeros_(fresh)
                    param.data.mul_(keep_ratio).add_(fresh, alpha=1.0 - keep_ratio)
            e.clear_buffers()

    # ── Tokenizer / verification helpers ──

    def _get_stop_ids(self):
        if hasattr(self, '_cached_stop_ids') and self._cached_stop_ids is not None:
            return self._cached_stop_ids

        ids = set()
        tok = self._decode_tokenizer
        if tok is None:
            return ids
        if hasattr(tok, 'eos_token_id') and tok.eos_token_id is not None:
            ids.add(tok.eos_token_id)
        for marker in ('<|im_end|>', '\uff5cUser\uff5c', '|User|', '<｜User｜>',
                       '<|User|>', '｜User｜', '<｜end▁of▁sentence｜>'):
            try:
                encoded = tok.encode(marker, add_special_tokens=False)
                if len(encoded) == 1:
                    ids.add(encoded[0])
            except Exception:
                pass
        self._cached_stop_ids = ids
        return ids
