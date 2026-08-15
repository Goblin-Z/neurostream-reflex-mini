import torch
import torch.nn as nn
import torch.nn.functional as F

from config.model_config import ReflexConfig
from core.expert import Expert
from core.router import Router
from core.attention import MultiHeadAttention
from core.rmsnorm import RMSNorm
from core.attn_res import AttnResStack
from core.self_model import SelfModel
from core.memory_bank import MemoryBank
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
        # Differentiable sigma for calibration training (not detached)。
        # 修复：不再用 index_put_ 写零张量（in-place 断图），
        # 收集每专家 mean 后 stack——calibration 梯度真实回流 uncertainty_head。
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

        # Store learnable sigma for calibration loss
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


class ReflexModel(nn.Module):
    """
    NeuroStream-Reflex: main model with modern Transformer architecture.

    Architecture upgrades (v2):
      - GQA (Grouped Query Attention) for memory-efficient inference
      - RoPE (Rotary Position Embedding) replacing absolute position embedding
      - QK-Norm (RMSNorm on Q and K) for training stability
      - SwiGLU FFN in experts (better than GELU)
      - RMSNorm everywhere (faster than LayerNorm)
      - Block Delta Attention Residuals (Kimi AttnRes + Delta improvement)
      - Weight tying (lm_head = token_embedding)
      - Flash Attention via F.scaled_dot_product_attention
    """

    def __init__(self, config: ReflexConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        self.dropout = nn.Dropout(config.dropout)

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
            # post_norm 放大（记忆微调前提）：AttnRes/记忆 source 真实影响输出
            pn_init = getattr(config, 'attnres_postnorm_init', 0.1)
            for m in self.attn_res.modules_list:
                m.post_norm.weight.data.fill_(pn_init)
        else:
            self.attn_res = None

        self.ln_f = RMSNorm(config.d_model)

        # Weight tying: lm_head shares weights with token_embedding
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
            self.memory_bank = MemoryBank(
                d_model=config.d_model,
                capacity=getattr(config, 'memory_bank_capacity', 128),
                top_k=getattr(config, 'memory_context_top_k', 8),
                write_lr=getattr(config, 'memory_write_lr', 0.05),
                kv_rounds=getattr(config, 'kv_cache_rounds', 4),
                num_layers=config.n_layers,
                n_heads=config.n_heads,
                head_dim=config.d_model // config.n_heads,
            )
        else:
            self.memory_bank = None

        # Forward state (reset each forward)
        self._last_expert_sigmas = None
        self._last_sigma_aggregate = 0.0
        self._last_token_sigmas = None
        self._last_layer_outputs = None
        self._last_input_ids = None

        # Interaction tracking (written by internal loop, read by pipeline)
        self._internal_step_count = 0
        self._lock = threading.RLock()

        # Pending focal boosts from external feedback (expert_id -> boost).
        # Written by the external loop (main thread), consumed and cleared by
        # the internal loop (background thread) during Stage B Hebbian updates.
        self._pending_focal_boosts = {}
        self._focal_boost_lock = threading.Lock()

        # Critic (local ReflexCritic)
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

        # Tokenizer reference
        self._decode_tokenizer = None

    # ── Forward ──

    def _apply_layers_with_attnres(self, x, attention_mask=None,
                                    h_state=None,
                                    save_hebbian_buffers=True,
                                    is_internal=False, mem_kv=None):
        """
        Run through all layers with AttnRes at block boundaries.
        h_state (from SelfModel) is passed to each layer's Router for
        state-conditioned expert selection.
        mem_kv: L4 记忆 KV（dict: layer_idx -> (mem_k, mem_v)）或 None
        """
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
            # Skip checkpointing for internal forward (is_internal=True):
            # 1. Internal forward processes only 1 token (batch=1, seq=1) - no memory pressure
            # 2. Hebbian update (Stage B) needs autograd.grad(loss, expert._output)
            #    which requires the full computation graph - checkpointing breaks this
            # 3. h_to_bias_weight gradient flow requires non-detached inputs
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
                # L2: 语义记忆作为 AttnRes 额外 source（模型自检索）
                # 内部循环同样启用——意识流也能联想知识
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
        """
        return_hidden=True: skip lm_head, return [B,T,d] (saves 15GB for large vocab)
        mem_kv: L4 记忆 KV（dict: layer_idx -> (mem_k, mem_v)）或 None
        h_state: 内循环状态（外循环一致化——生成也"带着想法"，
                 经 Router 状态门控影响专家选择）
        """
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

        # Aggregate learnable sigmas from all layers for calibration.
        # C2 fix: 不再跨层 torch.stack（split/prune 后各层专家数不同会崩）。
        # 每层先 mean 成标量，再整体 mean——任何专家数配置都安全，
        # 且 sigma 校准目标也是标量（tanh(ce)），语义一致。
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
        # RoPE: no position embedding to add
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

    # ── Generation ──

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0,
                 attention_mask=None, repetition_penalty=1.5,
                 top_k=40, top_p=0.9, mem_kv=None, h_state=None):
        was_training = self.training
        self.eval()

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

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
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(probs / probs.sum(), 1)

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

    # ── Expert helpers ──

    def push_focal_boost(self, expert_id, boost):
        """External loop writes a focal boost for the internal loop to consume."""
        with self._focal_boost_lock:
            self._pending_focal_boosts[expert_id] = boost

    def pop_focal_boost(self, expert_id, default=1.0):
        """Internal loop reads and clears a focal boost for an expert."""
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
        """Aggregate load-balancing auxiliary loss from all layer routers."""
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
        """
        Build stop token IDs for generation.

        Only include single-token markers to avoid false positives.
        Multi-token markers (like <|User|>) are not added as individual
        tokens because common tokens like '|' or 'User' would trigger
        premature stopping.
        """
        if hasattr(self, '_cached_stop_ids') and self._cached_stop_ids is not None:
            return self._cached_stop_ids

        ids = set()
        tok = self._decode_tokenizer
        if tok is None:
            return ids
        if hasattr(tok, 'eos_token_id') and tok.eos_token_id is not None:
            ids.add(tok.eos_token_id)
        # Only add markers that encode to a SINGLE token
        for marker in ('\uff5cUser\uff5c', '|User|', '<｜User｜>', '<|User|>',
                       '｜User｜', '<｜end▁of▁sentence｜>'):
            try:
                encoded = tok.encode(marker, add_special_tokens=False)
                if len(encoded) == 1:
                    ids.add(encoded[0])
            except Exception:
                pass
        self._cached_stop_ids = ids
        return ids
