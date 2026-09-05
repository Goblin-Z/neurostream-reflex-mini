import time
import math
import sys
import threading
import torch
import torch.nn.functional as F

from config.model_config import ReflexConfig
from loop.dialectical_buffer import DialecticalBuffer
from loop.gradient_manager import GradientManager
from interaction.manager import InteractionManager
from learn.hebbian_update import focal_update
from learn.replay import PriorityReplayBuffer
from learn.critical_noise import CriticalNoiseScheduler
from learn.confusion_map import ConfusionMap
from learn.fluid_roles import FluidExpertRoles
from improve.architecture import ArchitectureModifier


class InternalLoop:
    """
    Event-driven internal learning loop — v2 with five philosophical upgrades:

    1. Self-regulated critical noise (replaces annealed schedule)
       → System maintains itself at the edge of chaos via sigma feedback.

    2. Dialectical memory buffer (replaces flat deque)
       → States classified into theses/antitheses/syntheses.
       The system thinks most about what it's confused about.

    3. Emergent question timing (replaces hardcoded cooldown)
       → No fixed step counter. The model asks when sigma is genuinely high,
       and stops asking when learning brings sigma down.

    4. Fluid expert roles (replaces fixed stable/plastic spectrum)
       → Experts that are consistently certain become more stable;
       experts that are consistently confused become more plastic.

    5. Concept-level confusion map (replaces token-only tracking)
       → The model tracks recurring knowledge gaps across turns.
    """

    def __init__(self, model, config: ReflexConfig = None,
                 interaction_mgr: InteractionManager = None):
        self.model = model
        self.config = config or model.config
        self.interaction = interaction_mgr or InteractionManager(config)

        self._event = threading.Event()
        self._event.set()
        self._running = True
        self._step_done = threading.Event()
        self._step_done.set()  # initially "no step in progress"

        self._step_count = 0
        self._total_steps = 0
        self._loss_history = []
        self._stats_lock = threading.Lock()  # 保护 _loss_history 跨线程读写（P2-1）

        self._gradient_mgr = GradientManager(config.n_layers)
        self._replay_buffer = PriorityReplayBuffer(
            config.replay_capacity, config.d_model
        )

        # ── Upgrade 1: Critical noise scheduler ──
        self._noise_scheduler = CriticalNoiseScheduler(
            target_sigma=config.internal_entropy_threshold,
            noise_min=config.noise_min,
            noise_max=config.noise_max,
        )

        # ── 记忆系统 v4: 内部循环的记忆 KV（缓存，轮次变化时重建）──
        self._mem_kv = None
        self._mem_kv_n = -1

        # ── Upgrade 2: Dialectical buffer (replaces EndoSphereBuffer) ──
        if hasattr(model, 'endosphere'):
            # Replace the model's flat buffer with dialectical one
            old_buf = model.endosphere
            model.endosphere = DialecticalBuffer(
                config.d_model, config.endosphere_capacity,
                config.verify_threshold
            )
            # Migrate any sequence data
            if hasattr(old_buf, 'seq_buffer'):
                model.endosphere._seq_buffer = old_buf.seq_buffer

        # ── Upgrade 4: Fluid expert roles ──
        self._fluid_roles = FluidExpertRoles(config)

        # ── Upgrade 5: Confusion map ──
        self._confusion_map = ConfusionMap()

        # ── Stage K: Architecture self-modification ──
        self._arch_modifier = ArchitectureModifier(config)

        # ── v2: SelfModel state ──
        self._h_t = None
        self._z_t = None
        self._kl_value = 0.0
        self._curiosity_beta = getattr(config, 'curiosity_beta', 0.1)
        self._imagination_lambda = getattr(config, 'imagination_lambda', 1.0)
        self._stability_lambda = getattr(config, 'stability_lambda', 0.01)

        # SelfModel optimizer
        if (getattr(self.model, 'self_model', None) is not None
                and getattr(config, 'self_model_enabled', True)):
            self._self_model_optimizer = torch.optim.AdamW(
                self.model.self_model.parameters(),
                lr=getattr(config, 'cont_lr', 1e-4),
                weight_decay=0.01,
            )
        else:
            self._self_model_optimizer = None

        self._v_t = None
        self._u_next_input = None
        self._loss_int = None
        self._last_sigma = 0.5
        self._internal_sigma = 0.5
        self._prev_v_pred = None

        # ── Global optimizer: 事后回放全局巩固 ──
        # 更新 Attention + Router + LM Head + RMSNorm + AttnRes
        # Expert 权重由 Hebbian 局部更新，Embedding 冻结
        # lr=1e-6，比 Hebbian 最快 plastic expert (1e-5) 慢 10x
        # 嫁接模式（graft_freeze_backbone）：主干 27B 无法承载 AdamW 状态，
        # 全局优化器只覆盖 Reflex 附加件（router/attn_res/memory_bank/
        # uncertainty_head/ln/q_norm/k_norm）。
        global_params = []
        if (getattr(config, 'graft_freeze_backbone', False)
                and getattr(config, 'backbone', '') != 'reflex'):
            for name, p in model.named_parameters():
                if any(k in name for k in
                       ['router', 'attn_res', 'post_norm', 'memory_bank',
                        'uncertainty_head', 'ln1', 'ln2', 'ln_f',
                        'q_norm', 'k_norm']):
                    if p.requires_grad:
                        global_params.append(p)
        else:
            for name, p in model.named_parameters():
                if any(k in name for k in
                       ['attention', 'router', 'lm_head',
                        'ln_f', 'ln1', 'ln2',
                        'q_norm', 'k_norm', 'q_proj', 'kv_proj', 'o_proj',
                        'attn_res', 'post_norm', 'memory_bank']):
                    if p.requires_grad:
                        global_params.append(p)
        self._global_optimizer = torch.optim.AdamW(
            global_params, lr=1e-6, weight_decay=0.01,
            betas=(0.9, 0.95),
        ) if global_params else None
        self._global_param_count = len(global_params)

        # ── Sigma 在线校准（graft_sigma_cal）：独立优化器只更新尾层
        # uncertainty_head——修复"sigma 头随机初始化且无训练路径"（审计 P0-1），
        # 使 sigma 学习反映 loss_int 不确定度，主动求证才有触发基础 ──
        self._sigma_optimizer = None
        if (getattr(config, 'graft_sigma_cal', False)
                and getattr(config, 'backbone', '') != 'reflex'):
            sigma_params = []
            k = getattr(config, 'graft_hebbian_layers', 8)
            for layer in model.layers[-k:]:
                for e in layer.all_experts:
                    sigma_params.extend(
                        p for p in e.uncertainty_head.parameters()
                        if p.requires_grad)
            if sigma_params:
                self._sigma_optimizer = torch.optim.AdamW(
                    sigma_params, lr=1e-4, weight_decay=0.01)
                print(f'[INFO] sigma 在线校准已启用（{len(sigma_params)} 个参数, '
                      f'每 {getattr(config, "graft_sigma_cal_interval", 20)} 步）')

        # 快照全局参数初始值（用于 stats 显示变化量）
        # 同时保存参数列表引用：global_drift 与快照按同一列表 zip 一一对应，
        # 避免按 named_parameters() 重遍历时顺序错位
        # （曾出现 2816(共享专家 RMSNorm) vs 1408(路由专家 RMSNorm) 崩溃）
        self._global_params = list(global_params)
        self._global_snapshot = None
        if self._global_params:
            self._global_snapshot = [p.data.clone() for p in self._global_params]

    # ── Public API ────────────────────────────────────────────────

    def start(self):
        self._running = True
        self._event.set()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def pause(self):
        """Pause the loop.  Does not return until the current step
        has completed, ensuring no model access race with the pipeline."""
        self._event.clear()
        # Wait for the current step to fully complete (all stages A-K),
        # not just the lock-protected portion.
        self._step_done.wait()
        # Additionally acquire the model lock to synchronise with any
        # lingering lock-holding section.
        if hasattr(self.model, '_lock'):
            with self.model._lock:
                pass

    def resume(self):
        self._event.set()

    def stop(self, timeout=5.0):
        self._event.set()
        self._running = False
        if hasattr(self, '_thread') and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    @property
    def step_count(self):
        return self._total_steps

    @property
    def confusion_map(self):
        return self._confusion_map

    @property
    def noise_scheduler(self):
        return self._noise_scheduler

    @property
    def dialectical_stats(self):
        if hasattr(self.model.endosphere, 'get_dialectical_stats'):
            return self.model.endosphere.get_dialectical_stats()
        return {}

    @property
    def global_drift(self):
        """全局参数相对初始值的最大变化量（用于 stats 显示）。

        快照与参数按同一列表 zip 一一对应（防御结构/顺序变化；
        曾因 2816 vs 1408 快照错位崩溃）。
        """
        if self._global_optimizer is None or self._global_snapshot is None:
            return 0.0
        max_drift = 0.0
        with self.model._lock:
            for p, s in zip(self._global_params, self._global_snapshot):
                if p.data.shape == s.shape:
                    d = (p.data - s).abs().max().item()
                    if d > max_drift:
                        max_drift = d
        return max_drift

    @property
    def global_param_count(self):
        return self._global_param_count

    @property
    def h_to_bias_drift(self):
        """Router 状态门控（h_to_bias_weight）当前最大绝对值——观测其是否随
        在线训练逐步生效（v4 §七·八：从"无影响"到"带想法"）。"""
        m = 0.0
        with self.model._lock:
            for layer in self.model.layers:
                v = layer.router.h_to_bias_weight.abs().max().item()
                if v > m:
                    m = v
        return m

    @property
    def hebbian_drift(self):
        """Hebbian 对主干（专家 FFN 权重）的累计修改量（观测仪表）。

        与 global_drift（Reflex 附加件漂移）互补：本值是"部署即学习"
        直接改写 Qwen 原始权重的量级指标（focal_update 内累加）。
        """
        m = 0.0
        with self.model._lock:
            for e in self.model.get_all_experts():
                v = getattr(e, '_hebbian_drift', 0.0)
                if v > m:
                    m = v
        return m

    # ── Internal loop ─────────────────────────────────────────────

    def _loop(self):
        # 忙循环限速：每步后休眠 internal_step_delay_ms（默认 5ms），
        # 保持"持续思考"同时避免无节流烧算力（internal_steps_per_cycle 旧语义废弃）
        delay = max(0.0, getattr(self.config, 'internal_step_delay_ms', 5.0)) / 1000.0
        while self._running:
            self._event.wait()
            try:
                self._execute_step()
                self._total_steps += 1
                self.model._internal_step_count = self._total_steps
            except Exception as e:
                import traceback
                with self._stats_lock:
                    self._loss_history.append({'error': str(e)})
                # 打印完整 traceback 到 stderr（部署诊断用）
                print(f'\n[INTERNAL LOOP ERROR step={self._total_steps}] '
                      f'{type(e).__name__}: {e}', file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                print('', file=sys.stderr)
            if delay > 0:
                time.sleep(delay)

    def _execute_step(self):
        self._step_done.clear()
        try:
            self._execute_step_inner()
        finally:
            self._step_done.set()

    def _execute_step_inner(self):
        # ── Stage A: Forward with self-regulated noise ──
        v_t = self._get_state()
        if v_t is None:
            v_t = self._init_state()

        device = next(self.model.parameters()).device
        # dtype 与模型一致（嫁接模式模型为 bf16；endosphere/初始状态可能是 fp32）
        model_dtype = next(self.model.parameters()).dtype
        v_t = v_t.to(device=device, dtype=model_dtype)

        # v2: noise driven by KL curiosity (if available), fallback to sigma
        if self._kl_value > 0:
            sigma_for_noise = min(self._kl_value / max(self._curiosity_beta, 0.01), 1.0)
        else:
            sigma_for_noise = self._last_sigma
        noise = self._noise_scheduler.get_noise(sigma_for_noise)
        if noise > 0:
            v_t = v_t + noise * torch.randn_like(v_t)

        with self.model._lock:
            self.model.train()

            # Sync SelfModel state from model
            self._h_t = getattr(self.model, '_h_state', None)
            self._z_t = getattr(self.model, '_z_state', None)

            # SelfModel imagination: h_t, z_t, action → z_prior
            sm = getattr(self.model, 'self_model', None)
            if sm is not None and self._h_t is not None and self._z_t is not None:
                h_t = self._h_t.to(device)
                z_t = self._z_t.to(device)
                action = v_t.unsqueeze(0) if v_t.dim() == 1 else v_t
                if action.dim() == 1:
                    action = action.unsqueeze(0)

                h_next, z_mean, z_logvar = sm(h_t, z_t, action)
                self._z_prior_mean = z_mean
                self._z_prior_logvar = z_logvar
                z_sample = sm.sample_z(z_mean, z_logvar)
                u_next = sm.decode(z_sample, h_next).squeeze(0)

                self.model._h_state = h_next.detach()
                self.model._z_state = z_sample.detach()
                self._h_t = h_next.detach()
                self._z_t = z_sample.detach()
            else:
                u_next = v_t
                h_next = None

            u_next_input = self._ema_blend(u_next, v_t)

            mem_kv = self._get_internal_mem_kv()
            if (getattr(self.config, 'graft_lite', False)
                    and getattr(self.config, 'backbone', '') != 'reflex'):
                # 嫁接轻量模式：头段 no_grad + 尾段建图（Hebbian 只覆盖尾层）
                tail_start = self.config.n_layers - getattr(
                    self.config, 'graft_hebbian_layers', 8)
                pred_emb = self.model.forward_internal_tail(
                    u_next_input, tail_start, h_state=h_next, mem_kv=mem_kv)
            else:
                pred_emb = self.model.forward_internal(
                    u_next_input, h_state=h_next, mem_kv=mem_kv)
            loss_int = self._compute_loss(pred_emb, v_t)

            self._v_t = v_t
            self._u_next_input = u_next_input
            self._loss_int = loss_int
            self._last_sigma = getattr(
                self.model, '_last_sigma_aggregate', 0.5
            )
            self._internal_sigma = self._last_sigma

            # ── Stage A2: sigma 在线校准（可选，graft_sigma_cal）──
            # 必须在 Stage B 之前（loss_int 的计算图还完整）
            self._calibrate_sigma()

            gamma = self._sigma_gamma()

            # ── Stage B: Per-layer gradient-managed Hebbian update ──
            # 嫁接轻量模式：只对最后 graft_hebbian_layers 层做 Hebbian
            hebb_layers = self.model.layers
            if (getattr(self.config, 'graft_lite', False)
                    and getattr(self.config, 'backbone', '') != 'reflex'):
                k = getattr(self.config, 'graft_hebbian_layers', 8)
                hebb_layers = self.model.layers[-k:]
            for layer_idx, layer, retain in self._gradient_mgr.iterate_layers(
                    loss_int, hebb_layers
            ):
                for expert, grad_y in self._gradient_mgr.compute_expert_gradients(
                        loss_int, layer, retain
                ):
                    boost = self.model.pop_focal_boost(expert.id, 1.0)
                    focal_update(grad_y, expert, expert.effective_lr, gamma,
                                 focus_boost=boost)

            # ── Stage B2: Gradual plastic decay (replaces abrupt soft_reset) ──
            # Continuous gentle regularization prevents unbounded weight growth
            # without the catastrophic amnesia of one-shot reset.
            reg = getattr(self.config, 'plastic_reg_strength', 1e-6)
            if reg > 0:
                with torch.no_grad():
                    for expert in self.model.get_plastic_experts():
                        for param in expert.parameters():
                            param.data.mul_(1.0 - reg)

            # ── Stage C: SelfModel update (replaces Contemplator) ──
            self._update_self_model(u_next_input)

            # ── Stage C2: Critic training ──
            self._train_critic()

            # ── Stage E: Mini-consolidation ──
            if (self._step_count > 0 and self._step_count % 50 == 0
                    and not getattr(self.config,
                                    'graft_disable_consolidation', False)):
                self._mini_consolidate()

            # ── Stage F: Major consolidation ──
            if (self._step_count > 0 and self._step_count % 500 == 0
                    and not getattr(self.config,
                                    'graft_disable_consolidation', False)):
                self._major_consolidate()

            # ── Stage K: Architecture self-modification (every 200 steps) ──
            # Runs INSIDE model._lock because it mutates model.layers
            # (append/remove/replace experts), which would race with
            # concurrent forward passes in the inference thread.
            if (self._step_count > 0 and self._step_count % 200 == 0
                    and getattr(self.config, 'arch_self_mod_enabled', False)):
                gating_stats = []
                for layer in self.model.layers:
                    router = layer.router
                    stats = router.get_gating_stats()
                    gating_stats.append(stats)
                change = self._arch_modifier.step(
                    self.model, gating_stats_per_layer=gating_stats,
                )
                if change is not None:
                    self._rebuild_after_arch_change(change)

        self.interaction.update_sigma(self._last_sigma, self._total_steps)

        # ── Stage D: Push to dialectical buffer ──
        # P0-C 修复（审计 v2）：priority 与 loss/KL 解耦——
        # 高 coherence（稳定知识）状态优先进入巩固采样，困惑状态保留 0.5 下限。
        # 避免 KL 主导 priority 导致蒸馏偏向困惑、stable 专家被"困惑"污染。
        loss_val = self._loss_int.item()
        coherence = 1.0 - math.tanh(loss_val)
        priority = 0.5 + 0.5 * coherence  # [0.5, 1.0]，稳定知识优先，困惑不饿死
        self._replay_buffer.add(
            self._u_next_input.squeeze(0).detach(),
            loss=loss_val,
            novelty=priority,
            priority=priority,
        )

        prev_stats = self.dialectical_stats
        self.model.endosphere.push(
            self._u_next_input.squeeze(0).detach(), sigma=self._last_sigma
        )
        new_stats = self.dialectical_stats

        # ── L3: 状态沉淀为长期记忆（语义槽，可微）──
        mb = getattr(self.model, 'memory_bank', None)
        if mb is not None and getattr(self.config, 'memory_enabled', True):
            try:
                mb.write(self._u_next_input.squeeze(0).detach())
            except Exception:
                pass

        # ── 自发固化: salience 成熟即固化（非程序性，行为驱动）──
        self._check_memory_consolidation()

        if prev_stats and new_stats:
            if new_stats['syntheses'] > prev_stats.get('syntheses', 0):
                print(f"\n  [DIALECTIC] Synthesis achieved! (antithesis resolved)")
                print(">>> ", end='', flush=True)

        self._step_count += 1
        with self._stats_lock:
            self._loss_history.append({
                'step': self._total_steps,
                'loss': self._loss_int.item(),
                'sigma': self._last_sigma,
                'noise': self._noise_scheduler.current_noise,
                'kl': self._kl_value,
            })
            if len(self._loss_history) > 10000:
                self._loss_history = self._loss_history[-5000:]

        # ── Stage G: Fluid expert role evaluation ──
        if self._step_count % 100 == 0:
            adjustments = self._fluid_roles.evaluate(self.model, self._total_steps)
            if adjustments:
                with self._stats_lock:
                    for adj in adjustments:
                        self._loss_history.append({
                            'step': self._total_steps,
                            'event': 'role_transition',
                            **adj,
                        })

        # ── Stage H: Coordinated verification generation ──
        if self._step_count % 20 == 0:
            self._maybe_timeout_question()
            if self.interaction.can_ask:
                self._try_generate_verification()

    # ── Verification generation ───────────────────────────────────

    def _try_generate_verification(self):
        """
        Emergent question generation — no templates, no mechanical
        span extraction.  The question arises from the model's own
        internal dialectical tension, articulated through its LM.

        The model just processed the user's input.  Its hidden state
        encodes its understanding — or lack thereof.  When sigma is
        high, the model is confused.  We let it CONTINUE generating
        from its own confused state: the confusion naturally surfaces
        as a question or expression of doubt.

        The internal signals (contemplator gap, sigma) modulate HOW
        the model generates (temperature, search parameters), not
        WHAT token to start with.  This preserves language coherence
        while letting the internal state influence expression.
        """
        # 修复: 复查与 can_ask 用同一 sigma（interaction 的当前值），
        # 避免"can_ask 用旧值过阈值、复查用新值低于阈值"的边缘抖动
        sigma = getattr(self.interaction, '_current_model_sigma',
                        self._last_sigma)
        # 观测模式（用户决策）：不限制触发次数——只要 σ > 阈值就输出，
        # 静默/哑文本/成功都是数据；生成本身耗时 30-60s 持模型锁，
        # 天然限频（Stage H 每 20 步检查 + 生成时长 ≈ 每分钟一条样本）。
        if sigma <= self.config.verify_threshold:
            print(f'[VERIFY] sigma={sigma:.4f} <= 阈值，跳过', file=sys.stderr)
            return

        tok = self.model._decode_tokenizer
        if tok is None:
            return

        # ── Identify the most confused expert ──
        # C4 fix: _last_expert_sigmas only contains the LAST layer's
        # sigmas.  Use the last layer's experts, not all experts.
        expert_id = None
        expert_sigmas = getattr(self.model, '_last_expert_sigmas', None)
        if expert_sigmas is not None and expert_sigmas.numel() > 0:
            most_confused_idx = expert_sigmas.argmax().item()
            last_layer = self.model.layers[-1]
            if most_confused_idx < len(last_layer.all_experts):
                expert = last_layer.all_experts[most_confused_idx]
                expert_id = expert.id

        # ── The question emerges from continuation ──
        # 日志：让用户知道模型正在生成问题（27B 生成需数十秒~数分钟，
        # 期间持有模型锁，界面无响应属正常）
        print(f'[VERIFY] sigma={sigma:.3f} > 阈值，开始生成澄清问题 '
              f'(上限 {getattr(self.config, "graft_verify_max_tokens", 256)} '
              f'tokens，请稍候)...', file=sys.stderr)
        t0 = time.time()
        question_text = self._generate_emergent_question(expert_id, sigma)
        print(f'[VERIFY] 问题生成完成，耗时 {time.time()-t0:.0f}s',
              file=sys.stderr)
        # 观测标签：无对话上下文 → [MODEL FREE]（自由联想/内在状态的
        # 语言投影）；有对话 → [MODEL SAYS]（对话尾部续写涌现）
        is_free = True
        last_q = getattr(self.model, '_last_query_ids', None)
        if last_q is not None and last_q.numel() > 0:
            is_free = False
        if not question_text or len(question_text) < 2:
            if is_free:
                # 观测版：静默（eos）或哑文本（全特殊 token）都是数据，
                # 不做冷却——继续等待下一次触发
                print(f'[MODEL FREE] 静默/eos (sigma={sigma:.3f}, '
                      f'expert={expert_id})', file=sys.stderr)
            else:
                print(f'[VERIFY] 问题生成失败 (sigma={sigma:.3f}, '
                      f'expert={expert_id})', file=sys.stderr)
            return

        # Keep user's last input as feedback training context
        input_ids = getattr(self.model, '_last_query_ids', None) \
            or getattr(self.model, '_last_input_ids', None)
        query_ids = None
        if input_ids is not None:
            ctx_len = min(input_ids.size(1), 30)
            query_ids = input_ids[0, -ctx_len:].clone()

        self._confusion_map.record(question_text, sigma, self._total_steps)

        # ── 观测模式（用户决策）：[MODEL FREE] 只记录、不挂起 ──
        # 无对话时"回答它的碎碎念"没有意义且会打断连续观测；
        # 不提交问题 → 状态保持 idle → can_ask 保持 → σ 高时持续触发。
        # 对话涌现（[MODEL SAYS]）才走 提交→等待→反馈训练 闭环。
        if is_free:
            print(f"\n  [MODEL FREE] {question_text}")
            print(f"  [sigma={sigma:.3f}, step={self._total_steps}]")
            print(">>> ", end='', flush=True)  # re-display prompt
            return

        self._confusion_map.mark_asked(question_text)

        question_data = {
            'expert_id': expert_id,
            'query_ids': query_ids,
            'confused_text': question_text,
            'sigma_aggregate': sigma,
            'step': self._total_steps,
        }

        self.interaction.submit_question(question_data, self._total_steps)

        # ── Real-time notification ──
        # Print the question directly to the terminal so the user
        # sees it immediately, without having to type something first.
        # Then transition to AWAITING_ANSWER so the user's next input
        # is treated as an answer (not a new query).
        if self.interaction.state == 'question_pending':
            print(f"\n  [MODEL SAYS] {question_text}")
            print(f"  [sigma={sigma:.3f}, step={self._total_steps}]")
            print(">>> ", end='', flush=True)  # re-display prompt
            self.interaction.notify_question_displayed()

    def _generate_emergent_question(self, expert_id, sigma):
        """
        The question EMERGES from the model's own confused state.

        After processing the user's input, the model's hidden state
        encodes its understanding.  When sigma is high, the model is
        confused — it didn't fully grasp something.  We let the model
        CONTINUE generating from its last state.  The confusion
        naturally surfaces as uncertainty, doubt, or a question.

        The internal dialectical signals influence the GENERATION
        PROCESS rather than the seed token:

        - Sigma controls temperature: higher sigma → higher
          temperature → more diverse, uncertain expression.
        - Contemplator gap (imagination vs. reality) controls
          repetition penalty: larger gap → stronger penalty →
          the model avoids repeating what it already said, forcing
          it to explore new territory.
        - Dialectical state provides no direct input to generation
          but its existence (the model's internal history) is
          already encoded in the weights through Hebbian learning.

        This is consciousness expressing itself: the model's
        internal confusion modulates its language generation, and
        the question emerges naturally from the interaction between
        what it knows and what it doesn't.
        """
        tok = self.model._decode_tokenizer
        if tok is None:
            return None

        device = next(self.model.parameters()).device

        device = next(self.model.parameters()).device

        # ── seed：纯涌现——从模型最近的真实对话状态无缝续写 ──
        # 用户决策：模型没训练过"要提问"，不要模板/引导/疑问句强制——
        # 模型能输出什么就是什么（这才能诚实检验"σ 高时模型是否自显疑惑"）。
        # 历史教训：引导模板 → 模型把内部信号当话题；裸文本 → 首 token eos。
        # 观测版：无对话上下文的"无源之水"——模型始终在内循环（σ/自洽性/
        # h_t 持续演化），触发时用 bos 自由联想让它"说内循环状态下想说的"，
        # 这同样算涌现（内在状态的语言投影，虽无显式映射通道）。
        input_ids = getattr(self.model, '_last_query_ids', None)
        if input_ids is None or input_ids.numel() == 0:
            bos_id = getattr(tok, 'bos_token_id', None)
            if bos_id is None:
                print('[VERIFY] 无对话上下文且无 bos_token，跳过涌现',
                      file=sys.stderr)
                return None
            input_ids = torch.tensor([[bos_id]], device=device)
        seed = input_ids.to(device)

        # ── Internal signals modulate generation parameters ──
        # σ 只调制生成过程（温度：σ 越高越"犹豫"），不注入生成内容
        temp = 0.3 + 0.4 * min(sigma, 1.0)  # 0.3 ~ 0.7
        rep_penalty = 1.0                   # v3.8 教训：rep 惩罚会毁掉 Chat 模型

        try:
            with self.model._lock:
                with torch.no_grad():
                    generated = self.model.generate(
                        seed,
                        max_new_tokens=getattr(
                            self.config, 'graft_verify_max_tokens', 48),
                        temperature=temp,
                        repetition_penalty=rep_penalty,
                        top_k=0, top_p=0.95,
                    )

            question_text = tok.decode(
                generated[0, seed.size(1):],
                skip_special_tokens=True,
            ).strip()

            # 仅做格式清理：标签/空白不剥离（v3.21）——
            # 观测哲学"能输出什么就是什么"；无上下文时模型先验常输出
            # HTML/网页骨架（首 token 常为 185='<'），剥 <[^>]+> 会把
            # "哑文本"误判为"静默"（v3.20 实测 512 token 全常规却 decode 空）
            import re
            question_text = re.sub(
                r'<think>.*?</think>', '', question_text, flags=re.S)
            question_text = re.sub(r'\s+', ' ', question_text).strip()

            # 仅长上限保护（防 512 token 超长刷屏），不截内容语义
            if len(question_text) > 150:
                question_text = question_text[:150]

            if len(question_text) < 3:
                # 空输出诊断：区分"真静默(eos@0)"与"哑文本/标签流/全特殊"
                # ——零基线数据质量关键；附未清理原始文本便于人工判别
                gen_tokens = generated[0, seed.size(1):]
                n_special = (
                    (gen_tokens >= 100000).sum().item()
                    if gen_tokens.numel() else 0)
                raw_text = tok.decode(
                    generated[0, seed.size(1):], skip_special_tokens=False
                )[:120]
                print(f'[VERIFY] 空输出诊断: 生成 {gen_tokens.numel()} token, '
                      f'首={int(gen_tokens[0].item()) if gen_tokens.numel() else -1}, '
                      f'特殊token {n_special}/{gen_tokens.numel()} '
                      f'| 原始前120: {raw_text!r}', file=sys.stderr)
                return None
            return question_text
        except Exception:
            return None

    # ── Internal methods ──────────────────────────────────────────

    def _maybe_timeout_question(self, max_steps=100000):
        q = self.interaction.active_question
        if q is None:
            return
        asked_step = q.get('step', 0)
        if self._total_steps - asked_step > max_steps:
            self.interaction.cancel_question()
            print(f"\n  [TIMEOUT] Question canceled (step {asked_step} -> {self._total_steps})")

    def _check_memory_consolidation(self):
        """自发固化触发（每步检查，非程序性）。

        salience 成熟（被反复聚焦）的记忆 → 单条蒸馏进 stable 专家。
        sigma 仅调制强度（高=正在用于解决困惑→更强）。
        """
        mb = getattr(self.model, 'memory_bank', None)
        if (mb is None
                or not getattr(self.config, 'memory_salience_enabled', True)
                or not getattr(self.config, 'memory_enabled', True)):
            return
        try:
            thr = getattr(self.config, 'memory_salience_threshold', 3.0)
            max_n = getattr(self.config, 'memory_consolidate_batch', 4)
            hot = mb.get_hot_memories(threshold=thr, max_n=max_n)
            if hot is None:
                return
            idx, vecs = hot  # idx [n], vecs [n, d]
            # sigma 调制强度：sigma 0.5→1.0, 0.7→1.04, 0.3→0.96
            sig = getattr(self, '_last_sigma', 0.5)
            strength = 1.0 + getattr(
                self.config, 'memory_sigma_strength', 0.2) * (sig - 0.5) * 2
            from learn.consolidation import distill_memory_slots
            distill_memory_slots(self.model, vecs.unsqueeze(1),
                                 self.config, strength=strength)
            cd = getattr(self.config, 'memory_consolidate_cooldown', 50)
            mb.mark_consolidated(idx, cooldown=cd)
        except Exception:
            pass  # 自发固化是 best-effort，绝不破坏内循环

    def _get_internal_mem_kv(self):
        """内部循环的记忆 KV（L4 内容记忆）——缓存，轮次变化时重建。

        内部循环同样能"想起"对话内容：1-token 状态的 Q
        对历史对话 KV 做注意力检索，意识流受对话影响。
        """
        mb = getattr(self.model, 'memory_bank', None)
        if mb is None or not getattr(self.config, 'memory_enabled', True):
            return None
        try:
            n = len(mb.kvcache)
            if self._mem_kv_n != n:
                self._mem_kv_n = n
                if n == 0:
                    self._mem_kv = None
                    return None
                mem_kv = {}
                dev = next(self.model.parameters()).device
                model_dtype = next(self.model.parameters()).dtype
                # 嫁接模式：只取全注意力层（kv_layers），位置索引存入 MemoryBank
                kv_idxs = self.model.kv_layers
                for pos, i in enumerate(kv_idxs):
                    kv = mb.get_kv(pos)
                    if kv is None:
                        self._mem_kv = None
                        return None
                    k, v = kv
                    if k.size(1) == 0:
                        self._mem_kv = None
                        return None
                    mem_kv[i] = (k.unsqueeze(0).to(device=dev, dtype=model_dtype),
                                 v.unsqueeze(0).to(device=dev, dtype=model_dtype))
                self._mem_kv = mem_kv
            return self._mem_kv
        except Exception:
            return None

    def _get_state(self):
        h = getattr(self.model, '_h_state', None)
        if h is not None:
            return h
        return self.model.endosphere.get_latest()

    def _init_state(self):
        device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        sm = getattr(self.model, 'self_model', None)
        if sm is not None:
            h, z = sm.init_state(device=device)
            self.model._h_state = h
            self.model._z_state = z
            return h
        noise = torch.randn(1, self.config.d_model, device=device, dtype=model_dtype)
        return noise.cpu()

    def _imagine(self, v_t):
        sm = getattr(self.model, 'self_model', None)
        if sm is not None and self._h_t is not None and self._z_t is not None:
            h_t = self._h_t.to(v_t.device)
            z_t = self._z_t.to(v_t.device)
            action = v_t.unsqueeze(0) if v_t.dim() == 1 else v_t
            if action.dim() == 1:
                action = action.unsqueeze(0)

            h_next, z_mean, z_logvar = sm(h_t, z_t, action)
            z_sample = sm.sample_z(z_mean, z_logvar)
            u_next = sm.decode(z_sample, h_next).squeeze(0)

            self.model._h_state = h_next.detach()
            self.model._z_state = z_sample.detach()
            self._h_t = h_next.detach()
            self._z_t = z_sample.detach()

            return u_next
        return v_t

    def _ema_blend(self, u, v):
        alpha = self.config.ema_alpha
        return alpha * u + (1 - alpha) * v

    def _get_target(self, u_next_input):
        sm = getattr(self.model, 'self_model', None)
        if sm is not None and self._h_t is not None and self._z_t is not None:
            h_t = self._h_t.to(u_next_input.device)
            z_t = self._z_t.to(u_next_input.device)
            with torch.no_grad():
                o_pred = sm.decode(z_t, h_t)
            return o_pred.squeeze(1)
        with torch.no_grad():
            return u_next_input

    def _compute_loss(self, pred_emb, v_t):
        sm = getattr(self.model, 'self_model', None)
        if sm is None or self._h_t is None or self._z_t is None:
            return self._fallback_loss(pred_emb, v_t)

        device = pred_emb.device
        h_t = self._h_t.to(device)
        z_t = self._z_t.to(device)

        if pred_emb.dim() == 3:
            pred_emb = pred_emb.squeeze(1)

        # Posterior: correct belief after observing actual output
        z_post_mean, z_post_logvar = sm.observe_and_correct(h_t, pred_emb)

        # Decode reconstruction from prior (detached z_t, h_t -> o_pred is constant)
        # This acts as a fixed target for pred_emb, pushing the transformer
        # to produce outputs that match the SelfModel's imagination.
        o_pred = sm.decode(z_t, h_t).detach()

        # 1) Imagination loss: pred_emb should match decoded imagination
        loss_imagination = F.mse_loss(pred_emb, o_pred)

        # 2) Curiosity loss: KL[Q(z|h,o) || P(z|h)]
        # Gradient flows through pred_emb (via posterior) and through
        # stored prior (z_mean, z_logvar from Stage A)
        prior_z_mean = getattr(self, '_z_prior_mean', None)
        prior_z_logvar = getattr(self, '_z_prior_logvar', None)
        if prior_z_mean is not None and prior_z_logvar is not None:
            loss_curiosity = sm.kl_divergence(
                z_post_mean, z_post_logvar,
                prior_z_mean.to(device), prior_z_logvar.to(device),
            )
            self._kl_value = loss_curiosity.item()
        else:
            loss_curiosity = torch.tensor(0.0, device=device)

        # Note: stability loss is in _update_self_model (Stage C) where
        # h_t and h_next both have gradients. Here they're detached.
        # The old Lyapunov term (mean(h_t**2)*1e-4) was removed because
        # h_t is detached here -- it contributed zero gradient and only
        # inflated the reported loss, corrupting replay admission and
        # critic targets.

        total = (self._imagination_lambda * loss_imagination
                 + self._curiosity_beta * loss_curiosity)

        # P1-4 修复：内循环梯度接地——想象输出与最近对话 embedding 对齐。
        # 让意识流的梯度源不纯是自指（SelfModel↔Transformer 互锁），
        # 而是扎根于用户真实输入（外部困惑源），Hebbian 更新方向获得外部锚点。
        dialog_emb = getattr(self.model, '_last_dialog_emb', None)
        grounded_w = getattr(self.config, 'grounded_weight', 0.0)
        if dialog_emb is not None and grounded_w > 0:
            # reshape 对齐形状（[d] -> [1, d]），消除 MSE 广播警告
            target = dialog_emb.to(pred_emb.device).detach().reshape_as(pred_emb)
            total = total + grounded_w * F.mse_loss(pred_emb, target)
        return total

    def _fallback_loss(self, pred_emb, v_t):
        """Fallback to original MSE loss when SelfModel is not available."""
        if pred_emb.dim() == 3:
            pred_emb = pred_emb.squeeze(1)
        mse = F.mse_loss(pred_emb, v_t.unsqueeze(0) if v_t.dim() == 1 else v_t)
        lyap_lambda = self.config.lyapunov_lambda
        if lyap_lambda > 0:
            lyap = F.mse_loss(pred_emb, torch.zeros_like(pred_emb))
            mse = mse + lyap_lambda * lyap
        return mse

    def _update_self_model(self, u_next_input):
        sm = getattr(self.model, 'self_model', None)
        if sm is None or self._self_model_optimizer is None:
            return
        device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        u_in = u_next_input.to(device=device, dtype=model_dtype)
        if u_in.dim() == 1:
            u_in = u_in.unsqueeze(0).unsqueeze(0)
        elif u_in.dim() == 2:
            u_in = u_in.unsqueeze(0)

        h_t = getattr(self.model, '_h_state', None)
        z_t = getattr(self.model, '_z_state', None)
        if h_t is None or z_t is None:
            return
        h_t = h_t.to(device=device, dtype=model_dtype)
        z_t = z_t.to(device=device, dtype=model_dtype)

        # Re-forward SelfModel
        action = u_in.squeeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        h_next, z_mean, z_logvar = sm(h_t, z_t, action)
        z_sample = sm.sample_z(z_mean, z_logvar)
        u_decoded = sm.decode(z_sample, h_next).squeeze(0)
        # C1 fix: forward_internal 只是"观察目标"——detach 输入，
        # 防止 backward 把 SelfModel 训练梯度泄漏进 Transformer 参数 .grad
        # （否则 49 步累积的 stale 梯度会在 consolidation 时污染 stable 专家）
        # 嫁接轻量模式：观察前向整体 no_grad（27B 主干不建图，省显存/算力；
        # 损失项只依赖 SelfModel 参数，观察值 detach 不影响训练语义）
        if (getattr(self.config, 'graft_lite', False)
                and getattr(self.config, 'backbone', '') != 'reflex'):
            with torch.no_grad():
                pred = self.model.forward_internal(
                    u_decoded.detach(), h_state=h_next.detach(),
                    mem_kv=self._get_internal_mem_kv())
        else:
            pred = self.model.forward_internal(u_decoded.detach(),
                                                h_state=h_next.detach(),
                                                mem_kv=self._get_internal_mem_kv())
        if pred.dim() == 3:
            pred = pred.squeeze(1)

        # Posterior + decode
        z_post_mean, z_post_logvar = sm.observe_and_correct(h_next, pred)
        o_pred = sm.decode(z_sample, h_next)

        # Multi-objective loss:
        # 1) Imagination: decoder reconstruction vs actual forward output
        loss_imagination = F.mse_loss(o_pred, pred)

        # 2) Curiosity: KL[Posterior || Prior] - how much did o_t change beliefs
        loss_curiosity = sm.kl_divergence(
            z_post_mean, z_post_logvar, z_mean, z_logvar
        )

        # 3) Stability: prevent h_t from jumping chaotically between steps
        loss_stability = F.mse_loss(h_t, h_next)

        loss = (self._imagination_lambda * loss_imagination
                + self._curiosity_beta * loss_curiosity
                + self._stability_lambda * loss_stability)

        self._self_model_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sm.parameters(), 1.0)
        self._self_model_optimizer.step()

    def _calibrate_sigma(self):
        """sigma 在线校准（graft_sigma_cal）：尾层 uncertainty_head 学习
        tanh(loss_int) 目标——sigma 高 ↔ 模型预测不确定度高，为主动求证
        提供真实触发基础（修复审计 P0-1：sigma 头随机初始化且无训练路径）。

        必须在 Stage B 之前调用（loss_int 的计算图还完整）；用 autograd.grad
        定向取 uncertainty_head 梯度（不污染其他参数 .grad），retain_graph=True
        保证 Stage B 的 Hebbian 图仍然可用。
        """
        if self._sigma_optimizer is None:
            return
        interval = getattr(self.config, 'graft_sigma_cal_interval', 20)
        if self._total_steps % interval != 0:
            return
        if self._loss_int is None:
            return
        try:
            k = getattr(self.config, 'graft_hebbian_layers', 8)
            sigmas = [getattr(layer, '_learnable_sigmas', None)
                      for layer in self.model.layers[-k:]]
            sigmas = [s for s in sigmas if s is not None]
            if not sigmas:
                return
            device = next(self.model.parameters()).device
            dtype = next(self.model.parameters()).dtype
            target = math.tanh(self._loss_int.detach().item())
            target_t = torch.tensor(target, dtype=dtype, device=device)
            loss = sum(F.mse_loss(s, target_t) for s in sigmas) / len(sigmas)
            params = list(self._sigma_optimizer.param_groups[0]['params'])
            grads = torch.autograd.grad(
                loss, params, retain_graph=True, allow_unused=True)
            self._sigma_optimizer.zero_grad()
            for p, g in zip(params, grads):
                if g is not None:
                    p.grad = g
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            self._sigma_optimizer.step()
        except Exception as e:
            with self._stats_lock:
                self._loss_history.append({'error': f'sigma_cal: {e}'})

    def _sigma_gamma(self):
        """
        Compute gamma for Hebbian update modulation.

        Combines sigma-driven modulation with Critic TD-error for
        better credit assignment.  The Critic learns to estimate
        state value V(s); when TD-error is large (surprise), gamma
        increases to drive larger updates.
        """
        sig = self._last_sigma
        threshold = self.config.verify_threshold
        base_gamma = min(max(sig / max(threshold, 0.01), 0.5), 2.0)

        # Critic TD-error modulation
        critic = getattr(self.model, 'critic', None)
        if critic is not None and self._v_t is not None:
            with torch.no_grad():
                v_current = critic.get_normalized_v(self._v_t)
                # TD-error proxy: |V(s) - 0| normalised.
                # High |V| means the state is surprising → larger update.
                td_mod = min(abs(v_current) / 2.0, 1.0)
                base_gamma = base_gamma * (1.0 + 0.5 * td_mod)
                base_gamma = min(base_gamma, 3.0)

        return base_gamma

    def _train_critic(self):
        """
        Train the Critic on pseudo-reward synthesized from internal signals.

        TD(0) with one-step delay (M5 fix):
          at step t we cache (state_{t-1}, r_{t-1}); at step t we train
            V(s_{t-1}) -> r_{t-1} + gamma * V(s_t).detach()
        i.e. the bootstrap always uses the NEXT state's value, never the
        previous one.  First step after init has no pending pair -> no update.
        """
        critic = getattr(self.model, 'critic', None)
        critic_opt = getattr(self.model, 'critic_optimizer', None)
        if critic is None or critic_opt is None:
            return
        if self._v_t is None or self._loss_int is None:
            return

        from learn.critic import compute_pseudo_reward

        loss_val = self._loss_int.item()
        sigma = self._last_sigma
        pseudo_r = compute_pseudo_reward(
            loss_int=loss_val,
            sigma_aggregate=sigma,
        )

        # ── 延迟 TD(0)：用缓存的上一步 (state, reward) 训练 ──
        pending = getattr(self, '_critic_pending', None)
        if pending is not None:
            state_prev, r_prev = pending  # (s_{t-1}, r_{t-1})
            gamma = getattr(self.config, 'gamma', 0.99)
            # V(s_t) 作为 bootstrap 目标（当前状态 = s_{t-1} 的 next state）
            with torch.no_grad():
                v_next = critic(self._v_t).squeeze()   # V(s_t)
            v_prev_pred = critic(state_prev).squeeze()  # V(s_{t-1})，可导
            td_target = r_prev + gamma * v_next
            critic_loss = F.smooth_l1_loss(v_prev_pred, td_target)
            critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            critic_opt.step()

        # 缓存当前 (state, r) 供下一步使用（s_{t-1} 视角）
        self._critic_pending = (self._v_t.detach().clone(), pseudo_r)

    def _mini_consolidate(self):
        """
        每 50 步：局部蒸馏（stable expert）+ 全局回放巩固。
        全局回放从事后角度更新 Attention/Router/LMHead/RMSNorm/AttnRes，
        让模型的基础设施慢慢适应 expert 的局部学习成果。
        """
        if len(self._replay_buffer) < 4:
            return
        replay = self._replay_buffer.sample(4)
        if replay is None:
            return
        from learn.consolidation import mini_distill
        mini_distill(
            self.model, replay, self.config,
            global_optimizer=self._global_optimizer,
        )

    def _major_consolidate(self):
        """
        每 500 步：完整巩固 + EWC 保护 + 全局回放。
        """
        if len(self._replay_buffer) < self.config.sleep_batch_size:
            return
        from learn.consolidation import major_sleep
        major_sleep(
            self.model, self._replay_buffer, self.config,
            global_optimizer=self._global_optimizer,
        )

    def _rebuild_after_arch_change(self, change_type):
        """
        Rebuild training infrastructure after architecture self-modification.

        - 'add_layer': rebuild global optimizer (new layer params not in it)
                       and GradientManager (stale layer count)
        - 'split'/'prune': router dims changed; no optimizer rebuild needed
                       (router params are resized in-place), but Fisher
                       protection from the previous major_sleep is stale.
                       This is safe because ewc_penalty silently skips
                       mismatched keys, and the next major_sleep recomputes
                       Fisher fresh.
        - 'replace': no structural change, just weight copy; nothing to do.
        """
        if change_type == 'replace':
            return

        if change_type == 'add_layer':
            # Update layer count
            self.config.n_layers = len(self.model.layers)
            self.model.config.n_layers = len(self.model.layers)

            # Rebuild GradientManager with new layer count
            self._gradient_mgr = GradientManager(self.config.n_layers)

        # For add_layer / split / prune: rebuild global optimizer
        # because add_layer adds entirely new parameter tensors,
        # and split/prune resize router weight matrices (new Parameter
        # objects that the old optimizer doesn't track).
        global_params = []
        if (getattr(self.config, 'graft_freeze_backbone', False)
                and getattr(self.config, 'backbone', '') != 'reflex'):
            for name, p in self.model.named_parameters():
                if any(k in name for k in
                       ['router', 'attn_res', 'post_norm', 'memory_bank',
                        'uncertainty_head', 'ln1', 'ln2', 'ln_f',
                        'q_norm', 'k_norm']):
                    if p.requires_grad:
                        global_params.append(p)
        else:
            for name, p in self.model.named_parameters():
                if any(k in name for k in
                       ['attention', 'router', 'lm_head',
                        'ln_f', 'ln1', 'ln2',
                        'q_norm', 'k_norm', 'q_proj', 'kv_proj', 'o_proj',
                        'attn_res', 'post_norm']):
                    if p.requires_grad:
                        global_params.append(p)
        if global_params:
            self._global_optimizer = torch.optim.AdamW(
                global_params, lr=1e-6, weight_decay=0.01,
                betas=(0.9, 0.95),
            )
            self._global_param_count = len(global_params)
            self._global_params = list(global_params)
            self._global_snapshot = [p.data.clone() for p in global_params]
        else:
            self._global_optimizer = None
            self._global_param_count = 0
            self._global_params = []
            self._global_snapshot = None
