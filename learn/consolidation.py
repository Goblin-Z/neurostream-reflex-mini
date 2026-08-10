import torch
import torch.nn as nn
import torch.nn.functional as F


_STABLE_GRAD_CLIP = 1.0


def _safe_sgd_step(experts, lr, grad_clip=_STABLE_GRAD_CLIP):
    """Manual SGD on expert params with isfinite guard + grad-norm clip.

    Accepts either Expert modules or a flat list of nn.Parameter
    (major_sleep passes get_stable_named_params() -> Parameters).
    A single NaN/Inf grad would permanently corrupt stable experts;
    this skips non-finite grads and clips the update magnitude.
    """
    params = []
    for e in experts:
        if isinstance(e, torch.nn.Parameter):
            params.append(e)
        else:
            params.extend(e.parameters())

    with torch.no_grad():
        for param in params:
            if param.grad is None:
                continue
            g = param.grad.data
            if not torch.isfinite(g).all():
                continue
            g_norm = g.norm()
            if g_norm > grad_clip:
                g = g * (grad_clip / g_norm)
            param.data -= lr * g


def distill_memory_slots(model, mem_vecs, config, strength=1.0):
    """按指定语义槽蒸馏进 stable 专家（支持强度调制）。

    自发固化（salience 驱动）与批量固化（memory_distill）共用的核心。
    mem_vecs: [B, 1, d] 记忆向量（B ≤ 8 轻量）
    strength: sigma 调制强度（高=正在使用→更强）
    """
    if mem_vecs is None or mem_vecs.size(0) == 0:
        return
    try:
        device = next(model.parameters()).device
        mem_vecs = mem_vecs.to(device)

        model.eval()
        with torch.no_grad():
            teacher = model.forward_embeddings(mem_vecs)

        model.train()
        model.set_stable_requires_grad(True)
        model.set_plastic_requires_grad(False)
        model.zero_grad()

        student = model.forward_embeddings(mem_vecs)
        loss = F.mse_loss(student, teacher.detach())
        loss.backward()
        lr = config.sleep_lr * max(0.5, min(2.0, strength))
        _safe_sgd_step(model.get_stable_experts(), lr)
        model.zero_grad()
        model.set_plastic_requires_grad(True)
    except Exception:
        model.set_plastic_requires_grad(True)
        model.zero_grad()


def memory_distill(model, config):
    """记忆→权重压缩固化（Memory Distillation，批量采样）。

    语义槽（长期记忆的压缩表示）作为蒸馏输入，stable 专家学习
    模型对记忆的处理——记忆的知识固化进权重，即使槽位被覆盖
    知识仍在权重中（持久）。

    模型内部能力：MemoryBank 是模型组件，forward_embeddings 是
    模型方法（计算图在模型内），梯度更新模型参数——外部仅调度
    触发时机（与 Hebbian/consolidation 相同模式）。

    兼容性：复用自我蒸馏机制（teacher=detach 快照 + _safe_sgd_step），
    独立于 replay 蒸馏执行（各自 zero_grad），不干扰全局回放。
    """
    mb = getattr(model, 'memory_bank', None)
    if (mb is None or not getattr(config, 'memory_distill_enabled', True)
            or mb._pos == 0):
        return
    try:
        n_used = min(mb._pos, mb.capacity)
        batch = min(getattr(config, 'memory_distill_batch', 8), n_used)
        # 采样已写入的语义槽（记忆的压缩表示）
        idx = torch.randperm(n_used)[:batch]
        mem_vecs = mb.memory_matrix[idx].unsqueeze(1)  # [B, 1, d]
        distill_memory_slots(model, mem_vecs, config, strength=1.0)
    except Exception:
        # 记忆蒸馏是 best-effort，绝不破坏 consolidation
        model.set_plastic_requires_grad(True)
        model.zero_grad()


def mini_distill(model, replay_batch, config, global_optimizer=None):
    """
    Mini-consolidation (every 50 steps):
    1. Self-distillation from teacher snapshot into stable experts (局部)
    2. Global replay: update Attention/Router/LMHead/RMSNorm/AttnRes (全局)

    The teacher is a detached snapshot of the current model (all params
    frozen).  The student is the same model with stable experts trainable.
    This pushes stable experts to reproduce the model's current behaviour,
    consolidating recent plastic-expert learning into long-term storage.

    The global replay then updates the model's infrastructure (attention,
    router, output projection) to better utilize the updated experts.
    This is the "事后回放" mechanism: local learning happens fast during
    interaction, global consolidation happens slowly in the background.
    """
    if replay_batch is None:
        return
    device = next(model.parameters()).device
    embeddings = replay_batch['embeddings'].to(device)
    if embeddings.dim() == 2:
        embeddings = embeddings.unsqueeze(1)

    # ── 局部蒸馏：stable expert 学习当前模型行为 ──
    stable_named = _save_stable(model)

    model.eval()
    with torch.no_grad():
        teacher = model.forward_embeddings(embeddings)

    model.train()
    model.set_stable_requires_grad(True)
    model.set_plastic_requires_grad(False)

    # C1 fix: 清空任何残留梯度（Stage C 泄漏或上一轮 backward 残留），
    # 保证 _safe_sgd_step 只应用本次蒸馏的梯度
    model.zero_grad()

    student = model.forward_embeddings(embeddings)
    loss = F.mse_loss(student, teacher.detach())

    loss.backward()

    _safe_sgd_step(model.get_stable_experts(), config.sleep_lr)

    model.zero_grad()
    model.set_plastic_requires_grad(True)

    # ── 全局回放：事后巩固 Attention/Router/LMHead/LayerNorm ──
    if global_optimizer is not None:
        # 冻结所有 expert 参数，只更新全局参数
        model.set_stable_requires_grad(False)
        model.set_plastic_requires_grad(False)

        # 用更新后的 expert 重新 forward，让全局参数适应新 expert
        student_global = model.forward_embeddings(embeddings,
                                                   save_hebbian_buffers=False)
        loss_global = F.mse_loss(student_global, teacher.detach())
        aux = model.get_aux_loss()
        if aux is not None:
            loss_global = loss_global + 0.01 * aux

        global_optimizer.zero_grad()
        loss_global.backward()
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters()
             if p.requires_grad and p.grad is not None],
            1.0,
        )
        global_optimizer.step()

        # 解冻 expert 参数
        model.set_stable_requires_grad(True)
        model.set_plastic_requires_grad(True)
        model.zero_grad()

    # ── 记忆→权重压缩固化（语义槽蒸馏进 stable 专家）──
    memory_distill(model, config)


def major_sleep(model, replay_buffer, config, global_optimizer=None):
    """
    Major consolidation (every 500 steps):
    Full sleep cycle with EWC, distillation, soft reset, buffer clear,
    and global replay for Attention/Router/LMHead/RMSNorm/AttnRes.

    The global replay uses the same teacher-student framework but
    updates non-expert parameters.  EWC on expert weights prevents
    the global gradient from undoing local Hebbian learning.
    """
    from learn.fisher import estimate_fisher, ewc_penalty, get_stable_named_params

    device = next(model.parameters()).device

    batch_data = replay_buffer.sample(config.sleep_batch_size)
    if batch_data is None:
        return

    embeddings = batch_data['embeddings'].to(device)
    if embeddings.dim() == 2:
        embeddings = embeddings.unsqueeze(1)

    old_stable = _save_stable(model)
    fisher = estimate_fisher(model, embeddings)

    model.eval()
    with torch.no_grad():
        teacher = model.forward_embeddings(embeddings)

    # ── 局部蒸馏：stable expert + EWC 保护 ──
    model.train()
    model.set_plastic_requires_grad(False)
    model.set_stable_requires_grad(True)

    # C1 fix: 清空残留梯度（fisher 末次采样残留 + Stage C 泄漏）
    model.zero_grad()

    student = model.forward_embeddings(embeddings)
    distill = F.mse_loss(student, teacher.detach())

    stable_named = get_stable_named_params(model)
    penalty = ewc_penalty(stable_named, fisher, old_stable,
                          config.ewc_lambda)

    total = distill + penalty
    total.backward()

    _safe_sgd_step(
        [e for _, e in stable_named], config.sleep_lr,
    )
    model.zero_grad()
    # ── 全局回放：事后巩固 Attention/Router/LMHead/RMSNorm/AttnRes ──

    if global_optimizer is not None:
        model.set_stable_requires_grad(False)
        model.set_plastic_requires_grad(False)

        student_global = model.forward_embeddings(embeddings,
                                                   save_hebbian_buffers=False)
        loss_global = F.mse_loss(student_global, teacher.detach())
        aux = model.get_aux_loss()
        if aux is not None:
            loss_global = loss_global + 0.01 * aux

        global_optimizer.zero_grad()
        loss_global.backward()
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters()
             if p.requires_grad and p.grad is not None],
            1.0,
        )
        global_optimizer.step()

        model.set_stable_requires_grad(True)
        model.set_plastic_requires_grad(True)
        model.zero_grad()

    # ── 记忆→权重压缩固化（语义槽蒸馏进 stable 专家）──
    memory_distill(model, config)

    _soft_reset_plastic(model, config.plastic_soft_reset_keep)

    replay_buffer.clear_old(config.sleep_keep_ratio)
    if hasattr(model, 'endosphere'):
        model.endosphere.clear()


def _save_stable(model):
    """Save stable expert parameter snapshots by name (not by 'stable_experts' substring)."""
    from learn.fisher import _get_stable_param_names
    stable_names = _get_stable_param_names(model)
    saved = {}
    for name, param in model.named_parameters():
        if name in stable_names:
            saved[name] = param.data.clone()
    return saved


def _soft_reset_plastic(model, keep_ratio=0.5):
    """Gentle soft-reset: keep 50% of plastic weights (was 10%->30%->50%).

    The gradual decay in InternalLoop._execute_step (plastic_reg_strength)
    provides continuous regularization. This periodic reset is a milder
    "fresh capacity" injection, not an aggressive wipe.
    """
    for expert in model.get_plastic_experts():
        with torch.no_grad():
            for param in expert.parameters():
                fresh = torch.empty_like(param.data)
                if param.dim() >= 2:
                    torch.nn.init.xavier_uniform_(fresh)
                else:
                    torch.nn.init.zeros_(fresh)
                param.data.mul_(keep_ratio).add_(fresh, alpha=1.0 - keep_ratio)
        expert.clear_buffers()
        expert.clear_momentum()  # reset momentum after weight reset
