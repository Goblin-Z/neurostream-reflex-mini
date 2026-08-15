from dataclasses import dataclass, field
from typing import Tuple


class ModelConfig:
    """Alias for SEQ compatibility during checkpoint loading."""
    pass


@dataclass
class ReflexConfig:
    # ── Architecture ──
    d_model: int = 512
    n_layers: int = 4
    n_heads: int = 8
    n_kv_heads: int = 4  # GQA: fewer KV heads (n_heads // n_kv_heads = repeat factor)
    d_ff: int = 2048
    n_stable: int = 5
    n_plastic: int = 3
    top_k: int = 2
    vocab_size: int = 151936
    max_seq_len: int = 1024
    dropout: float = 0.1
    attention_dropout: float = 0.1  # SDPA dropout during training (0 to always use Flash)

    # ── RoPE ──
    rope_theta: float = 10000.0

    # ── Attention Residuals (Kimi AttnRes) ──
    attnres_enabled: bool = True
    attnres_block_size: int = 2  # layers per block (n_layers=4 -> 2 blocks -> 1 boundary)
    attnres_rank: int = 128     # low-rank key dimension for depth-wise attention
    # post_norm 初始尺度：原 1e-3 使 AttnRes（含记忆 source）几乎无声；
    # 放大到 0.1 让跨块 delta 与记忆检索真实影响输出（记忆微调前提）
    attnres_postnorm_init: float = 0.1

    # ── Weight tying ──
    tie_word_embeddings: bool = True

    # ── Expert LR spectrum ──
    # idx 0-4 = stable, 5-7 = plastic
    expert_baseline_lrs: Tuple[float, ...] = (
        1e-7, 1e-7, 1e-7, 1e-7, 1e-7,  # stable x5
        1e-5, 1e-4, 1e-4,               # plastic x3
    )

    # ── Internal loop ──
    cont_lr: float = 1e-4
    ema_alpha: float = 0.8
    lyapunov_lambda: float = 0.01
    internal_noise_scale: float = 0.01
    internal_noise_growth: float = 2.0
    internal_entropy_threshold: float = 0.5

    noise_annealing_steps: int = 50000
    noise_min: float = 0.0001
    noise_max: float = 0.05

    # ── Verification ──
    verify_threshold: float = 0.5
    verify_min_interval: int = 10
    verify_query_max_tokens: int = 30
    max_cognitive_depth: int = 5
    # 部署初期保守冷却（审计 v2 P1-5）：前 verify_warmup_steps 步内，
    # 距上次提问不足 verify_warmup_cooldown 步则不提问，防止 sigma 未校准时的随机提问
    verify_warmup_steps: int = 500
    verify_warmup_cooldown: int = 50

    # ── SelfModel (v2 Contemplator upgrade) ──
    self_model_enabled: bool = True
    self_model_z_dim: int = 64
    self_model_hidden_dim: int = 512
    self_model_n_prior_experts: int = 3
    self_model_n_post_experts: int = 3

    # ── Curiosity drive (v2) ──
    curiosity_beta: float = 0.1
    imagination_lambda: float = 1.0
    stability_lambda: float = 0.01
    # P1-4: 内循环梯度接地——想象输出与最近对话 embedding 的 MSE 权重
    # （让意识流扎根于真实对话，而非纯自指）
    grounded_weight: float = 0.1

    # ── Consolidation ──
    sleep_lr: float = 1e-5
    distill_temperature: float = 2.0
    ewc_lambda: float = 40.0
    sleep_batch_size: int = 128
    sleep_keep_ratio: float = 0.5
    plastic_soft_reset_keep: float = 0.5
    plastic_reg_strength: float = 1e-6

    # ── Architecture self-modification ──
    # 默认关闭（无对照实验前不改写已训练权重）；部署可用 --arch-self-mod 显式开启
    arch_self_mod_enabled: bool = False
    arch_replace_reward_threshold: float = 0.2
    arch_split_load_ratio: float = 3.0
    arch_add_layer_improvement_threshold: float = 0.01

    # ── Critic ──
    critic_enabled: bool = True
    critic_hidden_dim: int = 256
    critic_lr: float = 1e-3
    actor_lr: float = 1e-3
    gamma: float = 0.99
    expert_lr_bias_range: float = 2.0
    query_lr: float = 1e-4
    verify_threshold_init: float = 0.5

    # ── Feedback (Reflex key innovation) ──
    feedback_lr: float = 1e-5
    feedback_max_grad_norm: float = 1.0
    feedback_gamma_modulation_strength: float = 0.3
    feedback_alignment_weight: float = 0.5  # deprecated (unused in feedback.py)
    feedback_align_loss_weight: float = 0.2  # alignment loss weight (CE complement)
    sigma_calibration_weight: float = 0.2  # sigma calibration loss weight (0.05→0.2，强化校准)
    feedback_alignment_threshold: float = 0.2  # 反馈对齐最低余弦相似度（原硬编码 0.3 过高，P2-4）

    # ── Sampling (generation) ──
    sampling_temperature: float = 0.8
    sampling_top_k: int = 40
    sampling_top_p: float = 0.9
    sampling_repetition_penalty: float = 1.5

    # ── Gradient management (Reflex key innovation) ──
    gradient_per_layer: bool = True

    # ── Others ──
    internal_steps_per_cycle: int = 1000
    # 内循环每步后的休眠毫秒数（防忙循环烧算力；0=不限制）。
    # 默认 5ms ≈ 200 steps/s 上限，保持"持续思考"同时释放 CPU/GPU
    internal_step_delay_ms: float = 5.0
    internal_loss_clip: float = 10.0
    endosphere_capacity: int = 1024
    replay_capacity: int = 10000
    supervised_enabled: bool = False
    supervised_batch_size: int = 4
    supervised_lr: float = 2e-5
    max_new_tokens: int = 256

    # ── Memory system v4 (L1-L4) ──
    dialog_memory_alpha: float = 0.3       # L1: 对话→h_t 混合率
    memory_enabled: bool = True            # 总开关
    memory_bank_capacity: int = 128        # L2/L3: 语义槽数量
    memory_context_top_k: int = 8          # L2: AttnRes 记忆候选数
    memory_write_lr: float = 0.05          # L3: 写入门强度
    kv_cache_rounds: int = 4               # L4: KV 内容记忆保留轮数
    memory_distill_enabled: bool = True    # 记忆→权重压缩固化开关
    memory_distill_batch: int = 8          # 每次蒸馏采样语义槽数
    # ── 自发固化（salience 驱动，非程序性）──
    memory_salience_enabled: bool = True   # salience 累积与即时固化开关
    memory_salience_threshold: float = 3.0  # 成熟阈值（≈高注意力聚焦 10 次）
    memory_salience_decay: float = 0.99    # 新鲜度衰减
    memory_consolidate_cooldown: int = 50  # 单条固化后冷却步数
    memory_consolidate_batch: int = 4      # 每步最多固化的记忆条数
    memory_sigma_strength: float = 0.2     # sigma 调制固化强度系数


@dataclass
class ReflexMediumConfig(ReflexConfig):
    """
    Scaled-up config for H800/H20 training.

    ~1.45B parameters, 16 layers, 8 experts/layer, d_model=1024.
    Modern architecture: GQA + RoPE + SwiGLU + RMSNorm + AttnRes.

    Depth-width balance: 16 layers (33% deeper than v1's 12) x 8 experts
    (33% narrower) addresses the "too shallow too wide" MoE problem.
    plastic(3) > top_k(2) ensures dialectical diversity -- not all
    plastic experts are activated simultaneously, allowing distinct
    thesis/antithesis modes to emerge.
    d_ff=2688 (21x128) is GPU tensor-core aligned.
    """
    d_model: int = 1024
    n_layers: int = 16
    n_heads: int = 16
    n_kv_heads: int = 4   # GQA 4:1 ratio
    d_ff: int = 2688      # 21x128, GPU tensor-core aligned
    n_stable: int = 5
    n_plastic: int = 3
    top_k: int = 2
    max_seq_len: int = 2048
    dropout: float = 0.1

    # RoPE
    rope_theta: float = 10000.0

    # AttnRes: 16 layers / 4 per block = 4 blocks -> 3 boundaries
    attnres_enabled: bool = True
    attnres_block_size: int = 4
    attnres_rank: int = 256  # d_model/4

    # Weight tying
    tie_word_embeddings: bool = True

    expert_baseline_lrs: Tuple[float, ...] = (
        1e-7, 1e-7, 1e-7, 1e-7, 1e-7,  # stable x5
        1e-5, 1e-4, 1e-4,               # plastic x3
    )

    self_model_z_dim: int = 128
    self_model_hidden_dim: int = 1024
    critic_hidden_dim: int = 512
    max_new_tokens: int = 192


@dataclass
class ReflexMiniConfig(ReflexConfig):
    """
    Mini config for 32GB GPU training.

    ~0.77B parameters, 24 layers, 6 experts/layer, d_model=640.
    Same core architecture: GQA + RoPE + SwiGLU + RMSNorm + AttnRes +
    SelfModel + DialecticalBuffer + Hebbian + EWC.

    Design choices:
    - 24 layers (same depth as Qwen2.5-0.5B) for hierarchical reasoning
    - 6 experts (4S+2P) with top_k=2: 27.5% activation (212M active/token)
    - d_model=640 (5x128, GPU aligned) -- moderate backbone
    - d_ff=2048 (16x128, 3.2x SwiGLU expansion, GPU tensor-core aligned)
    - Per expert: 4.4M params -- not too wide (3.2x), not too shallow
    - AttnRes block_size=6: 24/6=4 blocks -> 3 boundaries
    - All-expert mode (top_k=6): 636M active (82.6%)
    - 15B pretrain tokens -> 19.5x params = 97% Chinchilla (near optimal)
    """
    d_model: int = 640
    n_layers: int = 24
    n_heads: int = 10
    n_kv_heads: int = 2   # GQA 5:1 ratio
    d_ff: int = 2048      # 16x128, GPU tensor-core aligned, 3.2x expansion
    n_stable: int = 4
    n_plastic: int = 2
    top_k: int = 2
    max_seq_len: int = 2048
    dropout: float = 0.1

    # RoPE
    rope_theta: float = 10000.0

    # AttnRes: 24 layers / 6 per block = 4 blocks -> 3 boundaries
    attnres_enabled: bool = True
    attnres_block_size: int = 6
    attnres_rank: int = 160  # d_model/4

    # Weight tying
    tie_word_embeddings: bool = True

    expert_baseline_lrs: Tuple[float, ...] = (
        1e-7, 1e-7, 1e-7, 1e-7,  # stable x4
        1e-5, 1e-4,               # plastic x2
    )

    self_model_z_dim: int = 128
    self_model_hidden_dim: int = 640
    critic_hidden_dim: int = 320
    max_new_tokens: int = 192
