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

    # ── Backbone（嫁接扩展）──
    # 'reflex' = 原生 MoE 主干；'deepseek_v2' = DeepSeek-V2-Lite 嫁接主干
    backbone: str = 'reflex'

    # ── RoPE ──
    rope_theta: float = 10000.0
    # partial rotary（Qwen3.x 风格字段，本项目保留兼容）
    partial_rotary_factor: float = 1.0

    # ── Attention 扩展 ──
    # 显式 head_dim（0 = 自动 d_model // n_heads）
    head_dim: int = 0
    # attn_output_gate（Qwen3.x 字段，本项目保留兼容）
    attn_gate: bool = False

    # ── Attention Residuals (Kimi AttnRes) ──
    attnres_enabled: bool = True
    attnres_block_size: int = 2  # layers per block
    attnres_rank: int = 128     # low-rank key dimension for depth-wise attention
    attnres_postnorm_init: float = 0.1

    # ── Weight tying ──
    tie_word_embeddings: bool = True

    # ── Expert LR spectrum ──
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
    verify_warmup_steps: int = 500
    verify_warmup_cooldown: int = 50

    # ── SelfModel ──
    self_model_enabled: bool = True
    self_model_z_dim: int = 64
    self_model_hidden_dim: int = 512
    self_model_n_prior_experts: int = 3
    self_model_n_post_experts: int = 3

    # ── Curiosity drive ──
    curiosity_beta: float = 0.1
    imagination_lambda: float = 1.0
    stability_lambda: float = 0.01
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

    # ── Feedback ──
    feedback_lr: float = 1e-5
    feedback_max_grad_norm: float = 1.0
    feedback_gamma_modulation_strength: float = 0.3
    feedback_alignment_weight: float = 0.5
    feedback_align_loss_weight: float = 0.2
    sigma_calibration_weight: float = 0.2
    feedback_alignment_threshold: float = 0.2

    # ── Sampling ──
    sampling_temperature: float = 0.8
    sampling_top_k: int = 40
    sampling_top_p: float = 0.9
    sampling_repetition_penalty: float = 1.5

    # ── Gradient management ──
    gradient_per_layer: bool = True

    # ── Others ──
    internal_steps_per_cycle: int = 1000
    internal_step_delay_ms: float = 5.0
    internal_loss_clip: float = 10.0
    endosphere_capacity: int = 1024
    replay_capacity: int = 10000
    supervised_enabled: bool = False
    supervised_batch_size: int = 4
    supervised_lr: float = 2e-5
    max_new_tokens: int = 256

    # ── Memory system ──
    dialog_memory_alpha: float = 0.3
    memory_enabled: bool = True
    memory_bank_capacity: int = 128
    memory_context_top_k: int = 8
    memory_write_lr: float = 0.05
    kv_cache_rounds: int = 4
    memory_distill_enabled: bool = True
    memory_distill_batch: int = 8
    memory_salience_enabled: bool = True
    memory_salience_threshold: float = 3.0
    memory_salience_decay: float = 0.99
    memory_consolidate_cooldown: int = 50
    memory_consolidate_batch: int = 4
    memory_sigma_strength: float = 0.2

    # ── 会话提问上限（0=无限）──
    max_questions_per_session: int = 5


@dataclass
class ReflexMiniConfig(ReflexConfig):
    """Mini config for 32GB GPU training (~0.77B)."""
    d_model: int = 640
    n_layers: int = 24
    n_heads: int = 10
    n_kv_heads: int = 2
    d_ff: int = 2048
    n_stable: int = 4
    n_plastic: int = 2
    top_k: int = 2
    max_seq_len: int = 2048
    dropout: float = 0.1
    rope_theta: float = 10000.0
    attnres_enabled: bool = True
    attnres_block_size: int = 6
    attnres_rank: int = 160
    tie_word_embeddings: bool = True
    expert_baseline_lrs: Tuple[float, ...] = (
        1e-7, 1e-7, 1e-7, 1e-7,  # stable x4
        1e-5, 1e-4,               # plastic x2
    )
    self_model_z_dim: int = 128
    self_model_hidden_dim: int = 640
    critic_hidden_dim: int = 320
    max_new_tokens: int = 192


@dataclass
class DeepSeekV2GraftConfig(ReflexConfig):
    """
    DeepSeek-V2-Lite 嫁接配置 —— 将 DeepSeek-V2-Lite 原始权重作为 Reflex 主干。

    架构（deepseek-ai/DeepSeek-V2-Lite, 15.7B 总参数 / 2.4B 激活）:
      - 27 层；第 0 层 dense FFN（intermediate 10944），第 1-26 层 MoE
      - MLA（Multi-head Latent Attention）: kv_lora_rank=512, qk_nope=128,
        qk_rope=64, v_head_dim=128, 16 头；q 无低秩压缩（q_lora_rank=null）
      - MoE: 64 路由专家（moe_intermediate=1408, top-6, greedy, softmax 分数
        不归一化 norm_topk_prob=false）+ 2 共享专家（权重为合并 MLP
        intermediate=1408×2=2816，本嫁接拆分为两个独立专家——数学等价，
        实现"共享专家一高一低学习率"）
      - YaRN RoPE（half-split 旋转布局，θ=1e4, factor=40, mscale=0.707）
      - RMSNorm 标准实现（与项目同构，无 1+w 变换）
      - 词表 102400，不绑定权重；原生 163840 上下文（部署上限另行设定）
    权重加载见 scripts/load_deepseek_graft.py（共享专家行/列切片拆分）。
    """
    backbone: str = 'deepseek_v2'

    # ── 主动求证阈值与提问上限 ──
    verify_threshold: float = 0.5
    max_questions_per_session: int = 0   # 用户要求不限制提问/追问次数

    # ── 主干几何（DeepSeek-V2-Lite config.json 实测）──
    d_model: int = 2048
    n_layers: int = 27
    n_heads: int = 16
    head_dim: int = 192                  # q_head_dim = qk_nope(128) + qk_rope(64)
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    kv_lora_rank: int = 512
    v_head_dim: int = 128
    intermediate_size: int = 10944       # dense 层 FFN（层 0）
    moe_intermediate_size: int = 1408    # 每个路由专家
    vocab_size: int = 102400
    tie_word_embeddings: bool = False
    max_seq_len: int = 8192              # 部署截断上限（原生 163840）
    dropout: float = 0.0
    attention_dropout: float = 0.0
    attention_bias: bool = False
    rms_norm_eps: float = 1e-6

    # ── RoPE（YaRN, half-split 布局，官方 rotate_half 语义）──
    rope_theta: float = 10000.0
    rope_scaling: dict = field(default_factory=lambda: {
        'type': 'yarn',
        'factor': 40.0,
        'original_max_position_embeddings': 4096,
        'beta_fast': 32,
        'beta_slow': 1,
        'mscale': 0.707,
        'mscale_all_dim': 0.707,
    })

    # ── 层类型（加载器按 first_k_dense_replace 生成；显式传入可覆盖）──
    layer_types: Tuple[str, ...] = ()

    # ── MoE（专家数与 DeepSeek-V2-Lite 对齐）──
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    num_experts_per_tok: int = 6
    first_k_dense_replace: int = 1       # 前 1 层 dense
    moe_layer_freq: int = 1
    norm_topk_prob: bool = False         # DeepSeek-V2-Lite 不归一化 top-k 权重
    topk_method: str = 'greedy'
    n_group: int = 1
    topk_group: int = 1
    scoring_func: str = 'softmax'
    aux_loss_alpha: float = 0.001
    routed_scaling_factor: float = 1.0
    seq_aux: bool = True

    # ── 专家学习率光谱（用户方案：路由三阶梯；共享专家保持合并 MLP 完整）──
    # 64 路由专家三阶梯：16×1e-5 稳定 / 32×1e-4 中间 / 16×8e-2 可塑
    # （v3.9 整体升一量级、v3.10/v3.11 可塑档两连升、v3.12 可塑档 ×8
    #   且其余各升一量级——⚠ 8e-2 为极端档，实验性设置）。
    # 共享专家：DeepSeek 权重为合并 MLP（intermediate=1408×2=2816），
    # 为保持与原权重 1:1 直拷（零切片、零浮点舍入），不拆分、不设高低
    # 学习率，使用单一 shared_expert_lr。
    # 光谱只影响 Hebbian 学习率（部署即学习的强度分层），不影响前向。
    expert_baseline_lrs: Tuple[float, ...] = (
        (1e-5,) * 16 + (1e-4,) * 32 + (8e-2,) * 16
    )
    shared_expert_lr: float = 1e-4         # 共享专家（合并 MLP）学习率
    dense_expert_lr: float = 1e-5          # dense 层（层 0）专家学习率

    # ── AttnRes（Reflex 附加件，随机初始化；post_norm 1e-3 起步近零）──
    attnres_enabled: bool = True
    attnres_block_size: int = 3           # 27/3 = 9 块 → 8 个边界
    attnres_rank: int = 512               # d_model/4
    attnres_postnorm_init: float = 0.001

    # ── Reflex 附加件尺寸（d_model=2048）──
    self_model_z_dim: int = 128
    self_model_hidden_dim: int = 512
    critic_hidden_dim: int = 256
    endosphere_capacity: int = 1024
    max_new_tokens: int = 900             # 对话回答上限（用户要求）
    memory_bank_capacity: int = 128
    memory_context_top_k: int = 8
    memory_write_lr: float = 0.01         # 记忆写入更稳

    # ── 嫁接运行模式（与 Qwen 嫁接一致的基础设施）──
    graft_use_past: bool = True           # generate 增量解码（MLA 展开 KV）
    graft_lite: bool = True               # 轻量在线学习（冻结主干 + Hebbian 尾层）
    graft_hebbian_layers: int = 8         # Hebbian 梯度覆盖最后 8 层（--full-graft 全 27 层）
    graft_freeze_backbone: bool = True    # 全局优化器排除主干权重
    graft_disable_consolidation: bool = True
    graft_decode_attnres: bool = False
    graft_online_ce: bool = True
    graft_verify_max_tokens: int = 512    # 主动求证问题生成上限
    graft_think_eos_grace: int = 2        # think 未闭合时终止符宽容次数
    graft_gen_debug: bool = False
    graft_sigma_cal: bool = False         # sigma 在线校准（--sigma-cal 开启）
    graft_sigma_cal_interval: int = 20


def config_from_checkpoint(ck: dict):
    """从 checkpoint 的 config 字段重建配置对象（嫁接/原生通用）。"""
    cfg_dict = ck.get('config') if isinstance(ck, dict) else None
    if not cfg_dict:
        return ReflexMiniConfig()
    if cfg_dict.get('backbone') == 'deepseek_v2':
        fields = DeepSeekV2GraftConfig.__dataclass_fields__
        kwargs = {k: v for k, v in cfg_dict.items() if k in fields}
        return DeepSeekV2GraftConfig(**kwargs)
    fields = ReflexMiniConfig.__dataclass_fields__
    kwargs = {k: v for k, v in cfg_dict.items() if k in fields}
    return ReflexMiniConfig(**kwargs)
