"""
NeuroStream-Reflex 单卡云端训练脚本 (H800 80GB)

为单卡 H800 80GB 优化：
  - micro-batch 64/32/16，gradient accumulation 2 步
  - 等效 batch 128/64/32，匹配 multi-GPU 方案的总 batch
  - 自动检测 DDP 环境，兼容 torch.distributed.run（但推荐单卡）

三阶段训练：
  1. 继续预训练 (Wudao 10% / SkyPile) - 20B tokens
  2. SFT 微调 (Firefly + BELLE) - 2B tokens
  3. 蒸馏增强 (Self-Instruct / Evol-Instruct) - 500M tokens

数据下载 (国内镜像)：
  - Wudao 2.0 10%:  https://opendatalab.com  → /data/wudao/wudao_10pct.jsonl
  - Firefly 1.6M:   https://hf-mirror.com/YangyiYin/Firefly  → /data/firefly/firefly_1.6m.jsonl
  - BELLE:          https://hf-mirror.com/BelleGroup  → 合并到 SFT 数据
  Teacher 模型 (Qwen2.5-7B-Instruct) 和 Tokenizer 在启动时自动从镜像下载。
  HF_ENDPOINT 环境变量在 run_training.sh 中预设为 https://hf-mirror.com。

用法 (单卡 H800):
  python train/train_reflex.py --mode pretrain --device cuda --dtype bfloat16
  python train/train_reflex.py --mode sft --resume pretrain_final.pt
  python train/train_reflex.py --mode distill --resume sft_final.pt
  python train/train_reflex.py --mode full
"""
import sys, os, json, math, time, gc, argparse, glob
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from config.model_config import ReflexMiniConfig
from core.model import ReflexModel
from train.data_pipeline import (
    StreamingDataset, DataConfig, collate_fn, generate_teacher_logits,
)
from train.monitor import ReflexMonitor, MonitorConfig

# ── Training config ────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    mode: str = 'full'            # pretrain | sft | distill | full
    device: str = 'cuda'
    dtype: str = 'bfloat16'       # bfloat16 / float16 / float32
    seed: int = 42

    # Paths
    pretrain_data: str = '/data/wudao/wudao_30pct.jsonl'
    sft_data: str = '/data/firefly/firefly_1.6m.jsonl'
    distill_data: str = '/data/sft_all.jsonl'
    teacher_name: str = 'Qwen/Qwen2.5-1.5B-Instruct'
    output_dir: str = '/checkpoints/reflex-mini'
    resume: Optional[str] = None
    fresh_optimizer: bool = False  # 精修阶段：resume 权重但用全新 optimizer/scheduler

    # Pretrain (32GB GPU, bf16 + gradient checkpointing)
    # 228883 steps x 65536 = 15.0B tokens (19.5x params, 97% Chinchilla)
    pretrain_steps: int = 228883
    pretrain_batch_size: int = 8
    pretrain_grad_accum: int = 8
    pretrain_lr: float = 3e-4
    pretrain_warmup: int = 2000

    # SFT: Firefly(50%)+BELLE(50%)+ShareGPT+COIG = 122.5万条, 2 epochs
    # 38000 steps x 65536 = 2.5B tokens, 2.0 epochs
    # batch=32, seq=2048 (short SFT data auto-pads less via collate_fn)
    sft_steps: int = 38000
    sft_batch_size: int = 16
    sft_grad_accum: int = 4
    sft_lr: float = 1e-5
    sft_warmup: int = 500

    # Distill (Qwen2.5-1.5B-Instruct teacher, offline logits)
    # 10000 steps, batch=16, seq=2048
    # Mem: model 1.5G + optim 5.5G + grad 1.4G + act 4G + teacher_logits 2G = ~14.4G / 32G
    distill_steps: int = 10000
    distill_batch_size: int = 16
    distill_grad_accum: int = 2
    distill_lr: float = 2e-5
    distill_warmup: int = 200
    distill_temperature: float = 2.0
    distill_ce_weight: float = 0.3
    distill_kd_weight: float = 0.7

    # Common
    max_seq_len: int = 1024  # 训练用1024, 推理用2048(RoPE支持)
    grad_clip: float = 1.0
    save_every: int = 2000
    log_every: int = 50
    val_every: int = 1000


def _zero_grad(optimizer, scaler):
    """Compat helper for scaler-aware zero_grad."""
    optimizer.zero_grad()


def _backward(loss, scaler):
    """Compat helper for scaler-aware backward."""
    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()


def _step(optimizer, scaler):
    """Compat helper for scaler-aware optimizer step."""
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()


# ── Cosine LR scheduler with warmup ────────────────────────────────────


def cosine_schedule(step, warmup, total, min_factor=0.1, plateau=0.3):
    """warmup -> plateau (peak hold) -> cosine decay to min_factor*peak.

    plateau: fraction of post-warmup steps held at peak LR.
    Without a plateau, short runs (few hundred opt steps) spend the whole
    schedule climbing and never learn at peak LR.
    """
    if step < warmup:
        return step / max(1, warmup)
    plateau_steps = int((total - warmup) * plateau)
    if step < warmup + plateau_steps:
        return 1.0
    progress = (step - warmup - plateau_steps) / max(1, total - warmup - plateau_steps)
    return min_factor + 0.5 * (1.0 - min_factor) * (1.0 + math.cos(math.pi * progress))


def chunked_loss(model, hidden, labels, scaler=None, accum=1,
                 chunk_size=256, ignore_index=-100):
    """
    分块计算 cross_entropy，每块算完立即 backward 释放显存。
    vocab=151936 时 B=32 chunk=256: 每块 logits=2.5GB + CE=5GB ≈ 7.5GB
    """
    B, T, D = hidden.shape
    V = model.config.vocab_size
    n_chunks = (T + chunk_size - 1) // chunk_size

    hd = hidden.detach().requires_grad_(True)

    total_loss = 0.0
    total_count = 0

    for i in range(0, T, chunk_size):
        chunk = hd[:, i:i + chunk_size, :].contiguous()
        logits = model.lm_head(chunk)
        label_chunk = labels[:, i:i + chunk_size].contiguous()

        count = (label_chunk != ignore_index).sum()
        if count.item() == 0:
            del logits, label_chunk, chunk
            continue

        ce_sum = F.cross_entropy(
            logits.view(-1, V),
            label_chunk.view(-1),
            ignore_index=ignore_index,
            reduction='sum',
        )
        loss = ce_sum / count.float() / n_chunks / accum

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        total_loss += loss.item() * n_chunks * accum
        total_count += count.item()

        del logits, loss, chunk, label_chunk, ce_sum

    if hd.grad is not None:
        hidden.backward(hd.grad)
    del hd

    return torch.tensor(total_loss / max(n_chunks, 1), device=hidden.device)


def chunked_distill_loss(model, hidden, labels, teacher_topk,
                         scaler=None, accum=1, chunk_size=256,
                         temperature=2.0, ce_weight=0.3, kd_weight=0.7):
    """
    分块计算 distill loss (CE + 稀疏 KL)，避免 vocab=151936 的完整 logits 爆显存。

    Sparse KD (correct up to teacher-entropy constant):
      teacher stores top-K FULL-distribution log-probs (t_logp_k)
      p_k = exp(t_logp_k),  p_rest = 1 - sum(p_k)
      q_k = student softmax at those indices (full-dist)
      q_rest = 1 - sum(q_k)
      CE_sparse = -[ sum(p_k log q_k) + p_rest log q_rest ]
      (minimizing CE == minimizing KL, since teacher entropy is constant)
    """
    B, T, D = hidden.shape
    V = model.config.vocab_size
    n_chunks = (T + chunk_size - 1) // chunk_size
    tk_idx = teacher_topk[0]  # [B, T, K]
    tk_logp = teacher_topk[1]  # [B, T, K]

    hd = hidden.detach().requires_grad_(True)

    total_loss = 0.0
    total_ce = 0.0
    total_kld = 0.0

    for i in range(0, T, chunk_size):
        j = min(i + chunk_size, T)
        chunk = hd[:, i:j, :].contiguous()
        student_logits = model.lm_head(chunk)          # [B, c, V]
        label_chunk = labels[:, i:j].contiguous()       # [B, c]
        tk_i = tk_idx[:, i:j, :].contiguous()           # [B, c, K]
        tk_lp = tk_logp[:, i:j, :].contiguous()         # [B, c, K]

        mask = (label_chunk != -100).float()
        count = mask.sum()
        if count.item() == 0:
            del student_logits, label_chunk, tk_i, tk_lp, chunk, mask
            continue

        # Cross-entropy (hard labels)
        ce_sum = F.cross_entropy(
            student_logits.view(-1, V),
            label_chunk.view(-1),
            ignore_index=-100,
            reduction='sum',
        )

        # Sparse KL / CE (temperature-scaled)
        T_kd = temperature
        # Student full softmax probs
        q_full = F.softmax(student_logits / T_kd, dim=-1)      # [B, c, V]
        # Teacher probs at top-K (full-distribution log-probs)
        p_k = torch.exp(tk_lp.float()).clamp(min=1e-8)          # [B, c, K]
        # Student probs at same indices
        q_k = torch.gather(q_full, -1, tk_i.clamp(0, V - 1))    # [B, c, K]
        # Probability mass outside top-K
        p_rest = (1.0 - p_k.sum(dim=-1)).clamp(min=1e-8)       # [B, c]
        q_rest = (1.0 - q_k.sum(dim=-1)).clamp(min=1e-8)       # [B, c]
        # Sparse cross-entropy (minimize == minimize KL)
        ce_sparse = -(p_k * torch.log(q_k.clamp(min=1e-8))).sum(dim=-1)
        ce_sparse = ce_sparse - p_rest * torch.log(q_rest)      # [B, c]
        kld_sum = (ce_sparse * mask).sum()

        loss = (ce_weight * ce_sum + kd_weight * kld_sum * T_kd * T_kd)
        loss = loss / count.float() / n_chunks / accum

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        total_loss += loss.item() * n_chunks * accum
        total_ce += ce_sum.item() / count.item() / n_chunks
        total_kld += kld_sum.item() / count.item() / n_chunks

        del student_logits, label_chunk, tk_i, tk_lp, chunk, mask
        del q_full, p_k, q_k, ce_sum, kld_sum, loss

    if hd.grad is not None:
        hidden.backward(hd.grad)
    del hd

    return (torch.tensor(total_loss / max(n_chunks, 1), device=hidden.device),
            total_ce, total_kld)


# ── Training loop ──────────────────────────────────────────────────────

class ReflexTrainer:
    def __init__(self, model, tokenizer, cfg: TrainConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.dtype = {
            'bfloat16': torch.bfloat16,
            'float16': torch.float16,
            'float32': torch.float32,
        }[cfg.dtype]
        self.step = 0
        self.best_val_loss = float('inf')
        # M3 fix: monitor 的 LR 缩放乘数（scheduler.step 从 base_lrs×λ 重算会覆盖，
        # 缩放需在 step 后重新应用并跨步持久化）
        self._lr_scale = 1.0

        # DDP support
        self.is_ddp = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.is_ddp else 0
        self.world_size = dist.get_world_size() if self.is_ddp else 1
        self.is_main = (self.rank == 0)

        # Wrap model with DDP if distributed
        if self.is_ddp:
            self.model = DDP(self.model, device_ids=[self.rank] if self.device.type == 'cuda' else None,
                             find_unused_parameters=True)
            self._raw_model = self.model.module
        else:
            self._raw_model = self.model

        # Disable Hebbian buffers during training (saves ~8 GB)
        self._raw_model._save_hebbian_buffers = False

        # 冻结预训练不需要的模块，节省优化器内存
        for name, param in self._raw_model.named_parameters():
            if any(k in name for k in ['self_model', 'contemplator', 'critic', 'query_proj',
                                        'uncertainty_head', 'lr_bias', 'verify_threshold',
                                        'endo_proj', 'endo_gate_proj', 'h_to_bias_weight']):
                param.requires_grad = False

        # Monitor for gradient/expert/sigma stability
        # 预训练阶段 sigma 自然趋同，禁用 sigma 崩塌检测避免误报
        self.monitor = ReflexMonitor(model, MonitorConfig(
            grad_norm_window=100,
            grad_explosion_threshold=10.0,
            grad_vanishing_threshold=1e-6,
            grad_explosion_patience=3,
            grad_vanishing_patience=10,
            expert_activation_window=500,
            dead_expert_threshold=0.01,
            dead_expert_patience=2000,
            sigma_window=200,
            sigma_collapse_threshold=0.001,
            sigma_collapse_patience=999999,
            weight_growth_threshold=10.0,
            weight_growth_patience=1000,
            lr_scale_on_explosion=0.5,
            noise_inject_on_collapse=0.05,
            max_resets_per_expert=3,
        ))

    def save_checkpoint(self, path, loss, val_loss=None, is_best=False):
        if not self.is_main:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        phase = 'pretrain'
        if 'sft_' in os.path.basename(path):
            phase = 'sft'
        elif 'distill_' in os.path.basename(path):
            phase = 'distill'
        elif 'warmup_' in os.path.basename(path):
            phase = 'warmup'
        # 滚动清理：先删最旧 checkpoint，确保磁盘空间足够再写新文件
        self._prune_checkpoints(phase)
        ckpt = {
            'model_state_dict': self._to_save_dtype(self._raw_model.state_dict()),
            'config': self._raw_model.config,
            'step': self.step,
            'phase': phase,
            'loss': loss,
            'val_loss': val_loss,
            'optimizer': self.optimizer.state_dict() if hasattr(self, 'optimizer') else None,
            'scheduler': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
        }
        # 原子写：先写 .tmp 再 rename，磁盘满时不会损坏已有 checkpoint
        tmp_path = path + '.tmp'
        torch.save(ckpt, tmp_path)
        if os.path.exists(path):
            os.remove(path)
        os.replace(tmp_path, path)
        if is_best:
            best_path = path.replace('.pt', '_best.pt')
            torch.save(ckpt, best_path)
        print(f"  [save] {path}  loss={loss:.4f}" +
              (f" val={val_loss:.4f}" if val_loss else ""))

    @staticmethod
    def _to_save_dtype(state_dict, dtype=torch.float16):
        """Cast float params to fp16 for storage (halves ckpt size); load cast back."""
        return {k: (v.to(dtype) if v.is_floating_point() else v)
                for k, v in state_dict.items()}

    def _prune_checkpoints(self, phase, keep_last=2):
        """Keep only the newest `keep_last` step checkpoints for this phase."""
        if not self.is_main:
            return
        files = glob.glob(
            os.path.join(self.cfg.output_dir, f'{phase}_step*.pt'))
        # 按步数数值排序（字符串排序在 9999→10000 边界会错序）
        def _step_of(path):
            base = os.path.basename(path)
            num = base.replace(f'{phase}_step', '').replace('.pt', '')
            try:
                return int(num)
            except ValueError:
                return -1
        files.sort(key=_step_of)
        for f in files[:-keep_last]:
            try:
                os.remove(f)
                print(f"  [prune] removed {f}")
            except OSError:
                pass
        # Clean up orphaned atomic-write temp files from interrupted saves
        for f in glob.glob(os.path.join(self.cfg.output_dir, f'{phase}_step*.tmp')):
            try:
                os.remove(f)
            except OSError:
                pass

    def load_checkpoint(self, path):
        print(f"Loading checkpoint: {path}")
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'], strict=False)
        self.step = ckpt.get('step', 0)
        self._loaded_ckpt = ckpt
        print(f"  resumed from step {self.step}")
        return ckpt

    def _resume_phase(self, phase):
        """True if the loaded checkpoint belongs to the same training phase."""
        ckpt = getattr(self, '_loaded_ckpt', None)
        if ckpt is None:
            return False
        return ckpt.get('phase', '') == phase

    def _restore_optimizer_state(self, phase=None):
        """Restore optimizer+scheduler state after _build_optimizer.

        Must be called AFTER _build_optimizer (the optimizer doesn't exist
        during main()'s load_checkpoint).

        phase (e.g. 'sft'): current training phase.  On CROSS-phase resume
        (e.g. pretrain ckpt loaded for SFT), the optimizer/scheduler are NOT
        restored -- the fresh ones (with this phase's base LR) are used.
        """
        ckpt = getattr(self, '_loaded_ckpt', None)
        if ckpt is None:
            return
        if getattr(self.cfg, 'fresh_optimizer', False):
            print("  [INFO] --fresh-optimizer: using brand-new optimizer/scheduler "
                  "(weights resumed, optimizer reset for refinement stage)")
            self._loaded_ckpt = None
            return
        if phase is not None and ckpt.get('phase', '') != phase:
            print(f"  [INFO] Cross-phase resume "
                  f"({ckpt.get('phase', '?')} -> {phase}): "
                  f"fresh optimizer/scheduler (step reset by caller)")
            self._loaded_ckpt = None
            return
        try:
            if 'optimizer' in ckpt and ckpt['optimizer'] is not None:
                self.optimizer.load_state_dict(ckpt['optimizer'])
            if 'scheduler' in ckpt and ckpt['scheduler'] is not None \
                    and hasattr(self, 'scheduler'):
                self.scheduler.load_state_dict(ckpt['scheduler'])
            print("  [INFO] Optimizer/Scheduler state restored")
        except Exception as e:
            print(f"  [WARN] Optimizer state not restored (fresh start): {e}")
        self._loaded_ckpt = None  # free memory

    def _build_optimizer(self, lr, warmup, total_steps):
        trainable = [p for p in self._raw_model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable, lr=lr, weight_decay=0.01,
            betas=(0.9, 0.95),
        )
        # 自适应 warmup：不超过总步数 1/8，保证有 plateau + 衰减空间
        warmup = min(warmup, max(1, total_steps // 8))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda s: cosine_schedule(s, warmup, total_steps,
                                      min_factor=0.1, plateau=0.3),
        )

    def _log(self, metrics):
        desc = ' | '.join(f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}' for k, v in metrics.items())
        tqdm.write(f"  step {self.step:>6d} | {desc}")

    def _get_attnres_stats(self):
        """Get AttnRes routing stats for logging. Returns dict or empty dict."""
        raw_model = self._raw_model
        if not hasattr(raw_model, 'attn_res') or raw_model.attn_res is None:
            return {}
        stats = raw_model.attn_res.get_all_routing_stats()
        if not stats:
            return {}
        result = {}
        for boundary_idx, s in stats:
            result[f'ar{boundary_idx}_max'] = s['max_weight']
            result[f'ar{boundary_idx}_ent'] = s['entropy']
        return result

    # ── Phase 1: Continue pretraining ──

    def train_pretrain(self):
        print(f"\n{'='*60}")
        print(f"Phase 1: Continue pretraining")
        print(f"  Data: {self.cfg.pretrain_data}")
        print(f"  Steps: {self.cfg.pretrain_steps}")
        print(f"  Batch (micro): {self.cfg.pretrain_batch_size}")
        print(f"  Grad accum: {self.cfg.pretrain_grad_accum}")
        print(f"  Effective batch: {self.cfg.pretrain_batch_size * self.cfg.pretrain_grad_accum}")
        print(f"  LR: {self.cfg.pretrain_lr}")
        print(f"{'='*60}\n")

        # ── Debug: print CUDA memory just before first training step ──
        if self.is_main and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            print(f"  [MEM] Model+Optim loaded: "
                  f"{torch.cuda.memory_allocated()/1e9:.2f}GB allocated, "
                  f"{torch.cuda.max_memory_allocated()/1e9:.2f}GB peak")

        dataset = StreamingDataset(
            self.tokenizer, DataConfig(max_seq_len=self.cfg.max_seq_len),
            pretrain_paths=[self.cfg.pretrain_data],
        )
        if self.is_ddp:
            sampler = DistributedSampler(dataset, shuffle=True,
                                         num_replicas=self.world_size,
                                         rank=self.rank)
            loader = DataLoader(
                dataset, batch_size=self.cfg.pretrain_batch_size,
                collate_fn=collate_fn, num_workers=4, sampler=sampler,
            )
        else:
            sampler = None
            loader = DataLoader(
                dataset, batch_size=self.cfg.pretrain_batch_size,
                collate_fn=collate_fn, num_workers=4,
            )
        self._build_optimizer(self.cfg.pretrain_lr, self.cfg.pretrain_warmup,
                              self.cfg.pretrain_steps // self.cfg.pretrain_grad_accum)
        self._restore_optimizer_state('pretrain')
        data_iter = iter(loader)

        self.model.train()
        scaler = torch.amp.GradScaler(enabled=(self.dtype != torch.float32))
        pbar = tqdm(range(self.step, self.cfg.pretrain_steps), initial=self.step, desc='Pretrain')

        accum = self.cfg.pretrain_grad_accum
        _zero_grad(self.optimizer, scaler)

        for step in pbar:
            self.step = step
            if sampler is not None and step % len(loader) == 0:
                sampler.set_epoch(step // len(loader))
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            # ---- Memory debug on step 0 ----
            if step == 0 and self.is_main:
                torch.cuda.reset_peak_memory_stats()
                a = torch.cuda.memory_allocated() / 1e9
                r = torch.cuda.memory_reserved() / 1e9
                print(f"  [MEM] Before fwd: {a:.2f}G alloc {r:.2f}G rsv")
                for n, p in self._raw_model.named_parameters():
                    if p.is_cuda and p.numel() > 5e7 and p.requires_grad:
                        print(f"    {n[-30:]}: {p.numel()//1e6:.0f}M {p.element_size()*p.numel()//1e9:.2f}G")

            with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype, enabled=(self.dtype != torch.float32)):
                hidden = self.model(input_ids, save_hebbian_buffers=False, return_hidden=True)
                loss = chunked_loss(self._raw_model, hidden, labels, scaler=scaler, accum=accum)

            # ---- Memory after chunked_loss ----
            if step == 0 and self.is_main:
                a = torch.cuda.memory_allocated() / 1e9
                r = torch.cuda.memory_reserved() / 1e9
                print(f"  [MEM] After chunk: {a:.2f}G alloc {r:.2f}G rsv")
                print(torch.cuda.memory_summary(abbreviated=True))

            del hidden
            self._raw_model._last_layer_outputs = {}

            is_update_step = (step % accum == accum - 1) or (step == self.cfg.pretrain_steps - 1)
            if is_update_step:
                # Unscale FIRST so the monitor reads TRUE gradients
                # (GradScaler scales gradients by ~65536; reading before
                # unscale_ would falsely trigger grad_explosion and crush LR)
                if scaler:
                    scaler.unscale_(self.optimizer)
                # Monitor on true gradients
                sigma = getattr(self._raw_model, '_last_sigma_aggregate', 0.5)
                actions, grad_norms = self.monitor.step(loss * accum, sigma, self.step)
                # Now safe to clear expert sigmas
                self._raw_model._last_expert_sigmas = None
                self._raw_model._last_token_sigmas = None
                if actions:
                    old_lr = self.optimizer.param_groups[0]['lr']
                    new_lr = self.monitor.apply_actions(actions, self.optimizer)
                    if new_lr is not None and old_lr > 0:
                        self._lr_scale *= (new_lr / old_lr)
                        pbar.set_postfix({'LR_SCALED': f'{new_lr:.2e}'})

                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                _step(self.optimizer, scaler)
                _zero_grad(self.optimizer, scaler)
                gc.collect()
                torch.cuda.empty_cache()
                self.scheduler.step()
                # M3 fix: scheduler 重算会覆盖 monitor 缩放，此处重新应用持久化乘数
                if self._lr_scale != 1.0:
                    for _g in self.optimizer.param_groups:
                        _g['lr'] *= self._lr_scale

            # Log every log_every steps (OUTSIDE is_update_step: log_every
            # and update-step conditions can be mutually exclusive)
            if step % self.cfg.log_every == 0:
                lr = self.scheduler.get_last_lr()[0] if hasattr(self, 'scheduler') else 0
                postfix = {'loss': f'{(loss * accum).item():.4f}', 'lr': f'{lr:.2e}'}
                ar_stats = self._get_attnres_stats()
                for k, v in ar_stats.items():
                    postfix[k] = f'{v:.2f}'
                pbar.set_postfix(postfix)

            # Save every save_every steps (OUTSIDE is_update_step!
            # is_update_step requires step%8==7 but save requires step%2000==0;
            # 2000 is a multiple of 8 so the two conditions are mutually
            # exclusive - saves would NEVER run if nested inside.)
            if step > 0 and step % self.cfg.save_every == 0:
                self.save_checkpoint(
                    os.path.join(self.cfg.output_dir, f'pretrain_step{step}.pt'),
                    (loss * accum).item(),
                )

        # Final save
        # Note: the loop's last step (step == pretrain_steps - 1) already
        # performs the final optimizer step (is_update_step includes it).
        # Do NOT call _step again here - gradients are already zeroed and
        # scaler.step() would fail with "No inf checks were recorded".
        self.save_checkpoint(
            os.path.join(self.cfg.output_dir, 'pretrain_final.pt'),
            (loss * accum).item() if 'loss' in locals() else 0.0,
        )
        print("Pretrain complete.\n")

    # ── Phase 2: SFT fine-tuning ──

    def train_sft(self):
        # Only reset step if NOT resuming SFT (main() loads step from ckpt)
        if not self._resume_phase('sft'):
            self.step = 0
        for param in self._raw_model.parameters():
            param.requires_grad = True
        print(f"\n{'='*60}")
        print(f"Phase 2: SFT fine-tuning")
        print(f"  Data: {self.cfg.sft_data}")
        print(f"  Steps: {self.cfg.sft_steps}")
        print(f"  Batch (micro): {self.cfg.sft_batch_size}")
        print(f"  Grad accum: {self.cfg.sft_grad_accum}")
        print(f"  Effective batch: {self.cfg.sft_batch_size * self.cfg.sft_grad_accum}")
        print(f"  LR: {self.cfg.sft_lr}")
        print(f"{'='*60}\n")

        dataset = StreamingDataset(
            self.tokenizer, DataConfig(max_seq_len=self.cfg.max_seq_len),
            sft_paths=[self.cfg.sft_data],
        )
        if self.is_ddp:
            sampler = DistributedSampler(dataset, shuffle=True,
                                         num_replicas=self.world_size,
                                         rank=self.rank)
            loader = DataLoader(
                dataset, batch_size=self.cfg.sft_batch_size,
                collate_fn=collate_fn, num_workers=4, sampler=sampler,
            )
        else:
            sampler = None
            loader = DataLoader(
                dataset, batch_size=self.cfg.sft_batch_size,
                collate_fn=collate_fn, num_workers=4,
            )
        self._build_optimizer(self.cfg.sft_lr, self.cfg.sft_warmup,
                              self.cfg.sft_steps // self.cfg.sft_grad_accum)
        self._restore_optimizer_state('sft')
        data_iter = iter(loader)

        self.model.train()
        scaler = torch.amp.GradScaler(enabled=(self.dtype != torch.float32))
        pbar = tqdm(range(self.step, self.cfg.sft_steps), initial=self.step, desc='SFT')

        accum = self.cfg.sft_grad_accum
        _zero_grad(self.optimizer, scaler)

        for step in pbar:
            self.step = step
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype, enabled=(self.dtype != torch.float32)):
                hidden = self.model(input_ids, save_hebbian_buffers=False, return_hidden=True)
                loss = chunked_loss(self._raw_model, hidden, labels, scaler=scaler, accum=accum)

            del hidden
            self._raw_model._last_layer_outputs = {}

            is_update_step = (step % accum == accum - 1) or (step == self.cfg.sft_steps - 1)
            if is_update_step:
                # Unscale FIRST so the monitor reads TRUE gradients
                if scaler:
                    scaler.unscale_(self.optimizer)
                # Monitor on true gradients
                sigma = getattr(self._raw_model, '_last_sigma_aggregate', 0.5)
                actions, grad_norms = self.monitor.step(loss * accum, sigma, self.step)
                # Now safe to clear expert sigmas
                self._raw_model._last_expert_sigmas = None
                self._raw_model._last_token_sigmas = None
                if actions:
                    old_lr = self.optimizer.param_groups[0]['lr']
                    new_lr = self.monitor.apply_actions(actions, self.optimizer)
                    if new_lr is not None and old_lr > 0:
                        self._lr_scale *= (new_lr / old_lr)
                        pbar.set_postfix({'LR_SCALED': f'{new_lr:.2e}'})

                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                _step(self.optimizer, scaler)
                _zero_grad(self.optimizer, scaler)
                self.scheduler.step()
                # M3 fix: scheduler 重算会覆盖 monitor 缩放，此处重新应用持久化乘数
                if self._lr_scale != 1.0:
                    for _g in self.optimizer.param_groups:
                        _g['lr'] *= self._lr_scale

            # Log every log_every steps (OUTSIDE is_update_step)
            if step % self.cfg.log_every == 0:
                lr = self.scheduler.get_last_lr()[0]
                postfix = {'loss': f'{(loss * accum).item():.4f}', 'lr': f'{lr:.2e}'}
                ar_stats = self._get_attnres_stats()
                for k, v in ar_stats.items():
                    postfix[k] = f'{v:.2f}'
                pbar.set_postfix(postfix)

            if step > 0 and step % self.cfg.save_every == 0:
                self.save_checkpoint(
                    os.path.join(self.cfg.output_dir, f'sft_step{step}.pt'),
                    (loss * accum).item(),
                    )

        # Final save (loop's last step already did the optimizer step)
        self.save_checkpoint(
            os.path.join(self.cfg.output_dir, 'sft_final.pt'),
            (loss * accum).item() if 'loss' in locals() else 0.0,
        )
        print("SFT complete.\n")

    # ── Phase 2.5: Consciousness warmup ──

    def train_consciousness_warmup(self, steps=1000):
        """
        预热 SelfModel 和 Critic，消除训练-部署鸿沟。

        训练阶段 SelfModel/Critic 被冻结（随机初始化），部署时从零开始
        学习"如何思考"。此阶段用 SFT 数据的 embedding 驱动内循环，
        让 SelfModel 学会从输入产生有意义的 h_t/z_t，Critic 学会估计
        V(s)，KL 好奇心信号变得可靠。

        只更新 SelfModel + Critic，不更新专家（保护预训练知识）。
        """
        print(f"\n{'='*60}")
        print(f"Phase 2.5: Consciousness warmup ({steps} steps)")
        print(f"  Target: SelfModel (GRU+Prior+Posterior+Decoder)")
        print(f"          Critic (value estimator)")
        print(f"  Expert weights: FROZEN (preserve pretrained knowledge)")
        print(f"{'='*60}\n")

        raw = self._raw_model
        device = self.device

        # 1. Freeze everything, then unfreeze SelfModel + Critic
        for param in raw.parameters():
            param.requires_grad = False
        if raw.self_model is not None:
            for param in raw.self_model.parameters():
                param.requires_grad = True
        if hasattr(raw, 'critic') and raw.critic is not None:
            for param in raw.critic.parameters():
                param.requires_grad = True
        # P0-1 强化：解冻 uncertainty_head——warmup 同步做 sigma 校准
        # （让主动求证的 sigma 信号在部署前就对齐 CE 不确定性，而非随机起步）
        cal_params = []
        for layer in raw.layers:
            for exp in layer.all_experts:
                for p in exp.uncertainty_head.parameters():
                    p.requires_grad = True
                    cal_params.append(p)

        # 2. Create optimizers
        sm_params = list(raw.self_model.parameters()) if raw.self_model else []
        critic_params = list(raw.critic.parameters()) if hasattr(raw, 'critic') else []
        sm_opt = torch.optim.AdamW(sm_params, lr=1e-4, weight_decay=0.01) if sm_params else None
        critic_opt = torch.optim.AdamW(critic_params, lr=1e-3, weight_decay=0.01) if critic_params else None
        cal_opt = torch.optim.AdamW(cal_params, lr=1e-4) if cal_params else None

        if not sm_opt:
            print("  SelfModel not found, skipping warmup.")
            return

        # 3. Prepare data: reuse SFT data, extract embeddings
        from train.data_pipeline import StreamingDataset, DataConfig, collate_fn, set_pad_id
        from torch.utils.data import DataLoader
        dataset = StreamingDataset(
            self.tokenizer, DataConfig(max_seq_len=self.cfg.max_seq_len),
            sft_paths=[self.cfg.sft_data],
        )
        loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn, num_workers=2)
        data_iter = iter(loader)

        # 4. Initialize SelfModel state
        sm = raw.self_model
        h_t = torch.zeros(1, raw.config.d_model, device=device)
        z_t = torch.zeros(1, raw.config.sm_z_dim if hasattr(raw.config, 'sm_z_dim') else
                          getattr(raw.config, 'self_model_z_dim', 128), device=device)

        from learn.critic import compute_pseudo_reward
        from tqdm import tqdm

        raw.train()
        pbar = tqdm(range(steps), desc='Warmup')
        for step in pbar:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            input_ids = batch['input_ids'].to(device)
            with torch.no_grad():
                emb = raw.token_embedding(input_ids)
                # Mean-pool to get a state vector
                v_t = emb.mean(dim=1)  # [B, d_model]
                # Use first sample for internal loop (batch=1)
                v_t = v_t[:1]  # [1, d_model]

            # SelfModel forward
            action = v_t
            h_next, z_mean, z_logvar = sm(h_t, z_t, action)
            z_sample = sm.sample_z(z_mean, z_logvar)
            u_decoded = sm.decode(z_sample, h_next)

            # Internal forward (no grad for experts)
            with torch.no_grad():
                pred_emb = raw.forward_internal(
                    u_decoded.squeeze(0) if u_decoded.dim() == 3 else u_decoded,
                    h_state=h_next,
                )
                if pred_emb.dim() == 3:
                    pred_emb = pred_emb.squeeze(1)
                pred_emb = pred_emb.detach()

            # Posterior
            z_post_mean, z_post_logvar = sm.observe_and_correct(h_next, pred_emb)
            o_pred = sm.decode(z_sample, h_next)

            # Losses
            loss_imagination = torch.nn.functional.mse_loss(o_pred, pred_emb)
            loss_curiosity = sm.kl_divergence(z_post_mean, z_post_logvar, z_mean, z_logvar)
            loss_stability = torch.nn.functional.mse_loss(h_t, h_next)
            loss = (loss_imagination + 0.1 * loss_curiosity + 0.01 * loss_stability)

            sm_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sm_params, 1.0)
            sm_opt.step()

            # Critic update
            if critic_opt and hasattr(raw, 'critic'):
                loss_val = loss.item()
                pseudo_r = compute_pseudo_reward(loss_int=loss_val, sigma_aggregate=0.5)
                v_pred = raw.critic(v_t).squeeze()
                # Warmup bootstrap: target = r (pure MC).  TD bootstrap with
                # the PREVIOUS step's V would learn a backward-looking value
                # (see BUG_AUDIT M5/M10); s_{t+1} is unavailable here, so a
                # plain pseudo-reward regression is the correct warmup target.
                td_target = torch.tensor(pseudo_r,
                                         dtype=v_pred.dtype, device=device)
                critic_loss = torch.nn.functional.smooth_l1_loss(v_pred, td_target)
                critic_opt.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic_params, 1.0)
                critic_opt.step()

            # Update state
            h_t = h_next.detach()
            z_t = z_sample.detach()

            # ── sigma 校准（P0-1）：uncertainty_head 对齐 CE 不确定性 ──
            # 让 sigma 在部署前就携带真实不确定度信息（高 CE → 高 sigma），
            # 使"困惑→提问→学习→冷却"的涌现回路拥有可信的触发信号。
            if cal_opt and cal_params:
                try:
                    raw.train()
                    ce_ids = input_ids[:1, :128]
                    logits_cal = raw(ce_ids)
                    lbl = batch['labels'][:1, :ce_ids.size(1)]
                    ce_cal = F.cross_entropy(
                        logits_cal[:, :-1].reshape(-1, raw.config.vocab_size),
                        lbl[:, 1:].reshape(-1),
                        ignore_index=-100,
                    )
                    learnable = getattr(raw, '_learnable_sigmas', None)
                    if learnable is not None:
                        cal_target = torch.tanh(ce_cal.detach())
                        cal_loss = F.mse_loss(learnable, cal_target)
                        cal_opt.zero_grad()
                        cal_loss.backward()
                        torch.nn.utils.clip_grad_norm_(cal_params, 1.0)
                        cal_opt.step()
                except Exception as e:
                    print(f'  [warn] sigma 校准跳过: {e}')
                raw.zero_grad(set_to_none=True)

            if step % 100 == 0:
                pbar.set_postfix({
                    'imag': f'{loss_imagination.item():.4f}',
                    'kl': f'{loss_curiosity.item():.4f}',
                    'stab': f'{loss_stability.item():.4f}',
                })

        # 5. Set model state for deployment
        raw._h_state = h_t
        raw._z_state = z_t

        # 6. Re-freeze SelfModel + Critic (they'll be unfrozen by InternalLoop)
        for param in raw.parameters():
            param.requires_grad = True

        self.save_checkpoint(
            os.path.join(self.cfg.output_dir, 'warmup_final.pt'),
            loss.item(),
        )
        print(f"  Consciousness warmup complete. SelfModel/Critic initialized.\n")

    # ── Phase 3: Knowledge distillation ──

    def train_distill(self):
        # Only reset step if NOT resuming distill (main() loads step from ckpt)
        if not self._resume_phase('distill'):
            self.step = 0
        print(f"\n{'='*60}")
        print(f"Phase 3: Knowledge distillation")
        print(f"  Teacher: {self.cfg.teacher_name}")
        print(f"  Data: {self.cfg.distill_data}")
        print(f"  Steps: {self.cfg.distill_steps}")
        print(f"  Batch (micro): {self.cfg.distill_batch_size}")
        print(f"  Grad accum: {self.cfg.distill_grad_accum}")
        print(f"  Effective batch: {self.cfg.distill_batch_size * self.cfg.distill_grad_accum}")
        print(f"  LR: {self.cfg.distill_lr}")
        print(f"  T={self.cfg.distill_temperature}")
        print(f"{'='*60}\n")

        # Check if teacher logits exist; if not, generate them.
        # Generation writes to a .partial file and only renames on success,
        # so an interrupted run never leaves a corrupt file that passes
        # the os.path.exists check.
        # v3 = chat-template input + temperature-scaled storage (M6 fix:
        # teacher log-probs stored at the SAME temperature as student KD,
        # so CE+KL weights have their literal semantics).
        teacher_logits_path = (self.cfg.distill_data
                               .replace('.jsonl', '_teacher_logits_v3.jsonl.gz'))
        partial_path = teacher_logits_path + '.partial'
        if not os.path.exists(teacher_logits_path):
            print("Generating teacher logits (offline, one-time)...")
            from transformers import AutoModelForCausalLM
            teacher = AutoModelForCausalLM.from_pretrained(
                self.cfg.teacher_name, trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            # Only generate what training consumes (distill_steps x batch)
            # plus margin -- the full SFT corpus is 1.15M samples (~1.1TB),
            # which would overflow the disk / take days.
            n_consumed = self.cfg.distill_steps * self.cfg.distill_batch_size
            generate_teacher_logits(
                teacher, self.tokenizer,
                self.cfg.distill_data, partial_path,
                batch_size=4, max_seq_len=self.cfg.max_seq_len,
                device=self.device,
                max_samples=max(n_consumed, 200000),
                temperature=self.cfg.distill_temperature,
            )
            os.replace(partial_path, teacher_logits_path)
            del teacher
            gc.collect()
        else:
            print(f"Teacher logits found: {teacher_logits_path}")

        dataset = StreamingDataset(
            self.tokenizer,
            DataConfig(max_seq_len=self.cfg.max_seq_len, shuffle_buffer=1024),
            distill_paths=[teacher_logits_path],
        )
        if self.is_ddp:
            sampler = DistributedSampler(dataset, shuffle=True,
                                         num_replicas=self.world_size,
                                         rank=self.rank)
            loader = DataLoader(
                dataset, batch_size=self.cfg.distill_batch_size,
                collate_fn=collate_fn, num_workers=4, sampler=sampler,
            )
        else:
            sampler = None
            loader = DataLoader(
                dataset, batch_size=self.cfg.distill_batch_size,
                collate_fn=collate_fn, num_workers=4,
            )
        self._build_optimizer(self.cfg.distill_lr, self.cfg.distill_warmup,
                              self.cfg.distill_steps // self.cfg.distill_grad_accum)
        self._restore_optimizer_state('distill')
        data_iter = iter(loader)

        self.model.train()
        scaler = torch.amp.GradScaler(enabled=(self.dtype != torch.float32))
        pbar = tqdm(range(self.step, self.cfg.distill_steps), initial=self.step, desc='Distill')

        accum = self.cfg.distill_grad_accum
        _zero_grad(self.optimizer, scaler)

        for step in pbar:
            self.step = step
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            teacher_topk = (
                batch['teacher_topk_indices'].to(self.device),
                batch['teacher_topk_logp'].to(self.device),
            )

            with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype, enabled=(self.dtype != torch.float32)):
                # Use return_hidden + chunked loss to avoid OOM from vocab=151936
                hidden = self.model(input_ids, save_hebbian_buffers=False, return_hidden=True)
                loss, ce_item, kld_item = chunked_distill_loss(
                    self._raw_model, hidden, labels, teacher_topk,
                    scaler=scaler, accum=accum, chunk_size=256,
                    temperature=self.cfg.distill_temperature,
                    ce_weight=self.cfg.distill_ce_weight,
                    kd_weight=self.cfg.distill_kd_weight,
                )

            del hidden
            self._raw_model._last_layer_outputs = {}

            is_update_step = (step % accum == accum - 1) or (step == self.cfg.distill_steps - 1)
            if is_update_step:
                # Unscale FIRST so the monitor reads TRUE gradients
                if scaler:
                    scaler.unscale_(self.optimizer)
                # Monitor on true gradients
                sigma = getattr(self._raw_model, '_last_sigma_aggregate', 0.5)
                actions, grad_norms = self.monitor.step(loss * accum, sigma, self.step)
                # Now safe to clear expert sigmas
                self._raw_model._last_expert_sigmas = None
                self._raw_model._last_token_sigmas = None
                if actions:
                    old_lr = self.optimizer.param_groups[0]['lr']
                    new_lr = self.monitor.apply_actions(actions, self.optimizer)
                    if new_lr is not None and old_lr > 0:
                        self._lr_scale *= (new_lr / old_lr)
                        pbar.set_postfix({'LR_SCALED': f'{new_lr:.2e}'})

                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                _step(self.optimizer, scaler)
                _zero_grad(self.optimizer, scaler)
                self.scheduler.step()
                # M3 fix: scheduler 重算会覆盖 monitor 缩放，此处重新应用持久化乘数
                if self._lr_scale != 1.0:
                    for _g in self.optimizer.param_groups:
                        _g['lr'] *= self._lr_scale

            # Log every log_every steps (OUTSIDE is_update_step)
            if step % self.cfg.log_every == 0:
                lr = self.scheduler.get_last_lr()[0]
                postfix = {
                    'loss': f'{(loss * accum).item():.4f}',
                    'ce': f'{ce_item:.4f}',
                    'kl': f'{kld_item:.4f}',
                    'lr': f'{lr:.2e}',
                }
                ar_stats = self._get_attnres_stats()
                for k, v in ar_stats.items():
                    postfix[k] = f'{v:.2f}'
                pbar.set_postfix(postfix)

            if step > 0 and step % self.cfg.save_every == 0:
                self.save_checkpoint(
                    os.path.join(self.cfg.output_dir, f'distill_step{step}.pt'),
                    (loss * accum).item(),
                    )

        # Final save (loop's last step already did the optimizer step)
        self.save_checkpoint(
            os.path.join(self.cfg.output_dir, 'distill_final.pt'),
            (loss * accum).item() if 'loss' in locals() else 0.0,
        )
        print("Distill complete.\n")

    # ── Full pipeline ──

    def train_full(self):
        self.train_pretrain()
        self.step = 0
        self.train_sft()
        self.step = 0
        self.train_consciousness_warmup(steps=1000)
        self.step = 0
        self.train_distill()
        print(f"\n{'='*60}")
        print(f"Training complete! Final model: {self.cfg.output_dir}/distill_final.pt")
        print(f"{'='*60}")


# ── Main ───────────────────────────────────────────────────────────────

def main():
    # ── 显存碎片优化 + 强制 Flash Attention ──
    import os
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    import torch
    torch.backends.cuda.enable_flash_sdp(True)
    # 不强制关闭 math/mem_efficient，让 PyTorch 自动选择（Flash 不行就降级）

    # 检查 Flash Attention 可用性
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}, {props.total_memory/1e9:.1f}GB")
        print(f"  Flash SDP: {torch.backends.cuda.flash_sdp_enabled()}")

    parser = argparse.ArgumentParser(description='NeuroStream-Reflex Training')
    parser.add_argument('--mode', default='full', choices=['pretrain', 'sft', 'distill', 'full'])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', default='bfloat16', choices=['bfloat16', 'float16', 'float32'])
    parser.add_argument('--resume', default=None, help='Resume from checkpoint')
    parser.add_argument('--output-dir', default='/checkpoints/reflex-mini')
    parser.add_argument('--pretrain-data', default='/data/pretrain_all.jsonl')
    parser.add_argument('--sft-data', default='/data/sft_all.jsonl')
    parser.add_argument('--distill-data', default='/data/sft_all.jsonl')
    parser.add_argument('--teacher', default='Qwen/Qwen2.5-1.5B-Instruct')
    parser.add_argument('--pretrain-steps', type=int, default=228883)
    parser.add_argument('--sft-steps', type=int, default=38000)
    parser.add_argument('--distill-steps', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=None, help='Override all micro batch sizes')
    parser.add_argument('--grad-accum', type=int, default=None, help='Override all gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=None, help='Override all learning rates')
    parser.add_argument('--distill-temperature', type=float, default=None,
                        help='KD temperature (low T = precise alignment to teacher)')
    parser.add_argument('--distill-ce-weight', type=float, default=None)
    parser.add_argument('--distill-kd-weight', type=float, default=None)
    parser.add_argument('--fresh-optimizer', action='store_true',
                        help='Resume weights but reset optimizer/scheduler (for refinement)')
    args = parser.parse_args()

    # ── DDP initialization ────────────────────────────────────────────
    is_ddp = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    if is_ddp:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    rank = dist.get_rank() if is_ddp else 0

    torch.manual_seed(42 + rank)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(42 + rank)

    # Build model
    if rank == 0:
        print("Building ReflexMini model...")
    config = ReflexMiniConfig()
    model = ReflexModel(config)
    total = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"  Parameters: {total:,} ({total/1e9:.2f}B)")

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.teacher, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model._decode_tokenizer = tokenizer

    from train.data_pipeline import set_pad_id
    set_pad_id(tokenizer)

    # Training config
    cfg = TrainConfig(
        mode=args.mode, device=str(device), dtype=args.dtype,
        pretrain_data=args.pretrain_data, sft_data=args.sft_data,
        distill_data=args.distill_data, teacher_name=args.teacher,
        output_dir=args.output_dir, resume=args.resume,
        pretrain_steps=args.pretrain_steps, sft_steps=args.sft_steps,
        distill_steps=args.distill_steps,
    )
    if args.batch_size:
        cfg.pretrain_batch_size = args.batch_size
        cfg.sft_batch_size = args.batch_size
        cfg.distill_batch_size = args.batch_size
    if args.grad_accum:
        cfg.pretrain_grad_accum = args.grad_accum
        cfg.sft_grad_accum = args.grad_accum
        cfg.distill_grad_accum = args.grad_accum
    if args.lr:
        cfg.pretrain_lr = args.lr
        cfg.sft_lr = args.lr
        cfg.distill_lr = args.lr
    if args.distill_temperature is not None:
        cfg.distill_temperature = args.distill_temperature
    if args.distill_ce_weight is not None:
        cfg.distill_ce_weight = args.distill_ce_weight
    if args.distill_kd_weight is not None:
        cfg.distill_kd_weight = args.distill_kd_weight
    if args.fresh_optimizer:
        cfg.fresh_optimizer = True

    model.to(device)
    trainer = ReflexTrainer(model, tokenizer, cfg)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Run
    if args.mode == 'pretrain':
        trainer.train_pretrain()
    elif args.mode == 'sft':
        trainer.train_sft()
    elif args.mode == 'distill':
        trainer.train_distill()
    elif args.mode == 'full':
        trainer.train_full()

    if rank == 0:
        print("Done.")
    if is_ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
