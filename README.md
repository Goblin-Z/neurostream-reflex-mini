# NeuroStream-Reflex Mini

[简体中文](README.zh-CN.md) | English

**A 0.77B Chinese neuromorphic language model** — dual-loop architecture with external training and internal stream of consciousness. Memory and thought blend at every computation step, forming a continuously advancing self-referential loop.

## Highlights

- **Dual-loop architecture**: the outer loop (dialogue) and inner loop (stream of consciousness) share state and memory — essentially the same "memory → computation → new memory" self-referential cycle running at two scales
- **Memory system v4 (L1–L4)**:
  - L0: explicit history (template concatenation)
  - L1: short-term semantics (dialogue injected into `h_t`)
  - L2/L3: long-term semantics (AttnRes memory source + differentiable memory slots)
  - L4: content memory (layered KV cache — attention reaches historical token representations directly, enabling verbatim recall)
- **Spontaneous consolidation**: memory importance is quantified by model behavior (attention-weighted salience accumulation); memories are distilled into stable weights when mature — non-programmatic, no fixed step schedule
- **Hebbian learning**: expert weights are strengthened by local gradients (momentum + activation gating + dual clipping), not global backprop
- **Architecture self-modification**: expert split/prune/add-layer (safe after fixes, enabled by default at deployment)

## Quick Start

### Deploy (internal loop fully enabled)

```bash
python run_mini.py --checkpoint <ckpt.pt> --device cuda
# Interactive: ask questions / "quit" to exit / "stats" for internal state / "clear" to reset memory
```

### Test a checkpoint

```bash
python chat_sft.py --checkpoint <ckpt.pt> --device cpu --prompt "中国的首都是"
```

### Train (cloud / AutoDL)

```bash
# One-command pipeline (data gen → cleaning → SFT → KD → refine; idempotent, resumable)
nohup bash scripts/run_pipeline.sh > pipeline.log 2>&1 &
tail -f pipeline.log
```

### Memory fine-tuning (teach the model to *use* memory)

```bash
# 1. Generate long-range-reference multi-turn data (7 turns + cross-turn reference)
python scripts/generate_qa.py --teacher <teacher> \
  --output mt_memory_20k.jsonl --max-samples 20000 \
  --mode multi-turn --memory-tune --device cuda

# 2. Fine-tune (KV history genuinely participates in training)
python train_memory.py --checkpoint <ckpt.pt> \
  --data mt_memory_20k.jsonl --steps 3000 --lr 1e-5 --batch-size 4
```

## Architecture

```
User input → [Outer Loop Pipeline] ──► Generate (with h_state + mem_kv)
              │                              │
              ▼                              ▼
      Dialogue → h_t + KV store        Online training (memory + state)
              │                              │
              ▼                              ▼
       [Inner Loop InternalLoop] ◄───────────┘
        Stage A-K (noise→SelfModel→Hebbian→consolidate→verify)
              │  Memory (slots/KV/salience) blends with computation
              ▼
        New state + new memory → feeds next outer loop
```

| Component | Description |
|-----------|-------------|
| `core/` | Model backbone (MoE/Attention/AttnRes/SelfModel/MemoryBank) |
| `loop/` | Inner loop (Stage A-K stream of consciousness) |
| `learn/` | Online learning (Hebbian/consolidation/Critic) |
| `interaction/` | External interaction (dialogue/verification/feedback) |
| `train/` | Training (pretrain/SFT/distill) |
| `scripts/` | Data generation / pipeline |

## Model Spec

- **Architecture**: 24-layer GQA + RoPE + SwiGLU MoE (4 stable + 2 plastic) + block Delta attention residuals
- **Vocabulary**: Qwen2.5 (151936)
- **Parameters**: 769M
- **Training**: 15B tokens Chinese pretraining + teacher (Qwen2.5-1.5B-Instruct) QA/multi-turn distillation
- **Memory**: MemoryBank (128 semantic slots + 4-round KV cache + salience-based spontaneous consolidation)

## Deploy Command Reference

| Command | Function |
|---------|----------|
| `stats` | Internal loop state (steps/loss_int/sigma/can_ask/errors) |
| `clear` | Reset conversation memory (L0 history + KV) |
| `quit` | Exit |
| `--no-internal-loop` | Pure generation (comparison) |
| `--no-arch-self-mod` | Disable architecture self-modification (on by default) |
| `--verify-threshold 0.45` | Lower proactive-verification threshold |

## Dependencies

```bash
pip install torch transformers safetensors tqdm
export HF_ENDPOINT=https://hf-mirror.com   # mirror for China
```

## Known Limitations

- 0.77B capacity: limited complex reasoning and long-form knowledge
- Answering ability is bounded by training-data coverage (unseen instructions may be off-topic)
- Effective memory use depends on memory fine-tuning (`train_memory.py`) — structure ready + training reinforced
- Complex role-setting prompts work poorly (short-instruction interaction recommended)

## License

MIT License — neuromorphic research project, forks and improvements welcome.

---

*NeuroStream-Reflex Mini — a self-referential loop where memory and thought blend*
