# NeuroStream-Reflex Mini

[简体中文](README.zh-CN.md) | English

**A 0.77B Chinese neuromorphic language model** — dual-loop architecture with external training and internal stream of consciousness. Memory and thought blend at every computation step, forming a continuously advancing self-referential loop.

## Highlights

- **Dual-loop architecture**: an inner loop keeps "thinking" continuously (state evolution, Hebbian learning, consolidation), while the outer loop turns thoughts into dialogue — they share the same memory and state, so what the model "remembers" shapes what it says, and what it says becomes new memory
- **Multi-level memory**: remembers recent conversation verbatim (via attention over historical representations), understands what the conversation is about (via internal state), and keeps long-term knowledge (via a learnable memory bank) — like working memory + episodic memory + semantic memory
- **Spontaneous consolidation**: the more the model actually *uses* a memory (measured by its own attention), the more important it becomes — and when a memory matures, the model automatically distills it into its weights, without any fixed schedule
- **Hebbian learning**: experts strengthen through local correlation ("neurons that fire together wire together"), not global backprop
- **Architecture self-modification**: the model can split/prune/add experts over time as it learns

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

> Training pipeline (pretrain/SFT/distill/memory fine-tuning) is kept private.
> Contact the author if you need training code.

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
| `core/` | Model backbone (MoE/Attention/SelfModel/MemoryBank) |
| `loop/` | Inner loop (stream of consciousness: Hebbian/consolidation) |
| `learn/` | Online learning (Hebbian updates/Critic) |
| `interaction/` | External interaction (dialogue/verification/feedback) |
| `config/` | Model configuration |

## Model Spec

- **Architecture**: 24-layer GQA + RoPE + SwiGLU MoE (4 stable + 2 plastic experts) + block Delta attention residuals
- **Vocabulary**: Qwen2.5 (151936)
- **Parameters**: 769M
- **Training**: 15B tokens Chinese pretraining + teacher (Qwen2.5-1.5B-Instruct) QA/multi-turn distillation
- **Memory**: 128 learnable memory slots + 4-round KV conversation cache + attention-based spontaneous consolidation

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
