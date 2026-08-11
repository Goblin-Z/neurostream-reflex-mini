# NeuroStream-Reflex Mini

[简体中文](README.zh-CN.md) | English

**A 0.77B Chinese neuromorphic language model** — dual-loop architecture with external training and internal stream of consciousness. Memory and thought blend at every computation step, forming a continuously advancing self-referential loop.

## Highlights

### Why dual-loop?

The model does not merely respond to input — it continuously *thinks* even when nobody is talking.

- **Outer loop (interaction)**: turns input + memory + state into dialogue. This is the model's "action" layer — fast, responsive, grounded in the current conversation.
- **Inner loop (consciousness)**: runs continuously in the background — evolving internal state, strengthening experts (Hebbian), consolidating memory, and evaluating its own uncertainty. This is the "reflection" layer — slow, self-referential, independent of external stimuli.

Together they form a **self-referential cycle**: every computation changes the state; the new state determines the next computation. What the model remembers shapes what it thinks; what it thinks becomes new memory. The cycle is closed at two scales — a micro-cycle inside (state → think → new state) and a macro-cycle across interactions (converse → learn → remember → converse differently).

### Proactive verification (knowing what it doesn't know)

The model can **ask questions when it is confused**:

- Its experts carry an internal uncertainty signal (sigma). When sigma exceeds a threshold, the model recognizes it is not sure — not by rule, but by its own internal state.
- It then generates a clarification question from its confused state and asks the user.
- Your answer becomes a learning signal (focal boost on the confused expert) — the model updates, sigma drops, and it stops asking (emergent cooldown: learning itself is the cooldown, no timers).
- If a question remains unresolved (sigma stays high), it will ask again — genuine curiosity loop, not a script.

This is a primitive form of **metacognition**: the model monitors its own uncertainty and acts on it, closing the "confusion → ask → learn → resolve" loop.

### Multi-level memory

- **Verbatim recent memory**: attention reaches historical token representations directly — the model can recall what was actually said, not just a summary
- **Conversation gist**: internal state encodes what the conversation is about
- **Long-term knowledge**: a learnable memory bank keeps semantic memories
- **Spontaneous consolidation**: the more the model *uses* a memory (measured by its own attention), the more important it becomes — mature memories are distilled into weights automatically, no fixed schedule

### Other highlights

- **Hebbian learning**: experts strengthen through local correlation ("neurons that fire together wire together"), not global backprop
- **Architecture self-modification**: the model can split/prune/add experts as it learns

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
                        ┌──────────────────────────────────────────┐
                        │              MEMORY SYSTEM                │
                        │  verbatim (KV cache)  gist (h_t)  knowledge│
                        │  (learnable slots)                        │
                        └────▲────────────▲─────────────▲───────────┘
                             │            │             │
   OUTER LOOP (interaction)  │   writes   │   reads     │
   ┌────────────┐  generate ┌┴───────┐    │             │
   │ user input │───►───────│ THINK  │◄───┴─────────────┘
   └────────────┘           └───┬────┘     (h_state + memory)
                                │ new state / memory
   INNER LOOP (consciousness)   ▼
   ┌──────────────────────────────────────────────────────┐
   │  every step (continuous, no external input needed):  │
   │  A. take state → noise → imagine (SelfModel) → think │
   │  B. Hebbian update (strengthen used experts)         │
   │  C. world-model learning + critic value              │
   │  D. write state → memory (KV/slots/dialectics)       │
   │  E/F. consolidate (distill memory into weights)      │
   │  H. proactive verification (high sigma → ask user)   │
   │  memory participates in every computation step;      │
   │  every step produces new memory for the next one     │
   └──────────────────────────────────────────────────────┘
```

**The loop is closed**: inner loop continuously evolves state/memory; outer loop reads that state to speak; speaking writes new memory back into the loop. One self-referential cycle at two scales.

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
