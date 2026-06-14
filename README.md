# cubby-lm

Clean rebuild of Cubby after the v4 decode collapse.

v4 (`cubby_516m_v4`, 516M: MinGRU + sparse windowed attention + MoE + MTP + a
fixed-codebook **VSA output head**) reached val PPL ~7 but produced degenerate
generations under **both** greedy and sampling. The 5.6M MinGRU baseline with a
**weight-tied linear head** and everything else off free-runs coherent English
(only benign repetition loops a repetition penalty handles). So the recurrence
is fine — the collapse came from the components layered on at once, with the VSA
binding head as the prime suspect (low-rank softmax bottleneck feeding a 65k-way
cosine readout, a *fixed* temperature, and untied embeddings → low teacher-forced
CE but a miscalibrated, attractor-prone free-running distribution).

This repo rebuilds Cubby the disciplined way: start from the known-good baseline
and add **one component at a time** at matched budget, scoring **generation
quality** (not just perplexity) at every rung.

## The bigger picture (north-star)

Cubby is **one cortex** of a neuro-symbolic agent — the language faculty. It
interprets requests/responses, generates words, and emits **programs**; the
agent's **decisions are grounded in a VM** that executes auditable CubeLang
programs, not in the LM's activations. Full design in
[`ARCHITECTURE.md`](ARCHITECTURE.md). The map:

| Biological role | Component | Where |
|---|---|---|
| Language cortex | Cubby LM (MinGRU + tied-linear head) | this repo — the substrate |
| Thalamus | MindForge router (gates which cortex fires) | Stage 2 |
| Cortices / specialists | MindForge LoRA heads (opcode/intent/schema/rule) | Stage 2 |
| Hippocampus | neural episodic memory (zero-init gate, every 4th layer) | rung 2 |
| VM (grounding / QC) | CubeLang VM / opcode-vsa-rs (parse→compile→execute→verify) | exists |
| Dopamine | reward on loss improvement / verify success | partial |
| Neuromorphic | SNN / HybridFFN | late research rung |

Everything above the substrate rides on a trunk that already generates, so
**rung 0 is the one hard prerequisite.** VSA lives only in the small opcode/VM
space, never the word vocab; MTP is decode-time only, not pretraining.

## Design rules
- **Weight-tied linear output head by default.** The VSA head is a research rung,
  off by default, and only in a *corrected* form (learned temperature, optionally
  learned codebook).
- **Recurrence-first MinGRU backbone**, ported verbatim from the version that
  generates (`sigmoid(g)·tanh(v)` scanned with decay `0.001+0.998·sigmoid(d)`).
- **Standard SwiGLU FFN.**
- **Every advanced component is flag-gated and OFF.** Unbuilt rungs raise instead
  of silently no-op'ing, so a rung is never accidentally claimed as done.
- **No MTP during pretraining.** Cubby's own A/B showed MTP-as-aux hurt at 516M,
  consistent with the literature (the benefit emerges only at larger scale;
  Gloeckle et al. arXiv:2404.19737). MTP is decode-time only: a frozen
  self-speculative head attached post-training.
- **MoE balancing (when added) is auxiliary-loss-free bias-update**
  (DeepSeek-V3 / arXiv:2408.15664), not a hand-tuned PID controller.
- **Measure generation:** rep-n, distinct-n, looping-rate alongside PPL.

## Ablation ladder
0. MinGRU + tied-linear head + SwiGLU  ← **this repo, now**. Confirm coherent generation + PPL.
1. Scale to target dims (d=1024, L=18), still tied-linear, dense FFN.
2. + episodic memory injection (every 4th layer, zero-init gate) — validated-positive in prior versions; A/B to confirm (watch looping-rate / coherence, not just PPL).
3. + sparse windowed causal attention (every 3rd layer).
4. + MoE (4 experts, top-2, shared) with auxiliary-loss-free bias balancing.
5. + sparse-FFN / FFN-parallelism.
6. MTP excluded from pretraining; optional frozen speculative-decode attach.
7. + LoopMoE sandwich + IterAdaLN-style depth-conditioned modulation (research).
8. + corrected VSA head variants, LAST, each vs tied-linear at equal budget.

Rule: any rung that drops generation quality while PPL stays flat is quarantined
and understood before proceeding.

## Layout
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — full neuro-symbolic agent design: brain map, neuro-symbolic loop, LM↔VM bridge, build-order DAG, component status.
- `config.py`  — `CubbyConfig` (encodes the ladder; advanced flags off).
- `model.py`   — RMSNorm, MinGRU scan, MinGRU layer, SwiGLU, Block, `CubbyLM` (+ corrected VSA head).
- `generate.py`— autoregressive sampling with repetition penalty / no-repeat-ngram.
- `metrics.py` — rep-n, distinct-n, looping detection.
- `smoke.py`   — builds a tiny model; checks forward / backward / generation end to end.

## Run the smoke test
```
uv run python smoke.py
```
