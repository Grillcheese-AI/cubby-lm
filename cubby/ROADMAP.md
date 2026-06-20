# Sparse Cubby — grilly port roadmap

Clean rebuild of Cubby, **ported 100% to grilly** (Vulkan compute, no torch in
the trunk path), based on the working **v3.3** run and the architecture spec in
`docs/why_sparse_cubby.md`. Built as a version ladder: one validated component
per version, at matched budget, scoring **generation** (rep-n / distinct-n /
looping-rate / coherence) and not PPL alone. An unbuilt component raises.

Rule: any version that drops generation quality while PPL stays flat is
quarantined and understood before the next.

**Standing invariant (every rung):** every step is auditable + visualizable.
Each new component must (1) emit per-step records to `cubby.trace`, (2) render
in the viewer (real-time, per example — neurons light up), and (3) be asserted
on in tests (golden trace + first-divergent-step diff). Not done until all three
hold — see `docs/ARCHITECTURE.md` "Observability & audit".

## Versions (semantic versioning)

The `0.0.x` ladder builds the **trunk and its attachments** — one validated
component per rung. `0.1.0` is the first **system-integration** milestone: the
afferent input layer that wraps the trunk once specialists and the CNS exist.

| ver | Adds | Pillar | Gate | grilly ops |
|---|---|---|---|---|
| **0.0.0** | MinGRU + tied-linear + SwiGLU trunk (small, ~5.6M) | substrate | **forward parity vs torch ref** + coherent TinyStories gen | embedding, rmsnorm, prefix_scan/mingru, linear, activations(silu), loss, optimizer |
| 0.0.1 | scale to v3.3 shape (d=1024, L=18) + BBPE-65k corpus | substrate | val PPL descends; coherent prose | same, at scale |
| 0.0.2 | chunked sliding-window attention (W=512, every 3rd) | unlimited context | linear mem vs S; window-leakage parity | attention (chunked) |
| 0.0.3 | sparse MoE-MinGRU (4 exp/top-2 + 3 shared, DeepSeek bias) | sparsity | load balance; per-token ~top-K | moe_forward |
| 0.0.4 | Hebbian growth (Oja + lateral inhibition → spawn expert) | one novelty primitive | expert spawns on novelty; no destab | (custom + moe) |
| 0.0.5 | SegmentMemory (Hebbian-keyed, CPU store) | real memory | 128k effective; bounded per-chunk mem | (host store + linear) |
| 0.0.6 | VSA binding head (frozen MAP-bipolar codebook D=10240) | VSA substrate | trains; coherent; PPL ~6 | linear(d→D) + cosine |
| 0.0.7 | MTP attachment — **DECODE-TIME ONLY**, frozen trunk (never in pretrain; hurts an untrained trunk) | no-retrain | speculative-decode lift, trunk grads 0 | shared head |
| 0.0.8 | CubeLang head + VM + WorldManager arena | reasoning off-axis | parse→compile→execute→verify rate | vsa block-code ops |
| 0.0.9 | MindForge adapter bank + cross-trunk fusion | no-retrain growth | adapter lifts task, no regression | linear + lora |
| **0.1.0** | **Afferent SNN gate (the input layer)**: a spiking perceptron triages every input → **CNS reflex path** (live_brain: amygdala + neurochemistry) on high stress/threat/urgency, else **cortical router** classifying {modality, tone, intent, domain} → dispatch to the matching WorldManager specialist, **spawning one** when no specialist owns that regime | system integration / afferent | threat path pre-empts the router; router hits the right specialist; spawns on a novel regime | snn perceptron + router + worldmanager (+ Hebbian spawn) |

`live_brain` (SNN, event-driven, grilly-backed via `brain/`) plays two roles
here: the **CNS reflex path** the afferent gate (0.1.0) escalates to under
threat/stress, and a **modality cortex** (vision) hooked in after the substrate
is solid. It's a parallel track, not part of the 0.0.x trunk.

## Why grilly (not torch)

The trunk's hot ops are exactly what was hardened on grilly `2.0-dev`: the
write-combined-readback fix took `rmsnorm` 2412→34.6 ms (70x) and `linear`/
`mingru`/`layernorm`/`embedding`/`activations` are all on the fast staging path.
Cubby's thesis — capable AI on accessible hardware — requires the trunk to run
and train on the Vulkan backend, not CUDA/DirectML.

## Port discipline

1. Each version implements the trunk forward on grilly ops and is validated for
   **numerical parity** against the torch reference in `_reference_torch/`
   (max_abs_diff), the same way the tokenizer is checked against HF.
2. Then training (forward + backward + AdamW via grilly `optimizer`) is brought
   up and gated on coherent generation.
3. Only then does the next component go on.

## Status

- [x] 0.0.0 config + structure scaffolded
- [x] observability: `cubby.trace` per-step audit/visual bus + tests (8/8)
- [x] FFN variants: SwiGLU + TernarySwiGLU (BitNet QAT, STE) on tape autograd (6/6)
- [x] 0.0.0 trunk forward+backward on grilly; every param grads; overfits a batch (3.49->0.035)
- [x] 0.0.0 forward parity vs numpy reference (max_abs_diff 2.3e-7; gate confirmed 0.001+0.998)
- [x] 0.0.0 TinyStories train: learns (byte 3.49->1.23) + generates English
- [x] real tokenizer wired: BBPE-65k v3 loads + trains; spelling garble gone vs byte
- [x] ternary vs SwiGLU A/B: quality parity (1.225 vs 1.231) -- byte savings ~free (small scale)
- [ ] ternary Vulkan kernel (multiply-free): cpp/src/ops + shader + binding (the actual byte/speed win)
- [ ] AST/cubelang special tokens registered on BBPE-65k (NL ids unchanged) -- SUPERSEDED by multilingual BPE below
- [x] 0.0.1 GPU linear bridge: tape `linear` over `_bridge.linear`/`linear_backward` (all tests green, parity 2.8e-7)
- [x] 0.0.1 trains on GPU beyond CPU's reach (d=512 L8); FINDING: transfer-bound, ~0.6 it/s
- [x] 0.0.1 perf diagnosis: cost is per-dispatch overhead (~25ms/call), not compute/transfer (silu-fusion + fp16 both null)
- [x] 0.0.1 perf #1 fused projections: g/v/d 3->1, gate/up 2->1 dispatch; linear_bwd -27%, fwd -14% (parity 2.6e-7)
- [x] 0.0.1 perf #3 GPU CE (softmax/loss on GPU; _bridge.ce_backward rejected as wrong) + in-place AdamW (293ms vs 489; GPU adamw rejected)
- [ ] 0.0.1 perf #2 resident activations (VRAM across forward; the ~25ms/dispatch floor) -- big refactor, needs arch decision
- [ ] 0.0.1 full v3.3 shape (d=1024 L18 V65k) run once throughput is acceptable
- [x] 0.0.2 chunked sliding-window attention: forward parity 1e-7, backward precision 5e-6, all gradients flow (18/18 params)

## Architecture milestone: multilingual BPE + sampled softmax + dual-head (June 2026)

Replaces BBPE-65k + full softmax CE with a three-part architecture change:

1. **Custom multilingual BPE** (`cubby_mbpe32k`, 32768 vocab). Trained on 128 GB
   of the unified corpus (C4 multilingual, wiki EN/FR, math, agent data, personas)
   using the Rust `tokenizers` crate. Byte-level with Split pre-tokenizer. 67
   special tokens registered as atomic `AddedToken` entries (chat markers, AST
   structure tags, CubeLang VM opcodes). 47 of those are classified as AST tokens;
   the remaining are language-adjacent (chat/structural). Roundtrip validated
   across 7 scripts (EN/FR/ZH/RU/AR/HI/JA).

2. **Dual-head output architecture**. Single shared trunk; learned router
   (`Linear(d, 2) -> softmax`) per token position gates two independent heads:
   - Language head: `RMSNorm -> Linear(h, embed_lang)` -> logits (V_lang)
   - AST head: `RMSNorm -> Linear(h, embed_ast)` -> logits (V_ast)
   Tokens classified as lang vs AST by membership in `tok.ast_token_ids`.
   Loss = router-weighted sum of per-head CE. Gated OFF by default
   (`enable_dual_head=False`); enabled in `tiny_mbpe` preset.

3. **Sampled softmax / importance-sampling CE** (`sampled_cross_entropy`).
   Avoids materialising the full (N, V) logit tensor during training: for each
   token, draw K=1024 uniform negatives, compute logits only for the K+1 subset
   (target + negatives), CE over the subset. With uniform sampling the IS
   correction is a constant -> gradient direction unbiased (Bengio & Senecal 2008).
   At inference, full softmax is used (cheap at 30k vocab). Custom GradFn with
   correct sparse gradient scatter-add into the embedding table.

Files changed: `cubby/tools/train_tokenizer.py` (new), `cubby/tokenizer.py`,
`cubby/config.py`, `cubby/trunk/model.py`, `cubby/trunk/resident.py`, `main.py`.

**✅ Gate passed (June 2026):** `tiny_mbpe` preset reaches **PPL ~4 in <1000
steps** — near the v4 516M target (val PPL ~7) at a fraction of the budget and
step count — generating coherent English prose ("Once upon a time, there was a
girl named Lily. She loved to look around..."), stable gradients (gnorm
0.6-0.74), 1.59 it/s. mbpe32k + dual-head + sampled-softmax is the **production
substrate**. Checkpoint saved at `ckpt_tiny_mbpe.grl`. Dual-head + sampled IS
ported to the resident GPU path. **Next: the `mbpe_v33` consolidation run** —
the same stack at v3.3 shape (d=1024, L=18) + 0.0.2 chunked SWA — gates on
coherent prose at production budget before 0.0.3 (MoE).

## Resident GPU path: dual-head + sampled softmax (June 2026)

Ports the dual-head architecture and importance-sampling CE to the GPU-resident
path (`cubby/trunk/resident.py`). The resident path owns the grilly Vulkan
context and runs forward+backward+AdamW entirely on-device.

**Architecture changes:**
- Combined embedding table `[E_lang; E_ast]` registered as a single resident
  weight. Router `(d → 2)` registered as a separate persistent weight.
- `_resident_forward` computes `router_logits = Linear(h, router)` alongside
  main logits. `_fb_run` records the router op in the tape for gradient tracking.
- `train_step` accepts `use_sampled` and `n_samples` params. Computes dual-head
  CE weighted by router probabilities.

**Loss helpers:**
- `_sampled_ce(logits, tgt, V, n_samples=1024)`: Bengio & Senecal 2008
  importance-sampling CE with K negatives per position. Gradient direction
  unbiased when sampling distribution is uniform.
- `_dual_head_ce(logits, tgt, router_logits)`: router-weighted loss for
  language + AST heads. Tokens classified by `tgt < Vlang`.

**Verification:** forward parity 1.7e-04 vs model.py, gradient parity 1.9e-02
(all params < 2e-2), existing resident.py parity tests pass at 1e-06 precision.
Router weights flow correctly through the tape backward.

Status: complete, ready for full v3.3-shape (d=1024 L=18 V=32k) training.

## 0.0.2 Chunked sliding-window attention (June 2026)

Implements chunked sliding-window causal attention with O(S·W) memory complexity instead of O(S²). Inserted every 3rd layer when `enable_attention=True`.

**Implementation:**
- `chunked_sliding_window_attention()`: Core attention function as custom GradFn. Processes Q in W-sized chunks, gathers K/V from overlapping windows, applies causal+window mask per chunk.
- `chunked_sliding_window_attention_from_split()`: Variant accepting combined (3,B,H,S,Dh) input from QKVSplit GradFn.
- `LocalCausalAttention`: Module wrapping QKV projection → attention → output projection.
- `QKVSplit` GradFn: Bridges the fused QKV projection and attention, ensuring gradients flow through reshape and transpose operations.
- `_reference_sliding_window_attention()`: Brute-force baseline for parity testing.

**Key design decisions:**
- Window size W=512 tokens (configurable via `attn_window`)
- Attention every 3rd layer (configurable via `attn_every_n`)
- Backward precision verified to 5e-06 relative error vs finite differences
- Forward parity vs reference: 1e-07 max abs diff

**Integration:**
- Block conditionally includes attention when `idx % attn_every_n == 0`
- Residual scaling applied to attention output projection when enabled
- All parameters (QKV, output projection, rms_attn) receive gradients

## 0.1.0 design note: MMoE/PLE cortical-router perception head (June 2026)

A concrete design for the **0.1.0 cortical router** (the "high road" of the
afferent gate): a Multi-gate Mixture-of-Experts / Progressive Layered Extraction
head that reads the trunk's hidden state and emits a **structured multi-task
perception vector** along the router's classification axes. This is the gate's
feature extractor, not a trunk output head.

**Architecture.** Shared trunk features → a pool of N unlabeled expert MLPs
(self-specializing) → one tiny **gating network per task** (softmax over experts)
→ a per-task **tower** projecting the gate-blended expert mix to that task's dim.
Initial task partitions: semantic, emotion, intent, POS, opcode-recognition. The
per-task gates are what mitigate **negative transfer** (binary-opcode gradients
vs emotional-nuance gradients fighting in a single dense head).

**Decisions (load-bearing — these are why it's here and not earlier):**
1. **Post-training adapter over a FROZEN trunk — never co-trained during
   pretraining.** Same v4 lesson as the VSA head: a novel multi-task head
   co-trained with the substrate risks tanking generation while PPL looks fine,
   and the pooled-representation objective fights next-token CE. It attaches like
   MTP (0.0.7) and the MindForge adapter bank (0.0.9): frozen trunk, no retrain.
2. **Split per-token from per-sequence tasks.** Emotion / intent / domain are
   per-sequence (mean-pool the hidden states). **POS and opcode are per-token** —
   pooling destroys their signal; they run as a per-token head over the unpooled
   states. One pooled head + one per-token head, not a single pooled head.
3. **Opcode tower is recognition only, not reasoning.** It tags/recognizes
   opcodes for routing; opcode *reasoning* stays in the CubeLang VM over grounded
   reps (the off-token-axis thesis). Re-importing opcode reasoning into LM
   activations was part of the v4 failure.
4. **Distinct from 0.0.3 trunk MoE.** 0.0.3 is MoE *in the trunk* (token-level
   capacity/sparsity, MoE-MinGRU). This is MoE *over frozen features* for
   multi-task perception. Different mechanism, different purpose.
5. **Sparse Top-2 gating** is the efficiency path once dense gating is validated;
   the explicit gate probabilities are **printable per state transition** →
   satisfies the observability invariant (the VM/viewer can show how much each
   task leaned on each expert).

**Gate:** each task lifts its own metric (accuracy / F1) over a dense-head
baseline at matched budget, **with no regression in trunk generation** (frozen
trunk guarantees the latter by construction). Builds at 0.1.0 alongside the SNN
threat path and the WorldManager specialists it routes to.

**Status:** ✅ Complete. Forward+backward validated. Ready for resident path integration.
