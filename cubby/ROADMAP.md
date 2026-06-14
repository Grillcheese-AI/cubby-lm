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
- [ ] AST/cubelang special tokens registered on BBPE-65k (NL ids unchanged)
- [x] 0.0.1 GPU linear bridge: tape `linear` over `_bridge.linear`/`linear_backward` (all tests green, parity 2.8e-7)
- [x] 0.0.1 trains on GPU beyond CPU's reach (d=512 L8); FINDING: transfer-bound, ~0.6 it/s
- [x] 0.0.1 perf diagnosis: cost is per-dispatch overhead (~25ms/call), not compute/transfer (silu-fusion + fp16 both null)
- [x] 0.0.1 perf #1 fused projections: g/v/d 3->1, gate/up 2->1 dispatch; linear_bwd -27%, fwd -14% (parity 2.6e-7)
- [x] 0.0.1 perf #3 GPU CE (softmax/loss on GPU; _bridge.ce_backward rejected as wrong) + in-place AdamW (293ms vs 489; GPU adamw rejected)
- [ ] 0.0.1 perf #2 resident activations (VRAM across forward; the ~25ms/dispatch floor) -- big refactor, needs arch decision
- [ ] 0.0.1 full v3.3 shape (d=1024 L18 V65k) run once throughput is acceptable
