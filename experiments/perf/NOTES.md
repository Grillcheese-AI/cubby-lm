# 0.0.1 GPU scale-up: perf findings

## Keystone (done)

`cubby/trunk/gpu_linear.py`: tape `linear(x, W)` wrapping grilly_core's GPU
`linear` / `linear_backward` (via `grilly.backend._bridge`), W = (out, in).
Drop-in for the MinGRU projections, FFN, and the tied head. Correct (parity
2.6e-7; all ffn/model/parity tests pass), ternary STE composes through it, numpy
fallback keeps 0.0.0 CPU-only.

`_bridge` exposes the full GPU op set: `linear` / `linear_backward`,
`cross_entropy_loss` / `cross_entropy_backward`, `softmax` / `softmax_backward`,
`embedding_lookup`, `adamw_update`, `fused_layernorm_linear`.

## Diagnosis: the cost is PER-DISPATCH OVERHEAD, not compute or transfer

Profiled one step at d=512 L8 (`profile_step.py`, 5 steps). Original ~3.5 s/step:

| cost | tottime/5steps | share |
|---|---|---|
| `linear_backward` (GPU) | 5.65 s | 32% |
| `linear` fwd (GPU)      | 2.39 s | 14% |
| numpy AdamW optimizer   | 2.57 s | 15% |
| tape engine + Variable mul/tanh/add (silu, residuals) | ~2.0 s | 20% |
| mingru / rmsnorm / embedding | small | -- |

Each `_bridge.linear` is numpy-in / numpy-out: per call it submits, waits, stages,
downloads. Cost is ~25 ms/call FIXED regardless of size. ~110 dispatches/step.

### Two null results that pin the diagnosis
1. **Fused silu** (single numpy GradFn instead of 4 Variable ops): measured
   SLOWER (1.7 s vs ~1.3 s). The elementwise cost is numpy `exp`, not tape-node
   overhead. Reverted.
2. **fp16 transfer** on the linears (half the input bytes): linear_backward
   UNCHANGED (6.05 s). So it is not transfer-byte-bound either -- it is
   per-dispatch overhead, which dtype does not touch. Reverted (USE_FP16=False).

=> The only lever is FEWER DISPATCHES.

## Win #1 (the real one): fused projections -- DONE

Concatenate matmuls that share an input into one weight, one dispatch:
- MinGRUMixer: 3x `(d,d)` g/v/d  ->  one `(3d,d)`  (3 dispatches -> 1)
- SwiGLU: 2x `(d,d_ffn)` gate/up  ->  one `(2*d_ffn,d)`  (2 -> 1)
Autograd-aware `slice_cols` GradFn (forward view, backward scatter; pure CPU, no
dispatch) splits the fused output back into g/v/d / gate/up.

Measured at d=512 L8 (5 steps):
| | before | after |
|---|---|---|
| `linear` dispatches | 245 | **125** (49 -> 25 /step) |
| `linear_backward` | 5.82 s | **4.23 s** (-27%) |
| `linear` fwd | 2.40 s | **2.07 s** (-14%) |

Parity holds 2.6e-7; param count unchanged (50256); overfit 3.49->0.040. Ternary
composes through the fused gate_up (shares one per-tensor alpha -- fine for BitNet).

## Win #3: GPU CE + in-place AdamW -- DONE (with one rejection)

- **GPU cross-entropy**: forward via `_bridge.cross_entropy_loss` (per-row, matches
  numpy exactly), backward via `_bridge.softmax` (GPU, offloads the V=65k `exp`)
  then a cheap numpy onehot-subtract + `/N`. NOTE: `_bridge.cross_entropy_backward`
  was verified WRONG for our convention (grad diff 3.11 vs (sm-onehot)/N) -- do not
  use it; the GPU softmax + numpy finalize is correct (overfit 3.49->0.040).
- **AdamW**: benchmarked 3 ways on the real 31.6M-param set --
  numpy-alloc 489 ms, GPU `adamw_update` 429 ms (loses to per-param round-trips),
  **in-place numpy 293 ms** (winner, -40%). Applied the in-place version; GPU
  AdamW rejected.

End-to-end d=512 L8: byte 1.66 -> 1.54 s/step (GPU CE adds dispatches at V=256 that
mask the fusion/AdamW gains; at V=65k GPU CE earns its place). BBPE-65k runs and
learns (11.17 -> 10.07), ~0.2 it/s (head 512->65536 + CE dominate).

## Remaining lever: #2 resident activations (the step-change)

#1 and #3 are banked, but ~125 linear dispatches/step still each pay the ~25 ms
submit/wait/download floor, and the tape elementwise (silu/residuals/rmsnorm) is
CPU numpy. Both are only cured by keeping activations in VRAM across the forward:
one command-buffer region per step instead of ~125 round-trips, download only at
loss/sample. Approaches: Tensor-resident tape via `linear_t` (Tensor I/O), or a
hand-written resident fwd/bwd for the fixed trunk. Biggest win, biggest refactor --
needs an architectural decision before starting.
