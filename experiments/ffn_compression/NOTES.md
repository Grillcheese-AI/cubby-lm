# FFN compression track

Goal (Nick): project wide FFNs (d_ffn >= 4096) into an order-of-magnitude
smaller footprint **for both pretraining and inference**, on a memory-bound
single AMD GPU (RX 6750 XT, RADV/Vulkan, no CUDA / no sparse Tensor Cores).

## The three axes (don't conflate them)

| axis | what it saves | matters because |
|---|---|---|
| **bytes** (weight bits x reads) | VRAM bandwidth | **binding constraint** on this HW (the readback saga) |
| FLOPs | compute time | only if compute-bound (we're not, mostly) |
| params | disk / checkpoint, some generalization | smaller artifacts |

~10x on **bytes** is nearly free (ternary at scale). ~10x on **params/FLOPs**
structurally is *not* free — there's a capacity/quality frontier paid down with
training, depth, or distillation. Anyone promising a free structural 10x is
hiding the loss.

## Methods, ranked for THIS hardware

1. **Ternary / BitNet b1.58 (CHOSEN first).** Weights -> {-1,0,+1} with a scale,
   QAT + STE from day 1. Matmul becomes signed accumulation = multiply-free.
   Cuts *bytes* ~10x, HW-agnostic, trains from scratch. Best fit. Plan: prove
   quality parity vs fp32 SwiGLU first (QAT simulation, normal matmul), THEN
   build the multiply-free Vulkan kernel for the actual byte/speed win.
2. **Monarch / BTT.** W = L.P.R (block-diagonal x perm x block-diagonal).
   Real params+FLOPs cut, HW-agnostic, block-diagonal bmm is grilly-friendly.
   Strong #2; costs a custom kernel. "90% params, no PPL loss" is optimistic —
   believe the speedup, discount the free lunch.
3. **Fine-grained sparse MoE.** Already on the ladder at 0.0.3; doc's twist is
   64-128 micro-experts. Decouples capacity from active-FLOPs. Caveat: on a
   single memory-bound GPU the gather/route overhead can eat the win; the
   "order-of-magnitude" is capacity-on-disk vs active-FLOPs, not wall-clock.
4. **Low-rank + muP.** muP is the real insight (naive low-rank *does* fail), but
   it fixes *trainability*, not the fact that FFN weight matrices are near
   full-rank -> quality ceiling stays low. Gate on the spectrum diagnostic
   (singular values of a real trained FFN: sharp knee -> low-rank viable; fat
   spectrum -> skip).
5. **2:4 semi-structured sparsity. OUT on this HW.** The 2x throughput needs
   NVIDIA sparse Tensor Cores (Ampere+). On RADV/Vulkan a 2:4 mask = store
   zeros, still pay dense compute. Park until/unless on NVIDIA.

## Why NOT the binary-VSA hyperlayer (the prior idea)

Proven inert: see `../vsa_ffn/collapse_proof.py`. Frozen binds + a trainable
decode collapse to `sign(x@P_frozen) @ W_trainable` — a frozen random-features
FFN: less capacity than SwiGLU at *more* bytes/FLOPs (d->10240->d). VSA pays off
only where something downstream *unbinds* (memory 0.0.5, binding head 0.0.6),
never feeding a trainable linear.

## Decision

- `ffn_type` config knob selects the FFN variant per build.
- 0.0.0/0.0.1 baseline = `swiglu` (fp32).
- First experiment: `ternary_swiglu` A/B vs `swiglu` once 0.0.0 exists,
  matched params, scored on generation (not PPL alone), per ladder discipline.
- Monarch + low-rank(+spectrum) are queued behind it.


## Result: ternary A/B (0.0.0, byte-level)

Config: byte tok (V=256), d=128 L=4 B=16 S=96, 600 steps, seed 0, identical
except `ffn_type`. Numpy QAT sim (dense matmul -> measures quality cost only).

| FFN | final ema loss |
|---|---|
| fp32 swiglu | 1.231 |
| ternary_swiglu | 1.225 |

Ternary tracks fp32 across the whole run (step 300: 1.403 vs 1.408); delta is
within seed noise; samples comparably coherent. => BitNet ternary costs ~zero
quality here; ~10x byte savings (1.58-bit weights) is essentially free.

Caveats: tiny model / 600 steps / single seed -- BitNet is known to hold at
scale but confirm at 0.0.1. The QAT sim keeps the matmul dense, so this is the
*quality* result only; the byte/speed win needs the ternary Vulkan kernel
(cpp/src/ops + shader + binding) -- a separate step now that quality is proven.
