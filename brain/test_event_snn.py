"""Tests for the event-driven SNN integration.

  1. EventDrivenSynapsis sparse path == dense path (same weights), multi-bit input
  2. EventDrivenSNNFFN runs end-to-end; layer-2 synapse uses the sparse GPU path
  3. Speedup: event-driven vs GPU-dense (_bridge.linear) at realistic sparsity

Run with cubby-lm's venv python — grilly is a pyproject dependency (editable,
from the local drive repo) and its __init__ auto-registers the compiled
grilly_core module, so nothing is copied into site-packages and no sys.path
wiring is needed for grilly.
"""
import sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__))
CUBBY_ROOT = os.path.dirname(HERE)
if CUBBY_ROOT not in sys.path:
    sys.path.insert(0, CUBBY_ROOT)                               # `brain` package
import numpy as np
from brain.event_snn import (EventDrivenSynapsis, EventDrivenSNNFFN,
                             MiniGIF, grilly_available)

print("grilly available:", grilly_available())

# ── 1. correctness: sparse path == dense path, identical weights ──
rng = np.random.default_rng(0)
x = np.zeros((4, 6, 1024), np.float32)            # (batch, seq, in)
m = rng.random(x.shape) < 0.05                    # ~5% active
x[m] = rng.integers(1, 9, size=int(m.sum())).astype(np.float32)   # multi-bit 1..8

syn_s = EventDrivenSynapsis(1024, 512, seed=7, mode="sparse")
syn_d = EventDrivenSynapsis(1024, 512, seed=7, mode="dense")
syn_d.weight = syn_s.weight.copy(); syn_d.bias = syn_s.bias.copy()
out_s, _ = syn_s.forward(x)
out_d, _ = syn_d.forward(x)
print(f"[1] synapsis  path={syn_s.last_path:<6} max_abs_diff="
      f"{np.max(np.abs(out_s - out_d)):.2e}  shape={out_s.shape}")

# ── 2. end-to-end FFN ──
ffn = EventDrivenSNNFFN(input_dim=512, hidden_dim=1024, num_timesteps=4,
                        threshold=4.0, seed=1, mode="auto")
y = ffn.forward(rng.standard_normal((2, 8, 512)).astype(np.float32))
print(f"[2] ffn out={y.shape} finite={bool(np.isfinite(y).all())} "
      f"syn1_path={ffn.syn1.last_path} syn2_path={ffn.syn2.last_path}")

# ── 3. speedup vs GPU-dense at realistic sparsity ──
if grilly_available():
    from grilly.backend import _bridge as b
    def t_call(fn, iters=30, warm=5):
        for _ in range(warm): fn()
        t0 = time.perf_counter()
        for _ in range(iters): fn()
        return (time.perf_counter() - t0) / iters * 1e3
    print(f"\n[3] event-driven Synapsis vs GPU-dense (M rows, one W upload/call)")
    print(f"    {'M':>5} {'in':>5} {'out':>5} {'dens':>5} "
          f"{'sparse_ms':>10} {'dense_ms':>9} {'numpy_ms':>9} {'spd_vs_dense':>12}")
    for (M, NI, NO, p) in [(256, 1024, 1024, 0.05),
                           (256, 2048, 2048, 0.05),
                           (512, 2048, 2048, 0.02),
                           (256, 2048, 2048, 0.10)]:
        xb = np.zeros((M, NI), np.float32)
        mm = rng.random(xb.shape) < p
        xb[mm] = rng.integers(1, 9, size=int(mm.sum())).astype(np.float32)
        syn = EventDrivenSynapsis(NI, NO, seed=3, mode="sparse")
        W = syn.weight
        sp = t_call(lambda: syn.forward(xb))
        dn = t_call(lambda: b.linear(xb, W, syn.bias))
        npms = t_call(lambda: xb @ W.T + syn.bias, iters=10, warm=2)
        print(f"    {M:>5} {NI:>5} {NO:>5} {p:>5.2f} "
              f"{sp:>10.4f} {dn:>9.4f} {npms:>9.4f} {dn/sp:>11.2f}x")
print("\ndone")
