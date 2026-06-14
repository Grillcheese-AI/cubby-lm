"""GPU-resident linear as a tape op: y = x @ W.T (+ bias), W is (out, in).

Wraps grilly_core's GPU `linear` / `linear_backward` (via grilly.backend._bridge)
in a single tape GradFn, so the dominant trunk compute runs on the GPU while
staying in the Python-tape autograd world (min_gru, ternary STE, the custom
GradFns all keep composing). Falls back to numpy if the bridge is unavailable, so
0.0.0 still runs CPU-only.

Because W is an *input* to the GradFn, `linear(x, ternarize_ste(W))` composes:
linear's backward yields grad w.r.t. the ternarized weight, which flows through
the STE into the latent weight. Same for the tied head: `linear(x, embed)` with
embed (V, d) -> logits (..., V), grad merging back into the shared table.
"""
from __future__ import annotations

import numpy as np

from grilly.nn.autograd import Variable, GradFn, _ensure_variable
import grilly.nn.autograd as _ag

try:
    from grilly.backend import _bridge
    _GPU = bool(_bridge.is_available())
except Exception:                      # pragma: no cover
    _bridge, _GPU = None, False

# fp16 transfer was measured to NOT help: the GPU linear cost is per-dispatch
# overhead (submit/wait/stage/download, ~25ms/call fixed), not transfer bytes, so
# halving input bytes did nothing. The lever is fewer dispatches (see model.py
# fused projections), not smaller payloads. Left off.
USE_FP16 = False
_H = np.float16


def _fwd(X, W, b):
    if _GPU:
        if USE_FP16 and b is None:
            return np.asarray(_bridge.linear(X.astype(_H), W.astype(_H), None), np.float32)
        return np.asarray(_bridge.linear(X, W, b), dtype=np.float32)
    Y = X @ W.T
    return (Y + b if b is not None else Y).astype(np.float32)


def _bwd(go, X, W):
    if _GPU:
        if USE_FP16:
            gi, gw, gb = _bridge.linear_backward(go.astype(_H), X.astype(_H), W.astype(_H))
        else:
            gi, gw, gb = _bridge.linear_backward(go, X, W)
        return (np.asarray(gi, np.float32), np.asarray(gw, np.float32),
                np.asarray(gb, np.float32))
    Xf = X.reshape(-1, X.shape[-1])
    gof = go.reshape(-1, go.shape[-1])
    gi = (gof @ W).reshape(X.shape).astype(np.float32)
    gw = (gof.T @ Xf).astype(np.float32)
    gb = gof.sum(0).astype(np.float32)
    return gi, gw, gb


def linear(x, W, bias=None) -> Variable:
    x_var, W_var = _ensure_variable(x), _ensure_variable(W)
    b_var = None if bias is None else _ensure_variable(bias)
    X = np.asarray(x_var.data, np.float32)
    Wd = np.asarray(W_var.data, np.float32)
    b = None if b_var is None else np.asarray(b_var.data, np.float32)
    Y = _fwd(X, Wd, b)
    inputs = [x_var, W_var] + ([b_var] if b_var is not None else [])
    if not (_ag._grad_enabled and any(getattr(v, "requires_grad", False) for v in inputs)):
        return Variable(Y, requires_grad=False)

    def backward_fn(grad_output):
        gi, gw, gb = _bwd(np.asarray(grad_output, np.float32), X, Wd)
        return (gi, gw, gb) if b_var is not None else (gi, gw)

    return Variable(Y, requires_grad=True, grad_fn=GradFn("Linear", backward_fn, inputs))


def backend() -> str:
    return "gpu" if _GPU else "cpu"


def slice_cols(y, lo, hi) -> Variable:
    """Autograd-aware column slice y[..., lo:hi]. Forward is a view; backward
    scatters the upstream grad into a zero buffer of the parent's width. Pure
    CPU, no GPU dispatch -- lets one fused linear feed several consumers (g/v/d,
    gate/up) so N matmul dispatches collapse to 1."""
    y_var = _ensure_variable(y)
    Y = np.asarray(y_var.data, np.float32)
    out = Y[..., lo:hi]
    if not (_ag._grad_enabled and getattr(y_var, "requires_grad", False)):
        return Variable(out, requires_grad=False)
    shape = Y.shape

    def backward_fn(grad_output):
        g = np.zeros(shape, np.float32)
        g[..., lo:hi] = np.asarray(grad_output, np.float32)
        return (g,)

    return Variable(out, requires_grad=True, grad_fn=GradFn("SliceCols", backward_fn, [y_var]))
