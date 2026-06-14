"""FFN variants for the Cubby trunk (grilly-native).

- SwiGLU        : fp32 baseline -- down(silu(gate(x)) * up(x)), bias-free.
- TernarySwiGLU : BitNet b1.58 QAT. The three projection weights are ternarized
                  to {-a, 0, +a} (a = mean|W|) in the forward via a straight-
                  through estimator; the latent fp32 weight trains normally.
                  This SIMULATES the multiply-free kernel for a quality A/B --
                  the matmul stays dense here, so it measures only the quality
                  cost of ternarization. The real multiply-free Vulkan kernel
                  (the byte/speed win) is a separate step once quality holds.

Every FFN emits a per-step trace record (cubby.trace) for the audit/visual bus.
"""
from __future__ import annotations

import numpy as np

from grilly.nn.autograd import Variable, GradFn, _ensure_variable
import grilly.nn.autograd as _ag

from cubby import trace
from cubby.trunk.gpu_linear import linear as _linear
from cubby.trunk.gpu_linear import slice_cols as _slice


def ternarize_ste(w, eps: float = 1e-5) -> Variable:
    """Quantize a weight Variable to ternary {-a, 0, +a} with a straight-through
    estimator. forward = round(W/a) clipped to [-1,1] times a (a = mean|W|);
    backward = identity grad to the latent weight. Mirrors the custom-GradFn
    pattern in grilly.nn.prefix_scan."""
    w_var = _ensure_variable(w)
    wd = np.asarray(w_var.data, dtype=np.float32)
    alpha = float(np.abs(wd).mean()) + eps
    q = np.clip(np.round(wd / alpha), -1.0, 1.0)
    wq = (alpha * q).astype(np.float32)
    if not (_ag._grad_enabled and getattr(w_var, "requires_grad", False)):
        return Variable(wq, requires_grad=False)

    def backward_fn(grad_output):
        return (np.asarray(grad_output, dtype=np.float32),)   # STE: pass through

    return Variable(wq, requires_grad=True,
                    grad_fn=GradFn("TernarizeSTE", backward_fn, [w_var]))


def _silu(x):
    """SiLU = x * sigmoid(x). Kept as chained Variable ops -- a fused numpy
    GradFn was measured SLOWER (the cost is numpy exp/elementwise, not tape-node
    overhead). The real fix for elementwise cost is GPU-resident execution (#2)."""
    return x * (0.5 * (1.0 + (x * 0.5).tanh()))


class _Linear:
    """Weight-owning linear, bias-free. Weight stored (out, in); forward is the
    GPU `linear` op (numpy fallback). Ternarization composes through it: the
    ternarized weight is an input to the op's GradFn, so its grad flows via STE
    into the latent weight."""

    def __init__(self, d_in: int, d_out: int, ternary: bool = False, std=None):
        std = (1.0 / d_in ** 0.5) if std is None else std
        self.ternary = ternary
        self.weight = Variable((np.random.randn(d_out, d_in) * std).astype(np.float32),
                               requires_grad=True)

    def parameters(self):
        yield self.weight

    def __call__(self, x):
        w = ternarize_ste(self.weight) if self.ternary else self.weight
        return _linear(x, w)


class SwiGLU:
    """down(silu(gate(x)) * up(x)). gate+up are fused into one (2*d_ffn, d) matmul
    (both read the same input) -- 2 GPU dispatches collapse to 1. ternary=True
    ternarizes all weights; the fused gate_up shares one per-tensor alpha (a
    bigger ternary tensor), which is acceptable BitNet behaviour."""

    def __init__(self, d_model: int, d_ffn: int, ternary: bool = False, name: str = "ffn"):
        self.name = name
        self.ternary = ternary
        self.d_ffn = d_ffn
        self.gate_up = _Linear(d_model, 2 * d_ffn, ternary=ternary)
        self.down = _Linear(d_ffn, d_model, ternary=ternary)

    def parameters(self):
        yield from self.gate_up.parameters()
        yield from self.down.parameters()

    def __call__(self, x):
        gu = self.gate_up(x)
        f = self.d_ffn
        h = _silu(_slice(gu, 0, f)) * _slice(gu, f, 2 * f)
        y = self.down(h)
        trace.probe(self.name, np.asarray(y.data), topology="ffn",
                    meta={"ternary": self.ternary})
        return y


def make_ffn(ffn_type: str, d_model: int, d_ffn: int, name: str = "ffn") -> SwiGLU:
    if ffn_type == "swiglu":
        return SwiGLU(d_model, d_ffn, ternary=False, name=name)
    if ffn_type == "ternary_swiglu":
        return SwiGLU(d_model, d_ffn, ternary=True, name=name)
    raise ValueError(f"unknown ffn_type {ffn_type!r}")
