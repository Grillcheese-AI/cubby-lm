"""Tests for cubby.trunk.ffn -- SwiGLU baseline + TernarySwiGLU (BitNet QAT).

Run: python -m cubby.trunk.test_ffn
The decisive check is test_ste_grads_reach_latent_weight: it proves F.linear
backprops through a *computed* (ternarized) weight into the latent fp32 weight,
which is what makes ternary QAT trainable.
"""
from __future__ import annotations

import numpy as np

import grilly  # noqa: F401  (loads shaders / autograd)
from grilly.nn.autograd import Variable

from cubby import trace
from cubby.trunk.ffn import make_ffn, ternarize_ste, _Linear


def _x(B=2, S=4, d=16):
    return Variable(np.random.randn(B, S, d).astype(np.float32), requires_grad=False)


def test_forward_shape_swiglu():
    ffn = make_ffn("swiglu", 16, 32)
    y = ffn(_x())
    assert np.asarray(y.data).shape == (2, 4, 16)


def test_forward_shape_ternary():
    ffn = make_ffn("ternary_swiglu", 16, 32)
    y = ffn(_x())
    assert np.asarray(y.data).shape == (2, 4, 16)


def test_ternarize_values_are_ternary():
    w = Variable(np.random.randn(32, 16).astype(np.float32), requires_grad=True)
    wq = np.asarray(ternarize_ste(w).data)
    vals = np.unique(np.abs(wq))
    # at most {0, alpha}; every nonzero entry equals the same alpha
    assert len(vals) <= 2
    nz = wq[wq != 0]
    assert np.allclose(np.abs(nz), np.abs(nz)[0])     # single magnitude
    assert (wq == 0).any()                            # some zeros (round to 0)


def test_param_shapes_match_across_variants():
    a = list(make_ffn("swiglu", 16, 32).parameters())
    b = list(make_ffn("ternary_swiglu", 16, 32).parameters())
    assert [np.asarray(p.data).shape for p in a] == [np.asarray(p.data).shape for p in b]


def test_ste_grads_reach_latent_weight():
    lin = _Linear(16, 32, ternary=True)
    y = lin(_x())
    loss = y.mean()
    loss.backward()
    g = lin.weight.grad
    assert g is not None, "STE did not backprop into the latent weight"
    assert np.asarray(g).shape == np.asarray(lin.weight.data).shape
    assert np.isfinite(np.asarray(g)).all()


def test_emits_trace():
    sink = trace.MemorySink()
    with trace.trace_to(sink, "audit"):
        make_ffn("ternary_swiglu", 16, 32, name="ffn0")(_x())
    recs = sink.by_component("ffn0")
    assert len(recs) == 1 and recs[0].meta["ternary"] is True


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
