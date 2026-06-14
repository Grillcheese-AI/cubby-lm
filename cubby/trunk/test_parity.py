"""0.0.0 forward parity: grilly trunk vs an independent numpy reference.

Same weights -> same logits. This validates the grilly forward path (embedding
gather, RMSNorm, MinGRU scan, SwiGLU, tied head) against a from-scratch numpy
implementation of the *intended* math. The MinGRU reference uses grilly's actual
gate (x_scan = sigmoid(g)*tanh(v); a = 0.05 + 0.9*sigmoid(d)) computed as a plain
sequential recurrence -- so this also confirms the GPU log-domain scan matches the
documented formula. Run: python -m cubby.trunk.test_parity
"""
from __future__ import annotations

import numpy as np

import grilly  # noqa: F401
from cubby.config import make_config
from cubby.trunk.model import CubbyLM


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _silu(x):
    return x * _sigmoid(x)


def _rmsnorm(x, g, eps):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * g


def _mingru(g, v, d):
    """Sequential reference for grilly's MinGRU (a = 0.001 + 0.998*sigmoid(d))."""
    x_scan = _sigmoid(g) * np.tanh(v)
    a = 0.001 + 0.998 * _sigmoid(d)
    B, S, D = x_scan.shape
    h = np.empty_like(x_scan)
    prev = np.zeros((B, D), dtype=np.float32)
    for t in range(S):
        prev = a[:, t] * prev + x_scan[:, t]
        h[:, t] = prev
    return h


def reference_forward(model, ids):
    cfg = model.cfg
    eps = cfg.rmsnorm_eps
    E = np.asarray(model.embed.data, np.float32)
    x = E[np.asarray(ids, np.int64)]
    for blk in model.blocks:
        h = _rmsnorm(x, np.asarray(blk.n1.data, np.float32), eps)
        dm = cfg.d_model
        gvd = h @ np.asarray(blk.mix.proj.weight.data, np.float32).T   # (.,3d)
        g, v, d = gvd[..., :dm], gvd[..., dm:2 * dm], gvd[..., 2 * dm:3 * dm]
        x = x + _mingru(g, v, d)
        h2 = _rmsnorm(x, np.asarray(blk.n2.data, np.float32), eps)
        f = cfg.d_ffn
        gu = h2 @ np.asarray(blk.ffn.gate_up.weight.data, np.float32).T  # (.,2*d_ffn)
        ff = _silu(gu[..., :f]) * gu[..., f:2 * f]
        x = x + ff @ np.asarray(blk.ffn.down.weight.data, np.float32).T
    x = _rmsnorm(x, np.asarray(model.final.data, np.float32), eps)
    return x @ E.T


def test_forward_parity():
    np.random.seed(0)
    cfg = make_config("0.0.0", vocab_size=48, d_model=32, n_layers=3, d_ffn=64, seq_len=16)
    model = CubbyLM(cfg)
    ids = np.random.randint(0, cfg.total_vocab, size=(2, 12))
    grilly_logits = np.asarray(model(ids).data, np.float32)
    ref_logits = reference_forward(model, ids)
    diff = float(np.abs(grilly_logits - ref_logits).max())
    rel = diff / (float(np.abs(ref_logits).max()) + 1e-9)
    print(f"  logits shape={grilly_logits.shape}  max_abs_diff={diff:.3e}  rel={rel:.3e}")
    assert diff < 1e-3, f"parity FAILED: max_abs_diff={diff:.3e}"


if __name__ == "__main__":
    test_forward_parity()
    print("\nPASS: grilly 0.0.0 forward matches numpy reference")
