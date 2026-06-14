"""Cubby 0.0.0 trunk -- grilly-native, Python-tape autograd.

embed -> [RMSNorm -> MinGRU(scan) -> RMSNorm -> SwiGLU]xL -> RMSNorm -> tied head.

Everything is a tape Variable (grilly.nn.autograd), because the mixer (min_gru /
prefix_scan_causal) returns tape Variables -- so the whole trunk backprops in one
world. Ops whose tape-reduction support is uncertain (embedding gather, RMSNorm,
cross-entropy) are explicit GradFns with hand-derived backward, mirroring the
custom-GradFn pattern in grilly.nn.prefix_scan. Optimizer is numpy AdamW over the
tape params (the gc.adamw_update GPU kernel is the 0.0.1 swap).

Decay gate: min_gru uses x_scan = sigmoid(g)*tanh(v), a = 0.001 + 0.998*sigmoid(d)
(verified bit-close against grilly's own mingru parity test and a numpy reference).
"""
from __future__ import annotations

import numpy as np

from grilly.nn.autograd import Variable, GradFn, _ensure_variable
import grilly.nn.autograd as _ag
from grilly.nn.prefix_scan import min_gru

from cubby import trace
from cubby.trunk.ffn import _Linear, make_ffn
from cubby.trunk.gpu_linear import linear as _linear
from cubby.trunk.gpu_linear import slice_cols as _slice
from cubby.trunk.gpu_linear import _bridge as _bridge, _GPU as _GPU


def embedding(weight, ids) -> Variable:
    """Gather rows of `weight` (V, d) by integer `ids` (B, S) -> (B, S, d).
    Backward scatter-adds upstream grad back into the (tied) table."""
    w = _ensure_variable(weight)
    W = np.asarray(w.data, dtype=np.float32)
    ids = np.asarray(ids, dtype=np.int64)
    out = W[ids]
    if not (_ag._grad_enabled and getattr(w, "requires_grad", False)):
        return Variable(out, requires_grad=False)

    def backward_fn(grad_output):
        g = np.asarray(grad_output, dtype=np.float32)
        dW = np.zeros_like(W)
        np.add.at(dW, ids.reshape(-1), g.reshape(-1, g.shape[-1]))
        return (dW,)

    return Variable(out, requires_grad=True, grad_fn=GradFn("Embedding", backward_fn, [w]))


def rmsnorm(x, weight, eps: float = 1e-6) -> Variable:
    """RMSNorm: y = x / sqrt(mean(x^2, -1) + eps) * g. Explicit backward."""
    x_var = _ensure_variable(x)
    g_var = _ensure_variable(weight)
    X = np.asarray(x_var.data, dtype=np.float32)
    G = np.asarray(g_var.data, dtype=np.float32)
    ms = np.mean(X * X, axis=-1, keepdims=True)
    r = 1.0 / np.sqrt(ms + eps)                      # (..., 1)
    xhat = X * r
    out = xhat * G
    if not (_ag._grad_enabled and (x_var.requires_grad or g_var.requires_grad)):
        return Variable(out, requires_grad=False)
    n = X.shape[-1]

    def backward_fn(grad_output):
        dy = np.asarray(grad_output, dtype=np.float32)
        u = dy * G                                   # (..., d)
        s = np.sum(u * X, axis=-1, keepdims=True)    # (..., 1)
        dx = r * u - (r ** 3 * X / n) * s
        dG = np.sum(dy * xhat, axis=tuple(range(dy.ndim - 1)))   # -> (d,)
        return (dx.astype(np.float32), dG.astype(np.float32))

    return Variable(out, requires_grad=True,
                    grad_fn=GradFn("RMSNorm", backward_fn, [x_var, g_var]))


def cross_entropy(logits, targets) -> Variable:
    """Mean token cross-entropy. logits (B,S,V), targets int (B,S). On GPU the
    softmax/loss + (softmax-onehot)/N backward run via _bridge (offloads the
    V=65k softmax); numpy fallback otherwise. Backward: (softmax - onehot) / N."""
    z = _ensure_variable(logits)
    Z = np.asarray(z.data, dtype=np.float32)
    B, S, V = Z.shape
    t = np.asarray(targets, dtype=np.int64).reshape(-1)
    Zf = Z.reshape(-1, V)
    N = Zf.shape[0]
    if _GPU:
        per_row = np.asarray(_bridge.cross_entropy_loss(Zf, t), dtype=np.float32)
        loss = float(per_row.mean())
    else:
        m = Zf.max(axis=1, keepdims=True)
        sm = np.exp(Zf - m); sm /= sm.sum(axis=1, keepdims=True)
        loss = float(-np.log(sm[np.arange(N), t] + 1e-12).mean())
    if not (_ag._grad_enabled and z.requires_grad):
        return Variable(np.float32(loss), requires_grad=False)

    def backward_fn(grad_output):
        go = float(np.asarray(grad_output))
        if _GPU:
            sm = np.asarray(_bridge.softmax(Zf), dtype=np.float32)   # GPU softmax (offloads V=65k exp)
        else:
            mm = Zf.max(axis=1, keepdims=True)
            sm = np.exp(Zf - mm); sm /= sm.sum(axis=1, keepdims=True)
        # _bridge.cross_entropy_backward was verified to use a different convention
        # (grad diff 3.11 vs (sm-onehot)/N), so finalize in numpy -- cheap vs exp.
        d = sm.copy()
        d[np.arange(N), t] -= 1.0
        d *= (go / N)
        return (d.reshape(B, S, V).astype(np.float32),)

    return Variable(np.float32(loss), requires_grad=True,
                    grad_fn=GradFn("CrossEntropy", backward_fn, [z]))


class MinGRUMixer:
    """Tape MinGRU: g/v/d = Linear(x); h = min_gru(g, v, d). Tape _Linear weights
    (not nn.Linear, whose C++ Parameters the tape would treat as constants)."""

    def __init__(self, d_model: int):
        self.d = d_model
        # Fused g/v/d projection: one (3d, d) matmul instead of three (d, d) --
        # all three read the same input, so concatenating the weights collapses
        # 3 GPU dispatches to 1 (the measured bottleneck is per-dispatch overhead).
        self.proj = _Linear(d_model, 3 * d_model)

    def parameters(self):
        yield from self.proj.parameters()

    def __call__(self, x):
        d = self.d
        gvd = self.proj(x)
        return min_gru(_slice(gvd, 0, d), _slice(gvd, d, 2 * d), _slice(gvd, 2 * d, 3 * d))


class Block:
    """Pre-norm: x = x + mix(norm1(x)); x = x + ffn(norm2(x))."""

    def __init__(self, cfg, idx: int):
        d = cfg.d_model
        self.idx = idx
        self.eps = cfg.rmsnorm_eps
        self.n1 = Variable(np.ones((d,), dtype=np.float32), requires_grad=True)
        self.mix = MinGRUMixer(d)
        self.n2 = Variable(np.ones((d,), dtype=np.float32), requires_grad=True)
        self.ffn = make_ffn(cfg.ffn_type, d, cfg.d_ffn, name=f"ffn{idx}")

    def parameters(self):
        yield self.n1
        yield from self.mix.parameters()
        yield self.n2
        yield from self.ffn.parameters()

    def __call__(self, x):
        x = x + self.mix(rmsnorm(x, self.n1, self.eps))
        x = x + self.ffn(rmsnorm(x, self.n2, self.eps))
        trace.probe(f"block{self.idx}", np.asarray(x.data), topology=f"layer:{self.idx}")
        return x


class CubbyLM:
    """0.0.0 substrate: embed -> blocks -> final RMSNorm -> tied linear head."""

    def __init__(self, cfg):
        self.cfg = cfg
        V, d = cfg.total_vocab, cfg.d_model
        self.embed = Variable((np.random.randn(V, d) * cfg.embed_init_std).astype(np.float32),
                              requires_grad=True)
        self.blocks = [Block(cfg, i) for i in range(cfg.n_layers)]
        self.final = Variable(np.ones((d,), dtype=np.float32), requires_grad=True)

    def parameters(self):
        yield self.embed
        for b in self.blocks:
            yield from b.parameters()
        yield self.final

    def __call__(self, ids):
        x = embedding(self.embed, ids)
        with trace.scope("trunk"):
            for b in self.blocks:
                x = b(x)
        x = rmsnorm(x, self.final, self.cfg.rmsnorm_eps)
        logits = _linear(x, self.embed)         # tied head (V,d): grads merge into embed
        return logits

    def loss(self, ids, targets):
        return cross_entropy(self(ids), targets)

    def generate(self, prompt_ids, max_new_tokens=128, temperature=1.0):
        ids = list(np.asarray(prompt_ids, dtype=np.int64).reshape(-1))
        for _ in range(max_new_tokens):
            arr = np.asarray(ids, dtype=np.int64)[None, :]
            with _ag.no_grad():
                logits = np.asarray(self(arr).data)[0, -1]
            if temperature <= 0:
                nxt = int(logits.argmax())
            else:
                z = logits / temperature
                p = np.exp(z - z.max()); p /= p.sum()
                nxt = int(np.random.choice(len(p), p=p))
            ids.append(nxt)
        return ids


class AdamW:
    """Numpy AdamW over tape Variables (.data / .grad). 0.0.0 optimizer; the GPU
    gc.adamw_update kernel is the 0.0.1 swap for the scaled run."""

    def __init__(self, params, lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01):
        self.params = list(params)
        self.lr, self.eps, self.wd = lr, eps, weight_decay
        self.b1, self.b2 = betas
        self.m = [np.zeros_like(np.asarray(p.data, np.float32)) for p in self.params]
        self.v = [np.zeros_like(np.asarray(p.data, np.float32)) for p in self.params]
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        c1, c2 = 1.0 - self.b1, 1.0 - self.b2
        lr, eps, wd = self.lr, self.eps, self.wd
        for i, p in enumerate(self.params):
            g = getattr(p, "grad", None)
            if g is None:
                continue
            g = np.asarray(g, dtype=np.float32)
            # In-place updates: measured ~293ms vs ~489ms allocating (and vs
            # ~429ms GPU adamw_update, which loses to per-param round-trips).
            m_, v_ = self.m[i], self.v[i]
            m_ *= self.b1; m_ += c1 * g
            v_ *= self.b2; v_ += c2 * (g * g)
            denom = np.sqrt(v_ / bc2); denom += eps
            w = np.asarray(p.data, dtype=np.float32)
            w *= (1.0 - lr * wd)             # decoupled weight decay (AdamW)
            w -= lr * (m_ / bc1) / denom
            p.data = w


def param_count(model) -> int:
    return int(sum(np.asarray(p.data).size for p in model.parameters()))
