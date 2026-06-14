"""Resident (GPU-resident, single-tape) execution path for the Cubby trunk.

This lives ALONGSIDE `model.py` (the numpy / Python-tape reference), per the port
discipline: it runs CubbyLM's architecture fully on grilly's resident TapeContext
(register_weight + forward_* + record_op + adamw_update -- the stack validated in
grilly/experimental/resident_train/train_trunk_lm.py), and is gated on FORWARD
PARITY then GRADIENT PARITY vs model.py before it can become the default.

It reads parameters straight out of a built `CubbyLM` (the Variables in model.py),
so the two share identical weights -- the only thing under test is whether the
resident kernels reproduce the numpy trunk.

Two cubby-specific details vs the train_trunk_lm reference:
  - the MinGRU projection is FUSED (one (3d,d) weight) and sliced G|V|D; the
    resident path has no Slice op, so the fused weight is split into three (d,d)
    blocks (rows [0:d]|[d:2d]|[2d:3d]) and run as three forward_linears
    (mathematically identical).
  - cubby SwiGLU is down(silu(gu[:f]) * gu[f:2f]) -- silu on the FIRST half --
    but the resident forward_swiglu computes x1*silu(x2) (silu on the SECOND
    half). So the gate_up weight's two row-blocks are SWAPPED at registration so
    the resident output is up*silu(gate) == silu(gate)*up.
"""
from __future__ import annotations

import sys

import numpy as np

# grilly_core.<abi>.pyd lives at the grilly repo root (copied there by rebuild.ps1);
# the editable install exposes `import grilly` but not the raw extension, so put the
# repo root on the path for the resident TapeContext API.
_GRILLY_ROOT = r"C:\Users\grill\Documents\GitHub\grilly"
if _GRILLY_ROOT not in sys.path:
    sys.path.insert(0, _GRILLY_ROOT)
import grilly_core as gc

_SPV = _GRILLY_ROOT + r"\shaders\spv"


def make_device():
    dev = gc.Device(); dev.load_shaders(_SPV); return dev


def _R(buf, shape, rg=True):
    r = gc.TensorRef(); r.buffer_id = buf; r.set_shape(shape); r.requires_grad = rg; return r


def _f32(a):
    return np.ascontiguousarray(np.asarray(a, dtype=np.float32))


class ResidentTrunk:
    """Resident forward (and, incrementally, backward/optimizer) for a CubbyLM.

    Built from the model's parameter Variables. Weights are registered ONCE as
    persistent resident buffers; forward() re-runs each call over a fresh batch.
    """

    def __init__(self, model, dev=None):
        cfg = model.cfg
        self.d = int(cfg.d_model)
        self.L = int(cfg.n_layers)
        self.dff = int(cfg.d_ffn)
        self.V = int(cfg.total_vocab)
        self.model = model
        self.dev = dev or make_device()
        self.t = gc.TapeContext(self.dev)
        self._register_weights()

    # --- weight registration (persistent resident) -----------------------------
    def _register_weights(self):
        t, d, dff = self.t, self.d, self.dff
        m = self.model
        self.E = t.register_weight(_f32(m.embed.data))               # (V, d) tied
        self.final = t.register_weight(_f32(m.final.data))           # (d,)
        self.layers = []
        for b in m.blocks:
            W = _f32(b.mix.proj.weight.data)                         # (3d, d) fused gvd
            gu = _f32(b.ffn.gate_up.weight.data)                     # (2*dff, d)
            lw = dict(
                n1=t.register_weight(_f32(b.n1.data)),               # (d,)
                WG=t.register_weight(W[0:d]),                        # gvd row blocks
                WV=t.register_weight(W[d:2 * d]),
                WD=t.register_weight(W[2 * d:3 * d]),
                n2=t.register_weight(_f32(b.n2.data)),               # (d,)
                # swap the two halves: resident swiglu does x1*silu(x2), cubby
                # does silu(gate)*up -> feed [up | gate] so it computes up*silu(gate).
                gate_up=t.register_weight(np.concatenate([gu[dff:2 * dff], gu[0:dff]], axis=0)),
                down=t.register_weight(_f32(b.ffn.down.weight.data)),  # (d, dff)
            )
            self.layers.append(lw)

    # --- resident forward -------------------------------------------------------
    def forward_ids(self, ids):
        """Run the resident forward over integer token ids (B, S). Returns
        (logits_buffer_id, B, S). Records nothing for backward (forward only)."""
        d, dff, V = self.d, self.dff, self.V
        ids = np.asarray(ids, dtype=np.int64)
        B, S = ids.shape
        BS = B * S
        t = self.t
        t.begin()
        E_id = self.E
        ids_u32 = t.register_input_u32(ids.reshape(-1).astype(np.uint32))
        t.forward_begin()
        x = t.forward_embedding(ids_u32, E_id, B, S, V, d)
        for lw in self.layers:
            n1 = t.forward_rmsnorm(x, lw['n1'], BS, d)
            G = t.forward_linear(n1, lw['WG'], 0, BS, d, d)
            Vv = t.forward_linear(n1, lw['WV'], 0, BS, d, d)
            D = t.forward_linear(n1, lw['WD'], 0, BS, d, d)
            H = t.forward_mingru(G, Vv, D, B, S, d)
            x = t.forward_add(x, H, BS * d)                          # residual 1
            n2 = t.forward_rmsnorm(x, lw['n2'], BS, d)
            gu = t.forward_linear(n2, lw['gate_up'], 0, BS, d, 2 * dff)
            h = t.forward_swiglu(gu, BS, dff)                        # up*silu(gate)
            ff = t.forward_linear(h, lw['down'], 0, BS, dff, d)
            x = t.forward_add(x, ff, BS * d)                        # residual 2
        nf = t.forward_rmsnorm(x, self.final, BS, d)
        logits = t.forward_linear(nf, E_id, 0, BS, d, V)            # tied head
        t.forward_submit()
        return logits, B, S

    def logits(self, ids):
        lid, B, S = self.forward_ids(ids)
        return self.t.read_buffer(lid, [B * S, self.V]).reshape(B, S, self.V)

    def _weight_ids(self):
        return dict(E=self.E, final=self.final, layers=self.layers)

    # --- resident forward+backward -> grads mapped back to model.py params ------
    def grads(self, ids, targets):
        """Resident single-tape forward+backward over the PERSISTENT weights.
        Returns grads in model.py's layout (see _fb_grads)."""
        self.t.begin()
        g, _ = _fb_grads(self.t, self._weight_ids(), ids, targets, self.d, self.dff, self.V, self.L)
        return g


def _register_layer_ids(t, P, d, dff, L, requires_grad=True):
    """Register a numpy param dict (model.py layout) as resident buffers, applying
    the gvd split and the swiglu half-swap. Returns (E_id, final_id, layers)."""
    reg = (lambda a: t.register_weight(_f32(a))) if False else (lambda a: t.register_input(_f32(a), requires_grad))
    E = reg(P['embed']); final = reg(P['final'])
    layers = []
    for li in range(L):
        Wp = _f32(P['proj'][li]); gu = _f32(P['gate_up'][li])
        layers.append(dict(
            n1=reg(P['n1'][li]), WG=reg(Wp[0:d]), WV=reg(Wp[d:2 * d]), WD=reg(Wp[2 * d:3 * d]),
            n2=reg(P['n2'][li]),
            gate_up=reg(np.concatenate([gu[dff:2 * dff], gu[0:dff]], axis=0)),
            down=reg(P['down'][li]),
        ))
    return E, final, layers


def _fb_grads(t, w, ids, targets, d, dff, V, L):
    """Resident single-tape forward+backward given weight buffer ids w =
    {E, final, layers:[{n1,WG,WV,WD,n2,gate_up,down}]}. Caller has t.begin() and
    registered the weights. Returns (grads_model_layout, logits_numpy).
    grads: embed(V,d), final(d), per-layer lists n1,n2,proj(3d,d),gate_up(2dff,d),
    down(d,dff) -- proj re-concats the 3 split blocks, gate_up un-swaps, embed
    merges head-weight + scatter. mean-CE scaled (/BS)."""
    op = gc.OpType
    E_id, final_id, layers = w['E'], w['final'], w['layers']
    ids = np.asarray(ids, dtype=np.int64); B, S = ids.shape; BS = B * S
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    ids_u32 = t.register_input_u32(ids.reshape(-1).astype(np.uint32))
    tgt_id = t.register_input(targets.astype(np.float32), False)

    t.forward_begin()
    emb = t.forward_embedding(ids_u32, E_id, B, S, V, d)
    x = emb; cap = []
    for lw in layers:
        n1 = t.forward_rmsnorm(x, lw['n1'], BS, d)
        G = t.forward_linear(n1, lw['WG'], 0, BS, d, d)
        Vv = t.forward_linear(n1, lw['WV'], 0, BS, d, d)
        D = t.forward_linear(n1, lw['WD'], 0, BS, d, d)
        H = t.forward_mingru(G, Vv, D, B, S, d)
        r1 = t.forward_add(x, H, BS * d)
        n2 = t.forward_rmsnorm(r1, lw['n2'], BS, d)
        gu = t.forward_linear(n2, lw['gate_up'], 0, BS, d, 2 * dff)
        h = t.forward_swiglu(gu, BS, dff)
        ff = t.forward_linear(h, lw['down'], 0, BS, dff, d)
        r2 = t.forward_add(r1, ff, BS * d)
        cap.append((n1, G, Vv, D, H, r1, n2, gu, h, ff, r2)); x = r2
    nf = t.forward_rmsnorm(x, final_id, BS, d)
    logits = t.forward_linear(nf, E_id, 0, BS, d, V)
    t.forward_submit()
    logits_np = t.read_buffer(logits, [BS, V])

    x = emb
    for li, lw in enumerate(layers):
        n1, G, Vv, D, H, r1, n2, gu, h, ff, r2 = cap[li]
        nn1 = t.record_op(op.RMSNorm, [_R(x, [BS, d]), _R(lw['n1'], [d])], [_R(n1, [BS, d])]); t.save_for_backward(nn1, [x, lw['n1']])
        nG = t.record_op(op.Linear, [_R(n1, [BS, d]), _R(lw['WG'], [d, d])], [_R(G, [BS, d])]); t.save_for_backward(nG, [n1, lw['WG']])
        nV = t.record_op(op.Linear, [_R(n1, [BS, d]), _R(lw['WV'], [d, d])], [_R(Vv, [BS, d])]); t.save_for_backward(nV, [n1, lw['WV']])
        nD = t.record_op(op.Linear, [_R(n1, [BS, d]), _R(lw['WD'], [d, d])], [_R(D, [BS, d])]); t.save_for_backward(nD, [n1, lw['WD']])
        nM = t.record_op(op.MinGRU, [_R(G, [B, S, d]), _R(Vv, [B, S, d]), _R(D, [B, S, d])], [_R(H, [B, S, d])]); t.save_for_backward(nM, [G, Vv, D, H])
        t.record_op(op.Add, [_R(x, [BS, d]), _R(H, [BS, d])], [_R(r1, [BS, d])])
        nn2 = t.record_op(op.RMSNorm, [_R(r1, [BS, d]), _R(lw['n2'], [d])], [_R(n2, [BS, d])]); t.save_for_backward(nn2, [r1, lw['n2']])
        ngu = t.record_op(op.Linear, [_R(n2, [BS, d]), _R(lw['gate_up'], [2 * dff, d])], [_R(gu, [BS, 2 * dff])]); t.save_for_backward(ngu, [n2, lw['gate_up']])
        nsw = t.record_op(op.SwiGLU, [_R(gu, [BS, 2 * dff])], [_R(h, [BS, dff])]); t.save_for_backward(nsw, [gu])
        ndn = t.record_op(op.Linear, [_R(h, [BS, dff]), _R(lw['down'], [d, dff])], [_R(ff, [BS, d])]); t.save_for_backward(ndn, [h, lw['down']])
        t.record_op(op.Add, [_R(r1, [BS, d]), _R(ff, [BS, d])], [_R(r2, [BS, d])])
        x = r2
    nFin = t.record_op(op.RMSNorm, [_R(x, [BS, d]), _R(final_id, [d])], [_R(nf, [BS, d])]); t.save_for_backward(nFin, [x, final_id])
    nHead = t.record_op(op.Linear, [_R(nf, [BS, d]), _R(E_id, [V, d])], [_R(logits, [BS, V])]); t.save_for_backward(nHead, [nf, E_id])
    nCE = t.record_op(op.CrossEntropy, [_R(logits, [BS, V])], [_R(0, [1], False)]); t.save_for_backward(nCE, [logits, tgt_id])
    t.backward(nCE, 0)

    def gr(b, sh): return t.read_buffer(t.get_grad_buffer(b), sh) / BS   # mean-CE
    out = dict(n1=[], n2=[], proj=[], gate_up=[], down=[])
    for lw in layers:
        out['n1'].append(gr(lw['n1'], [d]))
        out['n2'].append(gr(lw['n2'], [d]))
        out['proj'].append(np.concatenate([gr(lw['WG'], [d, d]), gr(lw['WV'], [d, d]), gr(lw['WD'], [d, d])], axis=0))
        gsw = gr(lw['gate_up'], [2 * dff, d])
        out['gate_up'].append(np.concatenate([gsw[dff:2 * dff], gsw[0:dff]], axis=0))
        out['down'].append(gr(lw['down'], [d, dff]))
    out['final'] = gr(final_id, [d])
    emb_g = gr(emb, [BS, d])
    dE = np.zeros((V, d), np.float32)
    np.add.at(dE, ids.reshape(-1), emb_g)
    dE = dE + gr(E_id, [V, d])
    out['embed'] = dE
    return out, logits_np


def _snapshot(model):
    """Init params from a CubbyLM in model.py layout. COPIES (model.py AdamW
    mutates p.data in place; a view would alias the trained weights)."""
    cp = lambda a: np.array(a, dtype=np.float32, copy=True)
    return dict(
        embed=cp(model.embed.data), final=cp(model.final.data),
        n1=[cp(b.n1.data) for b in model.blocks],
        n2=[cp(b.n2.data) for b in model.blocks],
        proj=[cp(b.mix.proj.weight.data) for b in model.blocks],
        gate_up=[cp(b.ffn.gate_up.weight.data) for b in model.blocks],
        down=[cp(b.ffn.down.weight.data) for b in model.blocks],
    )


def _ce(logits, tgt):
    z = logits - logits.max(1, keepdims=True); e = np.exp(z); sm = e / e.sum(1, keepdims=True)
    return float(-np.log(sm[np.arange(len(tgt)), tgt] + 1e-12).mean())


def _adamw_np(P, g, m, v, step, lr, b1, b2, eps, wd):
    """In-place AdamW over a model-layout param dict, matching model.py AdamW."""
    b1c, b2c = 1.0 - b1 ** step, 1.0 - b2 ** step

    def upd(w, gg, mm, vv):
        mm *= b1; mm += (1 - b1) * gg
        vv *= b2; vv += (1 - b2) * (gg * gg)
        w *= (1 - lr * wd)
        w -= lr * (mm / b1c) / (np.sqrt(vv / b2c) + eps)

    for k in ('embed', 'final'):
        upd(P[k], g[k], m[k], v[k])
    for k in ('n1', 'n2', 'proj', 'gate_up', 'down'):
        for i in range(len(P[k])):
            upd(P[k][i], g[k][i], m[k][i], v[k][i])


def loss_curve_match(dev, steps=30):
    """Train two trunks from IDENTICAL init on the SAME batch: model.py (numpy
    fwd/bwd + numpy AdamW) vs numpy params driven by the RESIDENT kernels for
    grads + the same numpy AdamW. The only difference is the forward/backward
    source, so the loss curves must track -- proving the resident path drives
    identical learning. (resident AdamW + persistent weights are gated separately
    in grilly; here AdamW is held identical to isolate the trunk.)"""
    import cubby.trunk.gpu_linear as _gl
    import cubby.trunk.model as _m
    _gl._GPU = False; _m._GPU = False                       # numpy reference (one GPU ctx)
    from cubby.config import make_config
    from cubby.trunk.model import CubbyLM, AdamW

    np.random.seed(1)
    cfg = make_config("0.0.0", vocab_size=256, d_model=128, n_layers=3, d_ffn=256)
    d, L, dff, V = cfg.d_model, cfg.n_layers, cfg.d_ffn, cfg.total_vocab
    B, S = 4, 8
    ids = np.random.randint(0, V, (B, S)).astype(np.int64)
    tgt = np.random.randint(0, V, (B, S)).astype(np.int64)   # fixed batch -> memorize
    lr, b1, b2, eps, wd = 0.01, 0.9, 0.95, 1e-8, 0.0

    ref_model = CubbyLM(cfg)
    P0 = _snapshot(ref_model)

    # reference: model.py numpy training
    opt = AdamW(ref_model.parameters(), lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd)
    ref = []
    for _ in range(steps):
        loss = ref_model.loss(ids, tgt)
        ref.append(float(loss.data))
        opt.zero_grad(); loss.backward(); opt.step()

    # resident: numpy params from the same init, grads from the resident kernels
    import copy as _copy
    t = gc.TapeContext(dev)
    P = _copy.deepcopy(P0)
    m = {k: (np.zeros_like(P[k]) if not isinstance(P[k], list) else [np.zeros_like(a) for a in P[k]]) for k in P}
    v = {k: (np.zeros_like(P[k]) if not isinstance(P[k], list) else [np.zeros_like(a) for a in P[k]]) for k in P}
    res = []
    for step in range(1, steps + 1):
        t.begin()
        E, final, layers = _register_layer_ids(t, P, d, dff, L)
        g, logits = _fb_grads(t, dict(E=E, final=final, layers=layers), ids, tgt, d, dff, V, L)
        res.append(_ce(logits, tgt.reshape(-1)))
        _adamw_np(P, g, m, v, step, lr, b1, b2, eps, wd)
    return ref, res


def gradient_parity(model, ids, targets, dev=None):
    """Per-parameter max_abs_diff between resident grads and model.py's tape grads."""
    for p in model.parameters():
        p.grad = None
    loss = model.loss(np.asarray(ids, np.int64), np.asarray(targets, np.int64))
    loss.backward()
    rt = ResidentTrunk(model, dev)
    rg = rt.grads(ids, targets)

    def rel(a, b):
        a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
        return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-9))

    results = {}
    results['embed'] = rel(rg['embed'], model.embed.grad)
    results['final'] = rel(rg['final'], model.final.grad)
    for li, b in enumerate(model.blocks):
        results['n1[%d]' % li] = rel(rg['n1'][li], b.n1.grad)
        results['n2[%d]' % li] = rel(rg['n2'][li], b.n2.grad)
        results['proj[%d]' % li] = rel(rg['proj'][li], b.mix.proj.weight.grad)
        results['gate_up[%d]' % li] = rel(rg['gate_up'][li], b.ffn.gate_up.weight.grad)
        results['down[%d]' % li] = rel(rg['down'][li], b.ffn.down.weight.grad)
    return results


def forward_parity(model, ids, dev=None):
    """max_abs_diff between the resident forward and model.py's numpy forward."""
    import grilly.nn.autograd as _ag
    rt = ResidentTrunk(model, dev)
    res = rt.logits(ids)
    with _ag.no_grad():
        ref = np.asarray(model(np.asarray(ids, dtype=np.int64)).data, dtype=np.float32)
    return float(np.abs(res - ref).max()), res, ref


if __name__ == "__main__":
    import sys
    sys.path.insert(0, r"C:\Users\grill\Documents\GitHub\cubby-lm")
    from cubby.config import make_config
    from cubby.trunk.model import CubbyLM

    import os
    import grilly.nn.autograd as _ag
    # Force the model.py reference onto PURE NUMPY: it uses grilly.backend._bridge
    # (a SECOND Vulkan context) for its linears/softmax, which competes with the
    # resident grilly_core.Device() -> corrupted grads + teardown fastfail. The
    # reference is the numpy oracle; only the resident path should touch the GPU.
    import cubby.trunk.gpu_linear as _gl
    import cubby.trunk.model as _m
    _gl._GPU = False
    _m._GPU = False
    np.random.seed(0)
    # small but real 0.0.0-shaped trunk for a fast parity check
    cfg = make_config("0.0.0", vocab_size=512, d_model=128, n_layers=4, d_ffn=256)
    model = CubbyLM(cfg)
    ids = np.random.randint(0, cfg.total_vocab, size=(2, 8)).astype(np.int64)
    tgts = np.random.randint(0, cfg.total_vocab, size=(2, 8)).astype(np.int64)
    dev = make_device()
    rt = ResidentTrunk(model, dev)                       # ONE resident trunk for both gates

    print("=== STEP-5 PARITY: resident trunk vs model.py (numpy/Python-tape) ===")
    print("    cfg: V=%d d=%d L=%d d_ffn=%d  ids=(2,8)\n" %
          (cfg.total_vocab, cfg.d_model, cfg.n_layers, cfg.d_ffn))

    # forward parity
    res = rt.logits(ids)
    with _ag.no_grad():
        ref = np.asarray(model(ids).data, dtype=np.float32)
    diff = float(np.abs(res - ref).max())
    print("  [forward] logits max_abs_diff = %.3e  %s"
          % (diff, "PASS" if diff < 1e-3 else "FAIL"))

    # gradient parity (model.py tape backward as reference)
    for p in model.parameters():
        p.grad = None
    model.loss(ids, tgts).backward()
    rg = rt.grads(ids, tgts)
    def _rel(a, b):
        a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
        return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-9))
    gp = {'embed': _rel(rg['embed'], model.embed.grad), 'final': _rel(rg['final'], model.final.grad)}
    for li, b in enumerate(model.blocks):
        gp['n1[%d]' % li] = _rel(rg['n1'][li], b.n1.grad)
        gp['n2[%d]' % li] = _rel(rg['n2'][li], b.n2.grad)
        gp['proj[%d]' % li] = _rel(rg['proj'][li], b.mix.proj.weight.grad)
        gp['gate_up[%d]' % li] = _rel(rg['gate_up'][li], b.ffn.gate_up.weight.grad)
        gp['down[%d]' % li] = _rel(rg['down'][li], b.ffn.down.weight.grad)
    print("  [gradient] per-param rel_err vs model.py tape:")
    worst = 0.0
    for k in sorted(gp, key=lambda s: (s.split('[')[0], s)):
        worst = max(worst, gp[k])
        print("    %-12s rel_err=%.3e  %s" % (k, gp[k], "PASS" if gp[k] < 1e-2 else "FAIL"))
    ok = diff < 1e-3 and worst < 1e-2
    print("\n  worst gradient rel_err = %.3e" % worst)
    print("STEP-5 PARITY:", "PASS" if ok else "FAIL")

    # loss-curve match: resident-driven training vs model.py numpy training
    print("\n=== STEP-5 loss-curve match: resident-grad training vs model.py (numpy) ===")
    ref, res = loss_curve_match(dev, steps=30)
    print("  step   model.py   resident   |diff|")
    worst_l = 0.0
    for s in range(0, len(ref), 5):
        dlt = abs(ref[s] - res[s]); worst_l = max(worst_l, dlt)
        print("  %4d   %.5f    %.5f   %.2e" % (s, ref[s], res[s], dlt))
    worst_l = max(abs(a - b) for a, b in zip(ref, res))
    descended = res[-1] < 0.5 * res[0]
    ok_l = worst_l < 1e-2 and descended
    print("  worst |loss diff| over all %d steps = %.3e ; descended %.3f->%.3f"
          % (len(ref), worst_l, res[0], res[-1]))
    print("LOSS-CURVE MATCH:", "PASS" if ok_l else "FAIL")

    print("\nSTEP-5 OVERALL:", "PASS" if (ok and ok_l) else "FAIL")
    sys.stdout.flush()
    os._exit(0 if (ok and ok_l) else 1)   # skip Python/C++ finalizers (GPU teardown order)
