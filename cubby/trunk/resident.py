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

# cubby-lm root on the path so `from cubby import trace` works when this file is
# run as a script (as a package import it's already importable).
_CUBBY_ROOT = r"C:\Users\grill\Documents\GitHub\cubby-lm"
if _CUBBY_ROOT not in sys.path:
    sys.path.insert(0, _CUBBY_ROOT)
from cubby import trace

_SPV = _GRILLY_ROOT + r"\shaders\spv"


def make_device():
    dev = gc.Device(); dev.load_shaders(_SPV); return dev


def _R(buf, shape, rg=True):
    r = gc.TensorRef(); r.buffer_id = buf; r.set_shape(shape); r.requires_grad = rg; return r


def _f32(a):
    return np.ascontiguousarray(np.asarray(a, dtype=np.float32))


def force_numpy_reference():
    """Make the model.py path pure-numpy so the resident grilly_core.Device() is
    the ONLY Vulkan context. Two contexts (the resident one + model.py's
    grilly.nn.autograd / _bridge GPU backward) nondeterministically corrupt each
    other. Forces gpu_linear+model off the bridge AND short-circuits autograd's
    LAZY gpu-backward singleton so model.loss().backward() never inits a 2nd
    device (e.g. MinGRU backward). Call before building/using any CubbyLM here."""
    import cubby.trunk.gpu_linear as _gl
    import cubby.trunk.model as _m
    import grilly.nn.autograd as _ag
    _gl._GPU = False
    _m._GPU = False
    _ag._gpu_backward_ops_checked = True       # pretend "already checked"
    _ag._gpu_backward_ops_cache = None         # -> no GPU backend -> CPU/numpy


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

    _WKEYS = ('n1', 'WG', 'WV', 'WD', 'n2', 'gate_up', 'down')

    # --- weight registration (persistent resident + Adam moments) ---------------
    def _register_weights(self):
        """ALL weights -- incl. the tied embedding E -- are PERSISTENT resident,
        each with persistent Adam m/v, so resident AdamW updates them in place.
        E is used as both the embedding table and the tied head weight; its grad
        (head-weight grad + embedding scatter) is assembled fully on-GPU in
        train_step (embedding_scatter_add), so E joins self.opt with no host path."""
        t, d, dff = self.t, self.d, self.dff
        m = self.model

        def regw(arr):
            a = _f32(arr)
            return dict(w=t.register_weight(a.copy()),
                        m=t.register_weight(np.zeros(a.size, np.float32)),
                        v=t.register_weight(np.zeros(a.size, np.float32)), n=int(a.size))

        self.E = regw(m.embed.data)                                  # (V, d) tied, resident
        self.final = regw(m.final.data)
        self.layers = []
        for b in m.blocks:
            W = _f32(b.mix.proj.weight.data); gu = _f32(b.ffn.gate_up.weight.data)
            self.layers.append(dict(
                n1=regw(b.n1.data), WG=regw(W[0:d]), WV=regw(W[d:2 * d]), WD=regw(W[2 * d:3 * d]),
                n2=regw(b.n2.data),
                gate_up=regw(np.concatenate([gu[dff:2 * dff], gu[0:dff]], axis=0)),
                down=regw(b.ffn.down.weight.data),
            ))
        # flat list for the optimizer sweep + GPU grad-norm (E included)
        self.opt = [self.E, self.final] + [self.layers[li][k] for li in range(self.L) for k in self._WKEYS]

    def _w(self):
        return dict(E=self.E['w'], final=self.final['w'],
                    layers=[{k: self.layers[li][k]['w'] for k in self._WKEYS} for li in range(self.L)])

    # --- inference forward (+ optional cubby.trace) -----------------------------
    def forward_ids(self, ids):
        """Resident forward over ids (B, S) -> (logits_buffer_id, B, S). Records
        nothing. Emits cubby.trace per block ONLY when a tracer is active (reads
        block outputs back) so production (trace OFF) pays zero overhead."""
        t = self.t; t.begin()
        emb, logits, cap, nf, B, S = _resident_forward(t, self._w(),
                                                        ids, self.d, self.dff, self.V, self.L)
        tr = trace.current()
        if tr.level > trace.Level.OFF:
            BS = B * S
            with tr.scope("trunk"):
                for li, c in enumerate(cap):
                    r2 = t.read_buffer(c[-1], [BS, self.d]).reshape(B, S, self.d)
                    tr.probe("block%d" % li, r2, topology="layer:%d" % li)
        return logits, B, S

    def logits(self, ids):
        lid, B, S = self.forward_ids(ids)
        return self.t.read_buffer(lid, [B * S, self.V]).reshape(B, S, self.V)

    # --- resident forward+backward -> grads mapped back to model.py params ------
    def grads(self, ids, targets):
        """Resident forward+backward over the persistent weights (E step-scoped).
        Returns grads in model.py layout (see _read_grads)."""
        t = self.t; t.begin()
        w = self._w()
        emb, logits_np, BS = _fb_run(t, w, ids, targets, self.d, self.dff, self.V, self.L)
        return _read_grads(t, w, emb, ids, BS, self.d, self.dff, self.V, self.L)

    # --- one resident training step (fully on-GPU: scatter + norm + AdamW) -------
    def train_step(self, ids, targets, step, lr=3e-4, betas=(0.9, 0.95), eps=1e-8,
                   wd=0.01, max_grad_norm=1.0):
        """Forward+backward then resident AdamW with GLOBAL grad-norm clipping --
        ALL on-GPU, no grad readback. ALL params (incl. the tied embedding E)
        update resident: E's grad = tied-head weight grad + embedding scatter,
        assembled in place by embedding_scatter_add; the global norm is a GPU
        reduction. clip + mean-CE 1/B are folded into the kernel grad_scale (scaled
        BEFORE Adam m/v, since Adam normalizes by sqrt(v)). Returns (mean-CE loss,
        pre-clip global grad-norm, skipped). Skips the step on a non-finite norm."""
        d, V = self.d, self.V
        ids = np.asarray(ids, dtype=np.int64); B, S = ids.shape; BS = B * S
        t = self.t; t.begin()
        w = self._w()
        emb, logits_np, _ = _fb_run(t, w, ids, targets, d, self.dff, V, self.L)
        loss = _ce(logits_np, np.asarray(targets, np.int64).reshape(-1))

        # E grad: scatter the embedding-output grad INTO E's grad buffer (which
        # already holds the tied-head weight grad) -- fully resident, in place.
        ids_u32 = t.register_input_u32(ids.reshape(-1).astype(np.uint32))
        t.embedding_scatter_add(t.get_grad_buffer(emb), ids_u32,
                                t.get_grad_buffer(self.E['w']), BS, d)

        # global L2 norm over MEAN grads (raw/BS), ALL params on-GPU (one scalar).
        sq = t.sum_squares([t.get_grad_buffer(p['w']) for p in self.opt],
                           [p['n'] for p in self.opt])
        gnorm = float(np.sqrt(float(sq))) / BS
        if not np.isfinite(gnorm):
            return loss, gnorm, True                                # poisoned grad -> SKIP
        clip = min(1.0, max_grad_norm / (gnorm + 1e-6)) if max_grad_norm else 1.0
        gscale = clip / BS                                          # clip + mean-CE, in-kernel

        b1, b2 = betas; b1t, b2t = b1 ** step, b2 ** step
        t.forward_begin()
        for p in self.opt:                                          # resident AdamW, one batch
            t.adamw_update(p['w'], t.get_grad_buffer(p['w']), p['m'], p['v'], p['n'],
                           lr, b1, b2, eps, wd, b1t, b2t, False, gscale)
        t.forward_submit()
        return loss, gnorm, False

    # --- autoregressive generation (resident forward) ---------------------------
    def generate(self, prompt_ids, max_new_tokens=80, temperature=0.8, seed=0):
        rng = np.random.default_rng(seed)
        ids = list(np.asarray(prompt_ids, dtype=np.int64).reshape(-1))
        for _ in range(max_new_tokens):
            lg = self.logits(np.asarray(ids, dtype=np.int64)[None, :])[0, -1]
            if not np.isfinite(lg).all():
                break                                       # diverged trunk -> stop, don't crash
            if temperature <= 0:
                nxt = int(lg.argmax())
            else:
                z = lg / temperature; z -= z.max(); p = np.exp(z); p /= p.sum()
                nxt = int(rng.choice(len(p), p=p))
            ids.append(nxt)
        return ids


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


def _resident_forward(t, w, ids, d, dff, V, L):
    """Resident forward over weight buffer ids w={E,final,layers}. Caller has
    t.begin(). Returns (emb_id, logits_id, cap, nf_id, B, S); cap[li] holds every
    intermediate buffer id (n1,G,Vv,D,H,r1,n2,gu,h,ff,r2) for the backward record."""
    E_id, final_id, layers = w['E'], w['final'], w['layers']
    ids = np.asarray(ids, dtype=np.int64); B, S = ids.shape; BS = B * S
    ids_u32 = t.register_input_u32(ids.reshape(-1).astype(np.uint32))
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
    return emb, logits, cap, nf, B, S


def _fb_run(t, w, ids, targets, d, dff, V, L):
    """Resident forward + backward (records the matching graph, one backward()).
    Returns (emb_id, logits_numpy, BS). Grad buffers are then readable via
    t.get_grad_buffer(weight_id) / get_grad_buffer(emb_id)."""
    op = gc.OpType
    E_id, final_id, layers = w['E'], w['final'], w['layers']
    emb, logits, cap, nf, B, S = _resident_forward(t, w, ids, d, dff, V, L)
    BS = B * S
    logits_np = t.read_buffer(logits, [BS, V])
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    tgt_id = t.register_input(targets.astype(np.float32), False)

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
    return emb, logits_np, BS


def _read_grads(t, w, emb, ids, BS, d, dff, V, L):
    """Read grads after _fb_run, mapped to model.py layout (mean-CE /BS): proj
    re-concats the 3 split gvd blocks, gate_up un-swaps, embed merges head + scatter."""
    E_id, final_id, layers = w['E'], w['final'], w['layers']
    ids = np.asarray(ids, dtype=np.int64)

    def gr(b, sh): return t.read_buffer(t.get_grad_buffer(b), sh) / BS
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
    dE = np.zeros((V, d), np.float32); np.add.at(dE, ids.reshape(-1), emb_g)
    out['embed'] = dE + gr(E_id, [V, d])
    return out


def _fb_grads(t, w, ids, targets, d, dff, V, L):
    """forward+backward + read grads (model.py layout). Returns (grads, logits)."""
    emb, logits_np, BS = _fb_run(t, w, ids, targets, d, dff, V, L)
    return _read_grads(t, w, emb, ids, BS, d, dff, V, L), logits_np


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
    force_numpy_reference()                                 # numpy reference (one GPU ctx)
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


def train_cubby_resident(version="0.0.0", steps=600, data="tinystory_50k.json",
                         B=8, S=64, lr=3e-4, max_tokens=4000000, sample_every=200,
                         prompt="Once upon a time", gen_tokens=60, dev=None,
                         warmup=0, max_grad_norm=1.0,
                         ckpt_path=None, ckpt_every=100, resume=True, max_consec_skips=10):
    """Train a CubbyLM via the RESIDENT backend (persistent weights + resident
    AdamW + E host path) on a BBPE-65k stream; sample periodically. The default
    `main.py train` backend. Returns (ResidentTrunk, tokenizer)."""
    import json
    import time as _time
    force_numpy_reference()                                 # resident path owns the GPU
    from cubby.config import make_config
    from cubby.trunk.model import CubbyLM, param_count
    from cubby.tokenizer import make_tokenizer

    import os
    if not os.path.isabs(data) and not os.path.exists(data):
        data = os.path.join(_CUBBY_ROOT, data)             # resolve relative to repo root
    np.random.seed(0)
    cfg = make_config(version)
    tok = make_tokenizer("bbpe65k")
    with open(data, "r", encoding="utf-8") as f:
        stories = json.load(f)
    sb = []
    for s in stories:
        sb.extend(tok.encode(s + "\n"))
        if len(sb) >= max_tokens:
            break
    sb = sb[:max_tokens]
    # Model vocab smaller than the BBPE-65k tokenizer (e.g. the 'tiny' preset)?
    # Compress ids into a dense [0, vocab) space by corpus frequency so targets
    # never exceed the head -- keeps wordpieces, just renumbers them.
    if cfg.total_vocab < tok.vocab_size:
        from cubby.tokenizer import RemapTokenizer
        tok = RemapTokenizer(tok, cfg.total_vocab, sb)
        sb = tok.encode_base(sb)
        print("[remap] V %d->%d  coverage=%.3f%% (rare ids -> <unk>)"
              % (tok.base.vocab_size, tok.vocab_size, tok.coverage * 100), flush=True)
    assert tok.vocab_size == cfg.total_vocab, (tok.vocab_size, cfg.total_vocab)
    stream = np.asarray(sb, dtype=np.int64)
    rng = np.random.default_rng(0)
    def batch():
        ix = rng.integers(0, len(stream) - S - 1, size=B)
        return (np.stack([stream[i:i + S] for i in ix]).astype(np.int64),
                np.stack([stream[i + 1:i + 1 + S] for i in ix]).astype(np.int64))
    def sample():
        s = tok.decode(rt.generate(tok.encode(prompt), max_new_tokens=gen_tokens))
        return s.encode("ascii", "backslashreplace").decode("ascii")          # cp1252 console

    from cubby.trunk import checkpoint as _ckpt
    if ckpt_path is None:
        ckpt_path = os.path.join(_CUBBY_ROOT, "ckpt_%s.grl" % version)

    model = CubbyLM(cfg)
    start_step = 0
    if resume and os.path.exists(ckpt_path):
        _ms, _meta = _ckpt.load_checkpoint(ckpt_path)
        if _ckpt.checkpoint_matches(_meta, cfg):
            _ckpt.apply_model_state(model, _ms)            # restore BEFORE ResidentTrunk()
            start_step = int(_meta.get("step", 0))
            rng = _ckpt.restore_rng(_meta)                 # continue the SAME data stream
            print("[resume] %s @ step %d" % (ckpt_path, start_step), flush=True)
        else:
            print("[resume] shape mismatch, ignoring %s" % ckpt_path, flush=True)
    rt = ResidentTrunk(model, dev or make_device())
    print("[resident] V=%d d=%d L=%d d_ffn=%d params=%d stream=%d B=%d S=%d"
          % (cfg.total_vocab, cfg.d_model, cfg.n_layers, cfg.d_ffn,
             param_count(model), len(stream), B, S), flush=True)
    print("[hp] lr=%.1e warmup=%d clip=%s betas=(0.9,0.95) wd=0.01" % (lr, warmup, max_grad_norm), flush=True)
    guard = _ckpt.SkipGuard(max_consecutive=max_consec_skips)
    step = start_step
    def _save(tag):
        try:
            _ckpt.save_checkpoint(ckpt_path, rt, step=step, rng=rng, version=version,
                                  lr=lr, warmup=warmup, max_grad_norm=max_grad_norm)
            print("[ckpt] %s @ step %d -> %s" % (tag, step, ckpt_path), flush=True)
        except Exception as _e:
            print("[ckpt] save FAILED (%s): %r" % (tag, _e), flush=True)

    t0 = _time.perf_counter(); nskip = 0
    try:
        for step in range(start_step + 1, steps + 1):
            ids, tgt = batch()
            lr_t = lr * min(1.0, step / warmup) if warmup else lr  # linear LR warmup
            loss, gnorm, skipped = rt.train_step(ids, tgt, step, lr=lr_t, max_grad_norm=max_grad_norm)
            nskip += int(skipped)
            print("[%4d/%d] ce=%.3f ppl=%.1f gnorm=%.2e lr=%.1e (%.2f it/s)%s"
                  % (step, steps, loss, np.exp(loss), gnorm, lr_t,
                     step / (_time.perf_counter() - t0), "  [skipped]" if skipped else ""), flush=True)
            if guard.update(skipped):                             # K-in-a-row -> divergence
                print("[abort] %d consecutive non-finite grads" % guard.max_consecutive, flush=True)
                _save("diverge"); break
            if sample_every and step % sample_every == 0:
                print("  sample:", repr(sample()), flush=True)
            if ckpt_every and step % ckpt_every == 0:
                _save("periodic")
    except KeyboardInterrupt:
        print("[interrupt] flushing checkpoint", flush=True); _save("interrupt"); raise
    except Exception as _e:                                       # OOM / any in-step failure
        print("[error] step %d: %r -> emergency checkpoint" % (step, _e), flush=True)
        _save("emergency"); raise
    else:
        _save("final")
    if nskip:
        print("[warn] %d step(s) skipped (non-finite grad)" % nskip, flush=True)
    print("[done] %.1fs  final sample: %r" % (_time.perf_counter() - t0, sample()), flush=True)
    return rt, tok


def train_demo(dev, steps=150, tokens=400000):
    """Train a real CubbyLM via the RESIDENT backend on TinyStories, then
    generate -- the train->generate gate before flipping the default. Also
    asserts cubby.trace emission works from the resident forward."""
    import json
    force_numpy_reference()                                 # only the resident path owns the GPU
    from cubby.config import make_config
    from cubby.trunk.model import CubbyLM
    from cubby.tokenizer import make_tokenizer

    np.random.seed(0)
    cfg = make_config("0.0.0", d_model=256, n_layers=6, d_ffn=512)   # V=65536
    tok = make_tokenizer("bbpe65k")
    assert tok.vocab_size == cfg.total_vocab, (tok.vocab_size, cfg.total_vocab)
    with open(r"C:\Users\grill\Documents\GitHub\cubby-lm\tinystory_50k.json", "r", encoding="utf-8") as f:
        stories = json.load(f)
    sb = []
    for s in stories:
        sb.extend(tok.encode(s + "\n"))
        if len(sb) >= tokens:
            break
    stream = np.asarray(sb[:tokens], dtype=np.int64)
    B, S = 8, 64
    rng = np.random.default_rng(0)
    def batch():
        ix = rng.integers(0, len(stream) - S - 1, size=B)
        return (np.stack([stream[i:i + S] for i in ix]).astype(np.int64),
                np.stack([stream[i + 1:i + 1 + S] for i in ix]).astype(np.int64))

    model = CubbyLM(cfg)
    rt = ResidentTrunk(model, dev)
    print("=== RESIDENT CubbyLM training on TinyStories (V=%d d=%d L=%d, B=%d S=%d) ==="
          % (cfg.total_vocab, cfg.d_model, cfg.n_layers, B, S))
    print("    starting CE ~ ln(V) = %.3f.  step  ce  ppl" % np.log(cfg.total_vocab))
    import time as _time
    t0 = _time.perf_counter()
    for step in range(1, steps + 1):
        ids, tgt = batch()
        loss, gnorm, _ = rt.train_step(ids, tgt, step, lr=3e-3)
        if step == 1 or step % 25 == 0:
            print("  %4d   %.4f   %.1f   gnorm=%.2e" % (step, loss, np.exp(loss), gnorm))
    dt = _time.perf_counter() - t0

    # cubby.trace gate: a traced resident forward emits one record per block
    sink = trace.MemorySink()
    with trace.trace_to(sink, "audit"):
        trace.set_step(0)
        rt.logits(batch()[0][:1])
    nblocks = len([r for r in sink.records if r.component.startswith("block")])
    print("\n  cubby.trace: %d/%d block records emitted from the resident forward"
          % (nblocks, cfg.n_layers))

    # train->generate gate
    seed = tok.encode("Once upon a time")
    out = rt.generate(seed, max_new_tokens=60, temperature=0.8, seed=0)
    sample = tok.decode(out).encode("ascii", "backslashreplace").decode("ascii")  # cp1252 console
    print("  sample: %r" % sample)
    print("\n  %d steps, %.0f ms/step. CE descended -> resident-trained CubbyLM generates."
          % (steps, 1e3 * dt / steps))
    ok = nblocks == cfg.n_layers
    print("RESIDENT TRAIN+GEN:", "PASS" if ok else "FAIL")
    return ok


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
    import os
    sys.path.insert(0, r"C:\Users\grill\Documents\GitHub\cubby-lm")

    if "train" in sys.argv:                              # resident train->generate gate
        _steps = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 150
        ok = train_demo(make_device(), steps=_steps)
        sys.stdout.flush()
        os._exit(0 if ok else 1)

    from cubby.config import make_config
    from cubby.trunk.model import CubbyLM

    import grilly.nn.autograd as _ag
    force_numpy_reference()                              # only the resident path owns the GPU
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
