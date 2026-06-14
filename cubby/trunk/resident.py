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

    # --- resident forward+backward -> grads mapped back to model.py params ------
    def grads(self, ids, targets):
        """Resident single-tape forward+backward for (ids, targets). Returns a
        dict of gradients in model.py's parameter layout (mean-CE scaled):
          embed (V,d), final (d,), and per-layer lists n1,n2,proj(3d,d),
          gate_up(2*dff,d),down(d,dff). proj re-concatenates the 3 split blocks;
          gate_up un-swaps the half-swap; embed merges head-weight + scatter."""
        op = gc.OpType
        d, dff, V, L = self.d, self.dff, self.V, self.L
        ids = np.asarray(ids, dtype=np.int64); B, S = ids.shape; BS = B * S
        targets = np.asarray(targets, dtype=np.int64).reshape(-1)
        t = self.t; t.begin()
        E_id, final_id = self.E, self.final
        ids_u32 = t.register_input_u32(ids.reshape(-1).astype(np.uint32))
        tgt_id = t.register_input(targets.astype(np.float32), False)

        # resident forward, capturing every intermediate buffer id
        t.forward_begin()
        emb = t.forward_embedding(ids_u32, E_id, B, S, V, d)
        x = emb; cap = []
        for lw in self.layers:
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

        # record the matching backward nodes (same graph as the forward)
        x = emb
        for li, lw in enumerate(self.layers):
            n1, G, Vv, D, H, r1, n2, gu, h, ff, r2 = cap[li]
            nn1 = t.record_op(op.RMSNorm, [_R(x, [BS, d]), _R(lw['n1'], [d])], [_R(n1, [BS, d])])
            t.save_for_backward(nn1, [x, lw['n1']])
            nG = t.record_op(op.Linear, [_R(n1, [BS, d]), _R(lw['WG'], [d, d])], [_R(G, [BS, d])]); t.save_for_backward(nG, [n1, lw['WG']])
            nV = t.record_op(op.Linear, [_R(n1, [BS, d]), _R(lw['WV'], [d, d])], [_R(Vv, [BS, d])]); t.save_for_backward(nV, [n1, lw['WV']])
            nD = t.record_op(op.Linear, [_R(n1, [BS, d]), _R(lw['WD'], [d, d])], [_R(D, [BS, d])]); t.save_for_backward(nD, [n1, lw['WD']])
            nM = t.record_op(op.MinGRU, [_R(G, [B, S, d]), _R(Vv, [B, S, d]), _R(D, [B, S, d])], [_R(H, [B, S, d])])
            t.save_for_backward(nM, [G, Vv, D, H])
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
        for lw in self.layers:
            out['n1'].append(gr(lw['n1'], [d]))
            out['n2'].append(gr(lw['n2'], [d]))
            out['proj'].append(np.concatenate([gr(lw['WG'], [d, d]), gr(lw['WV'], [d, d]), gr(lw['WD'], [d, d])], axis=0))
            gsw = gr(lw['gate_up'], [2 * dff, d])                       # [up | gate] layout
            out['gate_up'].append(np.concatenate([gsw[dff:2 * dff], gsw[0:dff]], axis=0))  # un-swap -> [gate | up]
            out['down'].append(gr(lw['down'], [d, dff]))
        out['final'] = gr(final_id, [d])
        emb_g = gr(emb, [BS, d])
        dE = np.zeros((V, d), np.float32)
        np.add.at(dE, ids.reshape(-1), emb_g)                           # embedding scatter
        dE = dE + gr(E_id, [V, d])                                      # + tied head weight grad
        out['embed'] = dE
        return out


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
    sys.stdout.flush()
    os._exit(0 if ok else 1)   # skip Python/C++ finalizers (GPU teardown order)
