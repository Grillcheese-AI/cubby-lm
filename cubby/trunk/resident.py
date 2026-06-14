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

    np.random.seed(0)
    # small but real 0.0.0-shaped trunk for a fast parity check
    cfg = make_config("0.0.0", vocab_size=512, d_model=128, n_layers=4, d_ffn=256)
    model = CubbyLM(cfg)
    ids = np.random.randint(0, cfg.total_vocab, size=(2, 8)).astype(np.int64)

    diff, res, ref = forward_parity(model, ids)
    print("=== STEP-5 forward parity: resident trunk vs model.py (numpy) ===")
    print("    cfg: V=%d d=%d L=%d d_ffn=%d  ids=(2,8)" %
          (cfg.total_vocab, cfg.d_model, cfg.n_layers, cfg.d_ffn))
    print("    logits max_abs_diff = %.3e" % diff)
    print("PARITY:", "PASS" if diff < 1e-3 else "FAIL")
    sys.exit(0 if diff < 1e-3 else 1)
