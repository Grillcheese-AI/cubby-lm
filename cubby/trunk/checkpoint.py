"""Crash guardrail for the resident Cubby trainer: checkpoint save/restore on
grilly's .grl format + a skip/divergence circuit-breaker.

Division of labor: grilly owns the .grl serializer (grilly.utils.grl_checkpoint);
this module owns the resident-side glue -- reading the persistent weight buffers
out of a ResidentTrunk into a host state_dict, restoring them on resume, the data
RNG round-trip, and the consecutive-skip breaker.

Why weights round-trip through the model.py layout: the resident TapeContext only
exposes register_weight/register_input (create) + read_buffer (read) -- there is no
write-into-an-existing-buffer. So a resume rebuilds a ResidentTrunk the normal way
(register from a CubbyLM whose Variables we've restored), which means saving the
weights in model.py layout: the gvd proj un-split and the SwiGLU gate_up un-swapped
(ResidentTrunk re-applies both at registration). Adam m/v are saved too (resident
layout, for a future exact resume) but not consumed by this weights-only restore --
on resume the optimizer moments restart from zero, a mild self-correcting transient
under grad-clip + warmup, not a divergence.

Atomic writes (temp + os.replace) so a crash mid-write can't corrupt the live file.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from grilly.utils.grl_checkpoint import save_grl, load_grl


def _read(rt, buf, shape):
    return np.asarray(rt.t.read_buffer(buf, list(shape)), dtype=np.float32)


def resident_model_state(rt):
    """Read a ResidentTrunk's persistent weights back in model.py layout (un-split
    gvd proj, un-swapped gate_up) -- the layout ResidentTrunk re-registers from."""
    d, dff, V = rt.d, rt.dff, rt.V
    blocks = {}
    for i, lw in enumerate(rt.layers):
        WG = _read(rt, lw['WG']['w'], [d, d])
        WV = _read(rt, lw['WV']['w'], [d, d])
        WD = _read(rt, lw['WD']['w'], [d, d])
        gu = _read(rt, lw['gate_up']['w'], [2 * dff, d])
        blocks[str(i)] = dict(
            n1=_read(rt, lw['n1']['w'], [d]),
            n2=_read(rt, lw['n2']['w'], [d]),
            proj=np.concatenate([WG, WV, WD], axis=0),                    # (3d, d)
            gate_up=np.concatenate([gu[dff:2 * dff], gu[0:dff]], axis=0),  # un-swap
            down=_read(rt, lw['down']['w'], [d, dff]),
        )
    return dict(embed=_read(rt, rt.E['w'], [V, d]),
                final=_read(rt, rt.final['w'], [d]),
                blocks=blocks)


def resident_moments(rt):
    """Resident-layout Adam m/v per opt param (saved for a future exact resume;
    not consumed by the weights-only restore below)."""
    return {str(j): dict(m=_read(rt, p['m'], [p['n']]), v=_read(rt, p['v'], [p['n']]))
            for j, p in enumerate(rt.opt)}


def save_checkpoint(path, rt, *, step, rng, version, lr, warmup, max_grad_norm,
                    best_ppl=None, save_moments=True):
    """Atomically write a .grl crash checkpoint of a ResidentTrunk's full state."""
    # NB: load_grl re-nests the whole state under a top-level "model" key, so we
    # must NOT add our own "model" wrapper here (that would double-nest and make
    # load_checkpoint return {"model":..., "opt":...} instead of the real keys).
    state = dict(resident_model_state(rt))     # {embed, final, blocks}
    if save_moments:
        state["opt"] = resident_moments(rt)
    meta = dict(
        schema_cubby="cubby.resident.v1", step=int(step), version=str(version),
        d=int(rt.d), L=int(rt.L), dff=int(rt.dff), V=int(rt.V),
        lr=float(lr), warmup=int(warmup), max_grad_norm=float(max_grad_norm),
        rng=rng.bit_generator.state, has_moments=bool(save_moments),
    )
    if best_ppl is not None:
        meta["best_ppl"] = float(best_ppl)
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    save_grl(tmp, state, metadata=meta)
    os.replace(tmp, path)                      # atomic on the same filesystem


def load_checkpoint(path):
    """Load a .grl checkpoint -> (model_state nested dict, metadata dict)."""
    out = load_grl(path)
    return out["model"], out["metadata"]


def checkpoint_matches(meta, cfg):
    """True iff the checkpoint shape matches the config (refuse a mismatched resume)."""
    return (int(meta.get("d", -1)) == int(cfg.d_model) and
            int(meta.get("L", -1)) == int(cfg.n_layers) and
            int(meta.get("dff", -1)) == int(cfg.d_ffn) and
            int(meta.get("V", -1)) == int(cfg.total_vocab))


def apply_model_state(model, model_state):
    """Restore saved model-layout weights INTO a CubbyLM's Variables, in place, so a
    freshly-built ResidentTrunk(model) comes up with the trained weights."""
    model.embed.data[...] = model_state["embed"]
    model.final.data[...] = model_state["final"]
    for i, b in enumerate(model.blocks):
        bs = model_state["blocks"][str(i)]
        b.n1.data[...] = bs["n1"]
        b.n2.data[...] = bs["n2"]
        b.mix.proj.weight.data[...] = bs["proj"]
        b.ffn.gate_up.weight.data[...] = bs["gate_up"]
        b.ffn.down.weight.data[...] = bs["down"]


def restore_rng(meta):
    """Rebuild the data-sampler Generator from saved bit-generator state."""
    g = np.random.default_rng()
    st = meta.get("rng")
    if st is not None:
        try:
            g.bit_generator.state = st
        except Exception:
            pass
    return g


class SkipGuard:
    """Consecutive-skip circuit breaker. train_step returns skipped=True on a
    non-finite grad-norm; an isolated skip is a blip, but K in a row is divergence
    -- abort so a poisoned run doesn't burn the rest of the budget."""

    def __init__(self, max_consecutive=10):
        self.max_consecutive = int(max_consecutive)
        self.consecutive = 0
        self.total = 0

    def update(self, skipped):
        """Record a step outcome; return True if the run should abort."""
        if skipped:
            self.consecutive += 1
            self.total += 1
        else:
            self.consecutive = 0
        return self.consecutive >= self.max_consecutive
