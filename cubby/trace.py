"""cubby.trace - per-step audit + visualization bus.

Requirement: every step, even the smallest, must be auditable AND visualizable -
in real time per example (which neuron fires lights up) and inside tests (golden
traces + per-step asserts for debugging).

Every component writes named records to the *active* Tracer. One record = one
atomic step: which units/neurons were involved, their activations, plus typed
events (routing decisions, expert spawns, gate escalations, VSA cosine hits).
The same stream feeds the real-time viewer and the test harness.

Levels (cost model):
    OFF     no-op; probe() returns before touching the array (production).
    AUDIT   cheap scalar summaries (mean/std/absmax/nonzero_frac) + events.
    VISUAL  + per-unit intensities / spike ids for lighting neurons up.
    FULL    + full arrays, for parity debugging (per-step max_abs_diff).

Nothing here imports grilly: it summarizes any array-like (numpy if present), so
it is testable standalone and adds ~zero deps.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, asdict
from enum import IntEnum
from typing import Any, Callable, Iterator

try:
    import numpy as _np
except Exception:        # numpy optional; summaries degrade gracefully
    _np = None


class Level(IntEnum):
    OFF = 0
    AUDIT = 1
    VISUAL = 2
    FULL = 3


_LEVEL_NAMES = {"off": Level.OFF, "audit": Level.AUDIT,
                "visual": Level.VISUAL, "full": Level.FULL}


def level_from(name) -> "Level":
    return name if isinstance(name, Level) else _LEVEL_NAMES[str(name).lower()]


@dataclass
class TraceRecord:
    step: int                          # example / token step index
    scope: str                         # nesting path, e.g. "trunk/layer3/mingru"
    component: str                     # short tag, e.g. "mingru"
    kind: str                          # "activation" | "event" | "tensor"
    summary: dict                      # cheap scalars (always present)
    topology: Any = None               # how units map in space, e.g. "layer:3"
    unit_ids: Any = None               # which units/neurons (VISUAL+)
    intensities: Any = None            # normalized 0..1 per unit (VISUAL+)
    values: Any = None                 # full array (FULL only)
    meta: dict = field(default_factory=dict)
    t: float = field(default_factory=time.perf_counter)

    def to_json(self) -> str:
        d = asdict(self)
        if _np is not None and isinstance(self.values, _np.ndarray):
            d["values"] = self.values.tolist()
        return json.dumps(d, default=str)


def _summarize(x) -> dict:
    """Cheap, always-on scalar summary of an array-like."""
    if _np is not None:
        a = _np.asarray(x)
        if a.size == 0:
            return {"shape": list(a.shape), "empty": True}
        flat = a.reshape(-1)
        return {
            "shape": list(a.shape),
            "mean": float(flat.mean()),
            "std": float(flat.std()),
            "min": float(flat.min()),
            "max": float(flat.max()),
            "absmax": float(_np.abs(flat).max()),
            "nonzero_frac": float((flat != 0).mean()),
        }
    try:
        flat = list(x)
        return {"len": len(flat), "min": min(flat), "max": max(flat)}
    except Exception:
        return {"scalar": float(x)}


def _intensities(x, unit_ids):
    """Reduce an activation tensor to per-unit normalized intensities (0..1)."""
    if _np is None:
        return (list(unit_ids) if unit_ids is not None else None, None)
    a = _np.abs(_np.asarray(x))
    a = a.reshape(-1, a.shape[-1]).mean(axis=0) if a.ndim > 1 else a
    m = float(a.max()) or 1.0
    inten = (a / m).tolist()
    ids = list(unit_ids) if unit_ids is not None else list(range(len(inten)))
    return ids, inten


# ── sinks ───────────────────────────────────────────────────────────────
class MemorySink:
    """Collects records in a list (tests)."""
    def __init__(self) -> None:
        self.records: list = []
    def write(self, rec) -> None:
        self.records.append(rec)
    def by_component(self, name: str) -> list:
        return [r for r in self.records if r.component == name]
    def scopes(self) -> list:
        return [r.scope for r in self.records]


class JsonlSink:
    """Appends one JSON object per line (audit log / real-time tail)."""
    def __init__(self, path: str) -> None:
        self._f = open(path, "a", encoding="utf-8")
    def write(self, rec) -> None:
        self._f.write(rec.to_json() + "\n"); self._f.flush()
    def close(self) -> None:
        self._f.close()


class CallbackSink:
    """Forwards each record to a callback (real-time viewer hook)."""
    def __init__(self, fn: Callable) -> None:
        self._fn = fn
    def write(self, rec) -> None:
        self._fn(rec)


class FanoutSink:
    def __init__(self, *sinks) -> None:
        self._sinks = list(sinks)
    def write(self, rec) -> None:
        for s in self._sinks:
            s.write(rec)


# ── tracer ──────────────────────────────────────────────────────────────
class Tracer:
    def __init__(self, sink, level: Level = Level.AUDIT) -> None:
        self.sink = sink
        self.level = level_from(level)
        self._step = 0
        self._scope: list = []

    def set_step(self, i: int) -> None:
        self._step = i

    @contextmanager
    def scope(self, name: str):
        self._scope.append(name)
        try:
            yield self
        finally:
            self._scope.pop()

    def _path(self) -> str:
        return "/".join(self._scope)

    def probe(self, component, x, *, topology=None, unit_ids=None, meta=None) -> None:
        """Record an activation step. No-op below AUDIT."""
        if self.level <= Level.OFF:
            return
        rec = TraceRecord(self._step, self._path(), component, "activation",
                          _summarize(x), topology=topology, meta=meta or {})
        if self.level >= Level.VISUAL:
            rec.unit_ids, rec.intensities = _intensities(x, unit_ids)
        if self.level >= Level.FULL:
            rec.values = x
        self.sink.write(rec)

    def spikes(self, component, neuron_ids, *, topology=None, meta=None) -> None:
        """Record which neurons fired (SNN 'lights up'). The ids ARE the payload,
        so this emits from AUDIT up (cheap)."""
        if self.level <= Level.OFF:
            return
        ids = list(neuron_ids)
        rec = TraceRecord(self._step, self._path(), component, "activation",
                          {"n_spikes": len(ids)}, topology=topology,
                          unit_ids=ids, intensities=[1.0] * len(ids), meta=meta or {})
        self.sink.write(rec)

    def event(self, component, name, **meta) -> None:
        """Record a typed decision/event (routing, spawn, gate escalation)."""
        if self.level <= Level.OFF:
            return
        self.sink.write(TraceRecord(self._step, self._path(), component, "event",
                                    {"event": name}, meta=meta))


# ── active-tracer plumbing ──────────────────────────────────────────────
_NULL = Tracer(MemorySink(), Level.OFF)
_ACTIVE: ContextVar = ContextVar("cubby_tracer", default=_NULL)


def current() -> Tracer:
    return _ACTIVE.get()


@contextmanager
def trace_to(sink, level="audit"):
    """Activate a tracer for the enclosed block."""
    tok = _ACTIVE.set(Tracer(sink, level_from(level)))
    try:
        yield _ACTIVE.get()
    finally:
        _ACTIVE.reset(tok)


# ── module-level convenience (write to whatever tracer is active) ───────
def probe(component, x, **kw) -> None:
    current().probe(component, x, **kw)

def spikes(component, neuron_ids, **kw) -> None:
    current().spikes(component, neuron_ids, **kw)

def event(component, name, **meta) -> None:
    current().event(component, name, **meta)

def set_step(i: int) -> None:
    current().set_step(i)

def scope(name: str):
    return current().scope(name)


# ── test helpers ────────────────────────────────────────────────────────
def diff_report(golden: list, actual: list) -> list:
    """Per-step absmax divergence between two traces - shows which step broke."""
    out = []
    for g, a in zip(golden, actual):
        dm = abs(g.summary.get("absmax", 0.0) - a.summary.get("absmax", 0.0))
        out.append({"scope": a.scope, "component": a.component, "absmax_diff": dm})
    return out


def max_step_diff(golden: list, actual: list) -> float:
    rep = diff_report(golden, actual)
    return max((r["absmax_diff"] for r in rep), default=0.0)
