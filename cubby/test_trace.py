"""Tests for cubby.trace - the per-step audit/visualization bus.

Doubles as the reference for how every component should emit traces.
Run: python cubby/test_trace.py   (or pytest cubby/test_trace.py)
"""
from __future__ import annotations

import os
import numpy as np

from cubby import trace as T


def test_off_is_noop():
    sink = T.MemorySink()
    with T.trace_to(sink, "off"):
        T.probe("mingru", np.ones((2, 4)))
        T.event("router", "route", expert=1)
    assert sink.records == []


def test_audit_has_summary_no_payload():
    sink = T.MemorySink()
    with T.trace_to(sink, "audit"):
        T.probe("mingru", np.array([[1.0, -3.0, 0.0, 2.0]]))
    (rec,) = sink.records
    assert rec.kind == "activation" and rec.component == "mingru"
    assert rec.summary["absmax"] == 3.0
    assert abs(rec.summary["nonzero_frac"] - 0.75) < 1e-9
    assert rec.intensities is None and rec.values is None   # cheap level


def test_visual_lights_up_units():
    sink = T.MemorySink()
    with T.trace_to(sink, "visual"):
        T.probe("layer", np.array([[0.0, 4.0, 2.0]]), topology="layer:0")
    (rec,) = sink.records
    assert rec.unit_ids == [0, 1, 2]
    assert rec.intensities == [0.0, 1.0, 0.5]               # normalized 0..1
    assert rec.values is None                                # not FULL


def test_full_keeps_tensor():
    sink = T.MemorySink()
    x = np.random.randn(3, 5)
    with T.trace_to(sink, "full"):
        T.probe("head", x)
    assert np.allclose(sink.records[0].values, x)


def test_spikes_payload_is_ids():
    sink = T.MemorySink()
    with T.trace_to(sink, "audit"):
        T.spikes("snn", [3, 17, 42], topology="grid:8x8")
    (rec,) = sink.records
    assert rec.unit_ids == [3, 17, 42] and rec.summary["n_spikes"] == 3


def test_scope_and_step_nest():
    sink = T.MemorySink()
    with T.trace_to(sink, "audit") as tr:
        tr.set_step(7)
        with T.scope("trunk"), T.scope("layer3"):
            T.probe("mingru", np.zeros((1, 2)))
    rec = sink.records[0]
    assert rec.scope == "trunk/layer3" and rec.step == 7


def test_jsonl_sink_roundtrips(tmp_path="."):
    import json
    path = os.path.join(tmp_path, "_trace_smoke.jsonl")
    sink = T.JsonlSink(path)
    with T.trace_to(sink, "visual"):
        T.probe("mingru", np.array([[1.0, 2.0]]))
    sink.close()
    line = open(path).read().splitlines()[0]
    obj = json.loads(line)
    assert obj["component"] == "mingru" and obj["intensities"] == [0.5, 1.0]
    os.remove(path)


def test_diff_report_finds_divergent_step():
    a = T.MemorySink(); b = T.MemorySink()
    with T.trace_to(a, "audit"):
        T.probe("l0", np.array([1.0])); T.probe("l1", np.array([2.0]))
    with T.trace_to(b, "audit"):
        T.probe("l0", np.array([1.0])); T.probe("l1", np.array([2.5]))
    assert abs(T.max_step_diff(a.records, b.records) - 0.5) < 1e-9


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
