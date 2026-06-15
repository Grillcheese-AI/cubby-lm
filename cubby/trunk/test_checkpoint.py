"""Crash-guardrail checkpoint tests: save/restore roundtrip (weights survive the
resident<->model layout round-trip), RNG continuity, and the skip circuit-breaker.

    python cubby/trunk/test_checkpoint.py     # stdlib runner (no pytest)
    pytest cubby/trunk/test_checkpoint.py
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, r"C:\Users\grill\Documents\GitHub\cubby-lm")

from cubby.trunk import checkpoint as ckpt


def test_skip_guard():
    """Isolated skips don't trip; K-in-a-row aborts; a good step resets the run."""
    g = ckpt.SkipGuard(max_consecutive=3)
    assert g.update(True) is False        # 1
    assert g.update(True) is False        # 2
    assert g.update(False) is False       # reset
    assert g.update(True) is False        # 1
    assert g.update(True) is False        # 2
    assert g.update(True) is True         # 3 consecutive -> abort
    assert g.total == 5                   # 5 skips total (the False doesn't count)


def test_rng_roundtrip():
    """A restored Generator continues the SAME data stream (no batch replay)."""
    rng = np.random.default_rng(123)
    rng.integers(0, 1000, 5)                          # advance
    meta = {"rng": rng.bit_generator.state}
    restored = ckpt.restore_rng(meta)
    ref = np.random.default_rng(123); ref.integers(0, 1000, 5)
    assert np.array_equal(restored.integers(0, 1000, 7), ref.integers(0, 1000, 7))


def test_checkpoint_roundtrip():
    """Train a trunk, checkpoint it, restore into a freshly-built trunk, and assert
    the restored trunk reproduces the trained trunk's logits exactly -- proving the
    gvd un-split + gate_up un-swap round-trip is correct -- while a fresh-init trunk
    does NOT match (so the equality isn't trivial)."""
    from cubby.config import make_config
    from cubby.trunk.model import CubbyLM
    from cubby.trunk.resident import ResidentTrunk, make_device, force_numpy_reference

    force_numpy_reference()
    dev = make_device()
    np.random.seed(0)
    cfg = make_config("0.0.0", vocab_size=512, d_model=128, n_layers=4, d_ffn=256)
    ids = np.random.randint(0, cfg.total_vocab, (2, 8)).astype(np.int64)
    tgt = np.random.randint(0, cfg.total_vocab, (2, 8)).astype(np.int64)

    # A: train two steps so the weights differ from any fresh init
    np.random.seed(1); mA = CubbyLM(cfg); A = ResidentTrunk(mA, dev)
    for s in (1, 2):
        A.train_step(ids, tgt, s, lr=3e-3)
    logitsA = A.logits(ids)

    path = os.path.join(tempfile.gettempdir(), "cubby_ckpt_roundtrip.grl")
    rng = np.random.default_rng(123); rng.integers(0, 1000, 5)
    ckpt.save_checkpoint(path, A, step=2, rng=rng, version="0.0.0",
                         lr=3e-3, warmup=0, max_grad_norm=1.0, best_ppl=42.0)

    model_state, meta = ckpt.load_checkpoint(path)
    assert meta["step"] == 2 and meta["best_ppl"] == 42.0
    assert ckpt.checkpoint_matches(meta, cfg)
    assert not ckpt.checkpoint_matches(meta, make_config("0.0.0", d_model=128, n_layers=6, d_ffn=256, vocab_size=512))

    # B: fresh model (different init), restore, fresh trunk -> must reproduce A
    np.random.seed(999); mB = CubbyLM(cfg)
    ckpt.apply_model_state(mB, model_state)
    B = ResidentTrunk(mB, dev)
    diff = float(np.abs(logitsA - B.logits(ids)).max())
    assert diff < 1e-3, "restored logits differ by %.3e" % diff

    # C: fresh-init trunk should NOT match (equality above is meaningful)
    np.random.seed(777); mC = CubbyLM(cfg); C = ResidentTrunk(mC, dev)
    diffC = float(np.abs(logitsA - C.logits(ids)).max())
    assert diffC > 1e-2, "fresh-init trunk unexpectedly matched (%.3e)" % diffC

    os.remove(path)


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = []
    print("=== checkpoint crash-guardrail tests ===")
    for name, fn in fns:
        sys.stdout.write("RUN   %s ... " % name); sys.stdout.flush()
        try:
            fn(); print("PASS")
        except Exception as e:
            failed.append(name); print("FAIL\n      %s" % str(e).replace("\n", "\n      "))
    print("\n%d/%d passed" % (len(fns) - len(failed), len(fns)))
    os._exit(1 if failed else 0)
