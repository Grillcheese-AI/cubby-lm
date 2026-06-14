"""0.0.0 trunk smoke + overfit gate.

Run: python -m cubby.trunk.test_model
Gate: forward shapes correct, EVERY param receives a finite grad, and the loss
drops sharply when overfitting a single tiny batch (learning works end to end).
"""
from __future__ import annotations

import numpy as np

import grilly  # noqa: F401
from cubby.config import make_config
from cubby.trunk.model import CubbyLM, AdamW, param_count


def _tiny_cfg(V=64, d=32, L=3):
    return make_config("0.0.0", vocab_size=V, d_model=d, n_layers=L, d_ffn=64, seq_len=16)


def test_forward_shape():
    cfg = _tiny_cfg()
    m = CubbyLM(cfg)
    ids = np.random.randint(0, cfg.total_vocab, size=(2, 8))
    logits = m(ids)
    assert np.asarray(logits.data).shape == (2, 8, cfg.total_vocab)


def test_every_param_gets_grad():
    cfg = _tiny_cfg()
    m = CubbyLM(cfg)
    ids = np.random.randint(0, cfg.total_vocab, size=(2, 8))
    tgt = np.random.randint(0, cfg.total_vocab, size=(2, 8))
    loss = m.loss(ids, tgt)
    loss.backward()
    params = list(m.parameters())
    missing = [i for i, p in enumerate(params) if getattr(p, "grad", None) is None]
    assert not missing, f"{len(missing)}/{len(params)} params got no grad: {missing}"
    for p in params:
        assert np.isfinite(np.asarray(p.grad)).all()


def test_overfits_single_batch():
    np.random.seed(0)
    cfg = _tiny_cfg(V=32, d=48, L=3)
    m = CubbyLM(cfg)
    opt = AdamW(m.parameters(), lr=3e-3)
    ids = np.random.randint(0, cfg.total_vocab, size=(2, 12))
    tgt = np.random.randint(0, cfg.total_vocab, size=(2, 12))
    losses = []
    for _ in range(60):
        loss = m.loss(ids, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.data))
    print(f"  param_count={param_count(m)}  loss {losses[0]:.3f} -> {losses[-1]:.3f}")
    assert losses[-1] < losses[0] * 0.4, f"did not overfit: {losses[0]:.3f} -> {losses[-1]:.3f}"


if __name__ == "__main__":
    for name, fn in sorted((k, v) for k, v in globals().items() if k.startswith("test_")):
        fn(); print(f"  ok  {name}")
    print("\npassed")
