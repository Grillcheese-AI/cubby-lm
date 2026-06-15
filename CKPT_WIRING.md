# Crash-guardrail wiring — `train_cubby_resident` (cubby/trunk/resident.py)

The checkpoint primitive is **done & committed** (`60318ff`): `cubby/trunk/checkpoint.py`
(+ `test_checkpoint.py`, 3/3 green). This is the last step: wire it into the resident
train loop. Lives in `resident.py`, which currently has uncommitted edits — apply this
once that file is committed so nothing gets clobbered.

API it uses (already committed):
- `checkpoint.load_checkpoint(path) -> (model_state, meta)`
- `checkpoint.checkpoint_matches(meta, cfg) -> bool`
- `checkpoint.apply_model_state(model, model_state)`  # mutates CubbyLM Variables in place
- `checkpoint.restore_rng(meta) -> np.random.Generator`
- `checkpoint.save_checkpoint(path, rt, *, step, rng, version, lr, warmup, max_grad_norm, best_ppl=None)`
- `checkpoint.SkipGuard(max_consecutive=K).update(skipped) -> bool abort`

---

## 1. Signature — add four params

```python
def train_cubby_resident(version="0.0.0", steps=600, data="tinystory_50k.json",
                         B=8, S=64, lr=3e-4, max_tokens=4000000, sample_every=200,
                         prompt="Once upon a time", gen_tokens=60, dev=None,
                         warmup=0, max_grad_norm=1.0,
                         ckpt_path=None, ckpt_every=100, resume=True, max_consec_skips=10):
```

## 2. Resume — REPLACE the two model/rt construction lines

`apply_model_state` mutates the CubbyLM's Variables, and `ResidentTrunk.__init__`
copies those into resident buffers — so the restore MUST happen BEFORE building rt.

REPLACE:
```python
    model = CubbyLM(cfg)
    rt = ResidentTrunk(model, dev or make_device())
```
WITH:
```python
    import os
    from cubby.trunk import checkpoint as _ckpt
    if ckpt_path is None:
        ckpt_path = os.path.join(_CUBBY_ROOT, "ckpt_%s.grl" % version)

    model = CubbyLM(cfg)
    start_step = 0
    if resume and os.path.exists(ckpt_path):
        model_state, meta = _ckpt.load_checkpoint(ckpt_path)
        if _ckpt.checkpoint_matches(meta, cfg):
            _ckpt.apply_model_state(model, model_state)    # restore BEFORE ResidentTrunk()
            start_step = int(meta.get("step", 0))
            rng = _ckpt.restore_rng(meta)                  # continue the SAME data stream
            print("[resume] %s @ step %d" % (ckpt_path, start_step), flush=True)
        else:
            print("[resume] shape mismatch, ignoring %s" % ckpt_path, flush=True)
    rt = ResidentTrunk(model, dev or make_device())
```
Note: `rng` is the same local the `batch()` closure reads, so reassigning it here
(after `def batch()`) correctly redirects the sampler. `start_step` drives the loop.

## 3. Guardrailed loop — REPLACE the loop + tail

REPLACE:
```python
    t0 = _time.perf_counter(); nskip = 0
    for step in range(1, steps + 1):
        ids, tgt = batch()
        lr_t = lr * min(1.0, step / warmup) if warmup else lr
        loss, gnorm, skipped = rt.train_step(ids, tgt, step, lr=lr_t, max_grad_norm=max_grad_norm)
        nskip += int(skipped)
        if step == 1 or step % 1 == 0 or step == steps:
            print("[%4d/%d] ce=%.3f ppl=%.1f gnorm=%.2e lr=%.1e (%.2f it/s)%s"
                  % (step, steps, loss, np.exp(loss), gnorm, lr_t,
                     step / (_time.perf_counter() - t0), "  [skipped]" if skipped else ""), flush=True)
        if sample_every and step % sample_every == 0:
            print("  sample:", repr(sample()), flush=True)
    if nskip:
        print("[warn] %d step(s) skipped (non-finite grad)" % nskip, flush=True)
    print("[done] %.1fs  final sample: %r" % (_time.perf_counter() - t0, sample()), flush=True)
    return rt, tok
```
WITH:
```python
    guard = _ckpt.SkipGuard(max_consecutive=max_consec_skips)
    step = start_step
    def _save(tag):
        try:
            _ckpt.save_checkpoint(ckpt_path, rt, step=step, rng=rng, version=version,
                                  lr=lr, warmup=warmup, max_grad_norm=max_grad_norm)
            print("[ckpt] %s @ step %d -> %s" % (tag, step, ckpt_path), flush=True)
        except Exception as e:
            print("[ckpt] save FAILED (%s): %r" % (tag, e), flush=True)

    t0 = _time.perf_counter(); nskip = 0
    try:
        for step in range(start_step + 1, steps + 1):
            ids, tgt = batch()
            lr_t = lr * min(1.0, step / warmup) if warmup else lr
            loss, gnorm, skipped = rt.train_step(ids, tgt, step, lr=lr_t, max_grad_norm=max_grad_norm)
            nskip += int(skipped)
            print("[%4d/%d] ce=%.3f ppl=%.1f gnorm=%.2e lr=%.1e (%.2f it/s)%s"
                  % (step, steps, loss, np.exp(loss), gnorm, lr_t,
                     step / (_time.perf_counter() - t0), "  [skipped]" if skipped else ""), flush=True)
            if guard.update(skipped):                       # K-in-a-row -> divergence
                print("[abort] %d consecutive non-finite grads" % guard.max_consecutive, flush=True)
                _save("diverge"); break
            if sample_every and step % sample_every == 0:
                print("  sample:", repr(sample()), flush=True)
            if ckpt_every and step % ckpt_every == 0:
                _save("periodic")
    except KeyboardInterrupt:
        print("[interrupt] flushing checkpoint", flush=True); _save("interrupt"); raise
    except Exception as e:                                   # OOM / any in-step failure
        print("[error] step %d: %r -> emergency checkpoint" % (step, e), flush=True)
        _save("emergency"); raise
    else:
        _save("final")
    if nskip:
        print("[warn] %d step(s) skipped (non-finite grad)" % nskip, flush=True)
    print("[done] %.1fs  final sample: %r" % (_time.perf_counter() - t0, sample()), flush=True)
    return rt, tok
```

## 4. (optional) main.py CLI flags

Defaults make it work with no CLI change (`resume=True`, auto path `ckpt_<version>.grl`,
save every 100). To expose: add `--ckpt PATH`, `--ckpt-every N`, `--no-resume`,
`--max-consec-skips K`, forwarding to `train_cubby_resident(...)`.

---

## Notes / honest caveats
- **OOM exception type not yet pinned.** The `except Exception` catches OOM (and anything
  else), flushes best-effort, and **re-raises** so the real error still surfaces. Once we
  confirm the exact type the C++ alloc path raises (likely `RuntimeError` with a VkResult
  message), we can add a targeted "out of VRAM, lower B/S" hint. Broad-catch-then-reraise
  is the correct guardrail behaviour regardless.
- **Emergency save may itself fail** if the GPU is wedged post-OOM (read_buffer needs the
  device) — `_save`'s own try/except handles that and prints rather than masking the cause.
  The **periodic** checkpoint is the real protection against a hard crash/power loss/freeze
  (an except block never runs on a hard crash); tune `ckpt_every` to your step cost.
- **Adam moments restart from zero on resume** (weights-only restore — see checkpoint.py
  docstring). Mild self-correcting transient under clip+warmup. Exact moment-restore is a
  later 1-line `resume_state` hook in resident.py once it's calm.
- `best_ppl` is plumbed through `save_checkpoint` but not tracked by the loop yet; add a
  running-min if you want "keep best" alongside "keep latest".
