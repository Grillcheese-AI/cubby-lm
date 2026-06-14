# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in the **cubby-lm** repo.

## What this is

A clean rebuild of **Cubby** ? a recurrence-first neuro-symbolic language model ?
after the v4 decode collapse. v4 (516M: MinGRU + sparse windowed attention + MoE
+ MTP + a fixed-codebook VSA output head) reached val PPL ~7 but generated
degenerate text under both greedy and sampling. The 5.6M MinGRU baseline with a
**weight-tied linear head** free-runs coherent English, so the recurrence is
fine ? the collapse came from stacking components at once (VSA binding head the
prime suspect). This repo rebuilds the disciplined way: **start from the
known-good baseline and add one component at a time at matched budget, scoring
generation quality (not just perplexity) at every rung.**

Cubby is the **language cortex** of a larger neuro-symbolic agent: it interprets
and generates language and emits programs, but the agent's *decisions* are meant
to run in a symbolic CubeLang VM over grounded world models, not in LM
activations. The full "why" is `README.md`; the architecture spec is
`docs/why_sparse_cubby.md` and `docs/ARCHITECTURE.md`.

> **Embargo.** The I-RAVEN-X / NeurIPS 2026 reasoning numbers in the README and
> `docs/papers/` are under submission embargo ? do not circulate them outside the
> repo.

## The core discipline (read before adding anything)

- **One validated component per version rung.** The ladder lives in
  `cubby/ROADMAP.md`. `0.0.0` = MinGRU + tied-linear + SwiGLU substrate; each
  later rung flips exactly the flags it adds.
- **Components are flag-gated and OFF at v0.** An unbuilt component **raises**
  rather than silently no-op'ing (`config.py` is the single source of truth;
  `main.py` subcommands raise with a pointer to ROADMAP for unbuilt paths).
- **Gate on generation, not PPL.** Any version that drops generation quality
  (rep-n / distinct-n / looping-rate / coherence) while PPL stays flat is
  quarantined and understood before the next rung.
- **Standing invariant ? every step is auditable + visualizable:** each new
  component must (1) emit per-step records to `cubby.trace`, (2) render in the
  viewer, and (3) be asserted in tests (golden trace + first-divergent-step
  diff). Not done until all three hold.
- **Port discipline:** implement each trunk op on grilly, validate **numerical
  parity vs the torch reference in `_reference_torch/`** (max_abs_diff), *then*
  bring up training (forward + backward + AdamW on grilly), *then* gate on
  coherent generation. Only then does the next component go on.

## Layout

- `main.py` ? the only root module; CLI entry. Subcommands dispatch into `cubby/`:
  `info` (resolved config), `smoke` (plumbing), `parity` (grilly vs torch ref),
  `train`, `gen`. Unbuilt ones raise.
- `cubby/` ? the package.
  - `config.py` ? **single source of truth** for the architecture. `SparseCubbyConfig`
    dataclass + version presets (`make_config`, `VERSIONS`, `DEFAULT_VERSION`).
  - `trace.py` / `test_trace.py` ? the per-step audit/visualization bus (the
    observability invariant above).
  - `tokenizer.py` ? BBPE-65k v3 (`grillcheese_bbpe65k_v3`) wiring.
  - `trunk/` ? the model itself: `model.py` (the trunk), `ffn.py` (SwiGLU +
    TernarySwiGLU/BitNet QAT on tape autograd), `gpu_linear.py` (GPU linear
    bridge), `train.py`, plus `test_ffn.py` / `test_model.py` / `test_parity.py`.
- `brain/` ? `live_brain` event-driven SNN (`event_snn.py`). A **parallel track**,
  not part of the 0.0.x trunk: it is the CNS reflex path the afferent gate (0.1.0)
  escalates to, and later a vision cortex.
- `embedding/`, `experiments/`, `external/` ? supporting/experimental code.
- `_reference_torch/` ? torch parity reference ONLY. torch must not appear in the
  trunk path; it lives here as the max_abs_diff oracle.
- `docs/` ? `why_sparse_cubby.md`, `ARCHITECTURE.md`, `papers/` (embargoed).
- `tinystory_50k.json` ? TinyStories training corpus; `gen1.txt` ? sample output.

## grilly is the backend (sibling repo, no CUDA, no torch in the trunk)

The trunk runs and trains on **grilly**, the Vulkan-compute framework in the
sibling repo `../grilly`. `pyproject.toml` pins grilly as an **editable uv path
source** (`tool.uv.sources.grilly = { path = "../grilly", editable = true }`), so
`import grilly` resolves to the repo and the compiled extension
`grilly_core.<abi>.pyd` is auto-loaded ? **rebuilding grilly via
`../grilly/rebuild.ps1` is picked up automatically, no reinstall.**

The bridge surface:
- `from grilly.nn.autograd import Variable, GradFn` ? the tape autograd the
  trunk/FFN build on.
- `from grilly.backend import _bridge` ? `_bridge.linear` / `_bridge.linear_backward`
  (GPU linear), `_bridge.is_available()`. See `cubby/trunk/gpu_linear.py`.

**When the trunk needs a new/faster GPU op, it is implemented in grilly** (C++
op + GLSL shader + binding, or the resident autograd path), not in this repo.
See `../grilly/CLAUDE.md` and `../grilly/AUTOGRAD_STATE.md`.

## Commands

```powershell
# venv (grilly editable-installed here; this interpreter loads grilly_core.pyd):
$py = "C:\Users\grill\Documents\GitHub\cubby-lm\.venv\Scripts\python.exe"

& $py main.py info   --version 0.0.0     # resolved config + which components are on
& $py main.py smoke  --version 0.0.0     # forward/backward/generate plumbing
& $py main.py parity --version 0.0.0     # grilly trunk vs torch reference (max_abs_diff)
& $py main.py train  --version 0.0.0 --steps 4000 --data tinystory_50k.json
& $py main.py gen    --version 0.0.0 --prompt "Once upon a time, "

# tests are colocated pytest files (not a tests/ dir):
& $py -m pytest cubby/ brain/ -v
```

## Status / what's in flight

From `cubby/ROADMAP.md` (check it for the live list):
- **0.0.0 DONE** ? trunk forward+backward on grilly (every param grads; overfits
  3.49?0.035), forward parity vs reference (max_abs_diff ~2.3e-7), TinyStories
  trains (byte 3.49?1.23) and generates English; BBPE-65k wired; SwiGLU vs
  TernarySwiGLU quality parity.
- **0.0.1 IN PROGRESS** ? GPU linear bridge green (parity 2.8e-7) and training
  on GPU beyond CPU's reach, but **transfer/dispatch-bound** (~0.6 it/s). Perf
  diagnosis: the cost is **per-dispatch overhead (~25 ms/call)**, not
  compute/transfer. Done: fused projections (g/v/d 3?1, gate/up 2?1), GPU CE +
  in-place AdamW. **Open (the big one): perf #2 ? resident activations** (keep
  activations in VRAM across the forward to beat the ~25 ms/dispatch floor). That
  is exactly what the grilly **`autograd-resident-backward`** branch is building;
  this is the gating item before the full v3.3-shape (d=1024, L=18, V=65k) run.

## Conventions

- Python ? 3.12, managed with **uv** (`uv.lock` present).
- Trunk data is `np.float32`; dtype switchable to bf16 via config.
- Don't introduce torch into the trunk path ? parity reference only.
- Respect the gating order: parity ? train ? generation-gate ? next component.
