"""Emission eval harness — the metric that gates secure CubeLang emission.

Scores generated `.cube` SOURCE on the ladder the contract demands (NOT PPL):
    parses -> compiles -> executes -> (verifies)
by shelling out to the real VM (`cubelang.exe`), which IS the security gate. The
trunk emits source; this measures how often that source is valid+runnable.

Usage:
    # validate the harness on ground-truth verified programs (should be ~100%):
    python -m cubby.tools.emit_eval --selfcheck

    # score a checkpoint's generations (after training Head 1):
    python -m cubby.tools.emit_eval --ckpt ckpt_mbpe_emit.grl --version mbpe_emit \
        --tokenizer mbpe32k --n 50
"""
from __future__ import annotations
import json, os, re, subprocess, tempfile
from pathlib import Path

CUBELANG_EXE = r"C:\Users\grill\Documents\GitHub\cubelang\cubelang.exe"
V4_DIR = Path(r"C:\Users\grill\Documents\GitHub\cubemind\sandbox\regen")
_RESULT_RE = re.compile(r"->\s*(-?\d+(?:\.\d+)?|null)\s*$", re.MULTILINE)


def run_cube(source: str, timeout: float = 15.0) -> dict:
    """Compile+run one CubeLang SOURCE string via the real VM. Returns the ladder:
    {parses, compiles, executes, result, error}. `check` then `run` (compile is
    implied by run; we separate parse from execute via check)."""
    d = tempfile.mkdtemp(prefix="emit_")
    fp = Path(d) / "p.cube"
    out = {"parses": False, "compiles": False, "executes": False, "result": None, "error": ""}
    try:
        fp.write_text(source, encoding="utf-8")

        def _run(cmd):
            try:
                r = subprocess.run([CUBELANG_EXE, cmd, str(fp)],
                                   capture_output=True, text=True, timeout=timeout)
                return r.returncode, (r.stdout or "") + (r.stderr or "")
            except subprocess.TimeoutExpired:
                return 124, "<timeout>"

        rc, o = _run("check")                       # parse + typecheck (the gate)
        out["parses"] = (rc == 0)
        if not out["parses"]:
            out["error"] = o.strip()[:200]
            return out
        rc, o = _run("compile")                     # to bytecode
        out["compiles"] = (rc == 0)
        if not out["compiles"]:
            out["error"] = o.strip()[:200]
            return out
        rc, o = _run("run")                         # execute
        out["executes"] = (rc == 0)
        m = _RESULT_RE.search(o)
        if m:
            s = m.group(1)
            out["result"] = None if s == "null" else (float(s) if "." in s else int(s))
        if not out["executes"]:
            out["error"] = o.strip()[:200]
    finally:
        try:
            fp.unlink(); os.rmdir(d)
        except OSError:
            pass
    return out


def score(sources, gold=None, label=""):
    """Score a list of source strings on the ladder. gold (optional) = expected
    scalar results for the verify-against-answer rate."""
    n = len(sources)
    agg = {"parses": 0, "compiles": 0, "executes": 0, "verifies": 0}
    fails = []
    for i, src in enumerate(sources):
        r = run_cube(src)
        for k in ("parses", "compiles", "executes"):
            agg[k] += int(r[k])
        if gold is not None and r["executes"] and r["result"] is not None and gold[i] is not None:
            agg["verifies"] += int(r["result"] == gold[i])
        if not r["executes"] and len(fails) < 5:
            fails.append((i, r["error"]))
    print(f"=== emission ladder {label} (n={n}) ===")
    for k in ("parses", "compiles", "executes"):
        print(f"  {k:9s}: {agg[k]:4d}/{n}  ({100*agg[k]/max(n,1):.1f}%)")
    if gold is not None:
        ne = agg["executes"]
        print(f"  verifies : {agg['verifies']:4d}/{n}  ({100*agg['verifies']/max(n,1):.1f}% of all)")
    if fails:
        print("  first failures:")
        for i, e in fails:
            print(f"    [{i}] {e[:120]}")
    return agg


def selfcheck(n=60):
    """Sanity: run ground-truth verified programs through the harness -> ~100%."""
    rows = []
    for fn in ("multitask_v4_arith.jsonl", "multitask_v4_atier.jsonl", "multitask_v4.jsonl"):
        p = V4_DIR / fn
        if p.exists():
            for ln in p.read_text(encoding="utf-8").splitlines()[: n // 3 + 1]:
                rows.append(json.loads(ln))
    rows = rows[:n]
    srcs = [r["cubelang_program"] for r in rows]
    score(srcs, label="(v4 ground-truth)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--version", default="mbpe_emit")
    ap.add_argument("--tokenizer", default="mbpe32k")
    ap.add_argument("--n", type=int, default=50)
    a = ap.parse_args()
    if a.selfcheck or not a.ckpt:
        selfcheck(a.n)
    else:
        # generate from held-out v4 prompts, extract the program, score
        from cubby.trunk.resident import generate_from_checkpoint
        rows = [json.loads(l) for l in (V4_DIR / "multitask_v4.jsonl").read_text(encoding="utf-8").splitlines()]
        rows = rows[-a.n:]
        srcs = []
        for r in rows:
            prompt = r.get("prompt") or r.get("question") or r.get("text") or ""
            gen = generate_from_checkpoint(version=a.version, tokenizer=a.tokenizer,
                                           ckpt_path=a.ckpt, prompt=f"[INSTRUCTION]\n{prompt}\n[/INSTRUCTION]\n",
                                           max_new_tokens=256, temperature=0.0,
                                           skip_special=False)   # render opcode/role AST tokens into source
            m = re.search(r"(program\s+\w+\s+implements[\s\S]+?\n\})", gen)
            srcs.append(m.group(1) if m else gen)
        score(srcs, label=f"(generated @ {os.path.basename(a.ckpt)})")
