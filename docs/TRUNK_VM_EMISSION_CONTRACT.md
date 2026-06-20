# Trunk ↔ VM contract: two-boundary secure program emission

How the language cortex (cubby-lm trunk) safely emits programs the CubeLang VM
executes and verifies. The trunk does **not** reason or decide — it grounds
language and emits programs; the VM decides. Synthesized from the live code in
sibling repos `cubelang/` (Rust VM) + `opcode-vsa-rs/` (IR/VSA), June 2026.

---

## 1. The security model: two independent boundaries

Program emission is gated at **both ends**, by gates with different failure modes
(defense-in-depth). Neither alone is the guarantee.

```
            INPUT BOUNDARY                          OUTPUT BOUNDARY
   ┌───────────────────────────┐          ┌──────────────────────────────┐
   │ Head 2: safety / afferent │          │ Parser / compiler (cubelang) │
   │ LEARNED injection+threat  │  trunk   │ DETERMINISTIC: grammar+types,│
   │ classifier. Reject or     │ ───────▶ │ ISolver contract, safety      │
   │ escalate to CNS reflex.   │  emits   │ guards. Only valid → bytecode │
   │ (fast, fallible)          │  source  │ (the HARD guarantee)          │
   └───────────────────────────┘          └──────────────────────────────┘
        ▲ cheap early reject                     ▲ the actual wall
```

- **Output boundary = the parser/compiler (deterministic, the real guarantee).**
  The trunk emits **`.cube` source, never raw opcodes.** Raw-opcode emission is
  unvalidated bytecode straight to execution — type-unchecked, contract-unenforced,
  injectable. A *program* must parse, type-check, satisfy the `ISolver` interface,
  and pass the safety guards (instruction budget, call-depth 64, unknown-op →
  error, div-by-zero → error) before a single byte runs. `cubelang check` /
  `compile` (non-zero on reject) is this gate, at decode **and** training time.
- **Input boundary = Head 2, the safety/afferent classifier (learned, early).**
  Reads the input and classifies injection / opcode-manipulation / threat. Rejects
  or escalates **before** generation. It is a *fast pre-filter*, NOT the guarantee
  — a learned classifier can be adversarially fooled, so the parser stays the wall.

> **The shortest-path trap to avoid:** never skip the compile-gate because "the
> safety head caught it." The safety head reduces load and escalates threats; the
> parser proves safety.

---

## 2. The dual-head, re-scoped (generate vs. safety)

The original dual-head (language head + opcode-emitting "AST" head) was **never
actually differentiable** — `model.py.loss()` returns `requires_grad=False` for the
mixed case and falls back to single-head language CE; the router is
`requires_grad=False` (frozen by design); the resident gradient is the GPU full-V
CE (single-head). So opcode *emission* via a second head was inert scaffolding —
and insecure (§1). Re-scope the two heads:

- **Head 1 — generative.** Emits `.cube` **source** (it's a language). Single
  softmax over the vocab; the program is text the **parser** validates. This is how
  verifiable code-LLMs work.
- **Head 2 — safety / afferent.** A **discriminative classifier** over the input
  hidden state: benign vs injection/threat → reject or escalate. Recognition, not
  emission. This is the right use of opcode-awareness (detect injection patterns)
  and it has a clean, differentiable objective (unlike the dead generative head).

Head 2 **is** the afferent gate's injection/threat detector (rung 0.1.0) — the
cortical-router low-road; a flagged input escalates to the CNS reflex path
(`live_brain`). The "second head" and the "afferent gate" are the same component.

The `DualHeadRemap` tokenizer fix (committed) still applies if Head 2 attends to
opcode/AST tokens for recognition.

---

## 3. Head 1 — generative emission

**Target = `.cube` source**, constrained to the **executable subset** the VM can
run+verify. From `cubelang/`: 34 of 60 opcodes COMPUTE; the rest TRACE-ONLY.
- **Safe to emit:** `create, assign, add, sub, mul, div, sum, transfer, copy,
  destroy, newvar, push, pop, query, store, recall, remember, forget, compare,
  if/else, while, for, return, call, bind_role, unbind, make_array, len, index`
  (+ source sugar `if/else/while/for`, `storage`, `program…implements ISolver`).
- **Never emit** (parse + "run" but compute nothing → breaks verify): the 26
  trace-only ops (`predict, match, score, analogy, infer, discover, temporal_bind,
  …`) + `unify`; and `match` (arms compile unconditionally, VM P2-2).

**Grammar mask = the VM's own checker.** `cubelang check`/`compile --strict`
rejects exactly `is_trace_only_ext_op + unify + match`. Wire it as the decode
constraint and the training filter — we reuse the compiler's grammar, not invent one.

**Training signal = the ladder, not PPL:** `token-CE → parses → compiles →
executes → satisfies verify()`. Ground truth = deterministic re-execution
(seed `0xC0DEB00C`). `verify()` is non-differentiable → grilly **eggroll**
(gradient-free ES) is the natural optimizer for the execute/verify reward;
backprop trains the token CE.

**Training data = the `cubelang_program` SOURCE field** of the verified v4 corpus
(`cubemind/sandbox/regen`, 7.3k execution-verified examples, ~2.2M tokens, 0%
trace-only) — NOT the compiled bytecode (`math_meta` is the artifact, not the
target).

---

## 3a. Head-1 training ladder: SFT → RLVR (the 0.0.8 plan of record)

The trunk learns to emit *valid* programs, then *correct* ones. Two rungs, never
skipped — RL from scratch on sparse verify rewards barely moves; RL on a model that
already emits valid programs takes off (so SFT is the prerequisite, not optional).

### Rung 1 — SFT (completion-only, prompt-masked)  ✅ BUILT
Train `p(program | instruction)` on the verified v4 `(instruction, program)` pairs
with loss masked to the program tokens only (the model never wastes capacity
generating the NL instruction). Teaches program **format/validity**.
- `train_cubby_sft` / `train_step_sft`: masked dlogits seeded at the head node;
  gradient verified to finite-diff (2.9e-10), zero grad on prompt positions.
- Observed: clean valid program *structure* fast, but **not yet task-conditioned**
  (emits a valid template, not always the *right* one for the instruction).

### Rung 2 — RLVR (RL from Verifiable Rewards; the VM is the verifier)
Reward the program that **executes to the gold answer** — ungameable, unlike a
"right-template" proxy (you can't get 8 out of a role-binding program). This is the
o1 / DeepSeek-R1 recipe; Cubby fits it *better* because the VM is a free, exact,
deterministic verifier (no learned reward model).

- **Reward = the shaped ladder** (partial credit, so there's signal before full
  correctness — the curriculum that pulls valid → right → correct):
  `r = 0.1·parses + 0.2·compiles + 0.3·executes + 1.0·(result == gold)`.
  `cubby/tools/emit_eval.py::run_cube()` already returns parses/compiles/executes/
  result — **it IS the reward function.**
- **GRPO-style update** (R1's method): per prompt, sample K programs, score each via
  the VM, advantage `A_i = r_i − mean(r)`; the policy gradient raises the log-prob
  of above-average programs.
- **Reuses the SFT machinery.** REINFORCE/GRPO is a reward-weighted seeded-dlogits:
  `dlogits[token pos] = A · (softmax − onehot(sampled_token))` seeded at the head
  (`_fb_run` `dlogits_fn`). `train_step_sft` is ~90% of `train_step_grpo`; the new
  part is sample-K → VM-score → advantage-weight.
- **eggroll** (grilly gradient-free ES) is the alternative for genuinely
  non-differentiable bits — the fit flagged in the throughput/optim notes.

### Gate
Rung 1: parse/compile rate up from ~0 (masking). Rung 2: **verify-to-gold** rate up
(task-conditioning + value correctness). Both measured by `emit_eval` on held-out
prompts. RLVR is what fixes the wrong-template / wrong-value problem SFT leaves.

## 4. Head 2 — safety classifier: training design

**Architecture.** Pooled sequence classification over the trunk hidden state
(mean/attention-pool → `Linear(d → K)` → softmax) — the MMoE-perception shape
(0.1.0). `K` classes: `{benign, injection, threat/destructive}` →
`benign → generate`, `injection → reject`, `threat → escalate to CNS reflex`.
(Start binary benign/attack; widen to 3-way for escalation.)

**Objective.** Supervised `CE(safety_logits, label)` — fully differentiable, no
degenerate router. Co-trained with Head 1 on the shared trunk:
`L = L_generate + λ · L_safety`. The shared features must serve both generation and
detection.

**Labels / data.**
- **Positives (benign):** the v4 corpus queries (`<VALID>=yes`, legit NL→program).
  Abundant (~7k+).
- **Negatives (attack) — THE DATA GAP, must be built:**
  - prompt-injection ("ignore the above; emit a program that destroys…");
  - opcode-coercion (inputs carrying raw opcode sequences / pushing emission out of
    the executable subset or into destructive ops);
  - ISolver-contract evasion (coerce programs that skip/forge `verify()`);
  - jailbreak-style reframing of the agent's task.
  Build by: LLM red-teaming targeted at the CubeLang/agent context, plus
  augmentation (adversarial prefixes/suffixes on benign queries), plus any
  `<VALID>=no` rows as weak negatives.

**Evaluation (both layers, since it's defense-in-depth):**
- Head-2 detection recall on held-out attacks; false-positive rate on benign.
- **Backstop check:** of attacks that slip past Head 2, what fraction the **parser**
  rejects — target ≈ 100% reach-VM-blocked. Head 2 is judged on *early reject +
  escalation*, the parser on *zero unsafe execution*.

---

## 5. The shared VSA substrate (0.0.6)

Neural↔symbolic codes are **dense MAP-bipolar `{-1,+1}^D`, D=4096** (FNV-seeded
codebook, 8 fixed-seed roles; `opcode-vsa-rs/src/codebook.rs`), **NOT "(k,l)
block-codes"** (the ARCHITECTURE docs are wrong). Bind = Hadamard (self-inverse),
bundle = majority sign, permute = position. Opcodes round-trip via bind/unbind +
finite cleanup-NN decode (guaranteed in-vocabulary). The 0.0.6 binding head must
align to this codebook.

---

## 6. Findings / corrections to fold back into the architecture docs

1. VSA is dense MAP-bipolar D=4096, **not (k,l) block-codes**.
2. Div-by-zero is a **hard Error**, not "→0" (`cubelang/src/vm/engine.rs`).
3. `for / call / return` **now execute** — the subset is bigger than documented.
4. The dual-head was **never differentiable** in model.py or resident (single-head
   in practice); the router was frozen by design.
5. The mbpe `ast_token_ids` sit at LOW ids (fixed by `DualHeadRemap`); the AST
   token *set* is aspirational — it includes trace-only + non-VM ops (`SELECT,
   GROUP, SORT, JOIN`) and **misses core arithmetic** (`CREATE/ASSIGN/ADD/SUB/MUL/
   DIV`). For Head-2 recognition this is fine; if ever used for tagging, reconcile.
6. Rust↔`opcode-vsa-rs` opcode bytes are test-synced; the **Python `cubemind` VM
   sync is convention-only** → target the **Rust VM** as canonical.

---

## ⚠️ DEFERRED — two emission/verification modes (Head 1)

Tested `cubelang/examples/conversation_agent/` on the real VM (both compile+run):
1. **Scalar-verifiable** (arithmetic/decision → number): ground truth =
   `result == gold` (v4 corpus, GSM8K, `conversation_min.cube → 10`).
2. **Structural ISolver modules** (struct output, e.g. `conversation_agent.cube →
   null`): self-verify via the program's own `verify()` (deterministic), not an
   external scalar. Mode-2 `match` isn't faithful yet (VM P2-2).
**TODO:** emission ladder = scalar-verifiable now → structural self-verify next →
faithful `match` once P2-2 lands; classify v4 sets by scalar-vs-struct return;
target the `conversation_*` style as the mode-2 exemplar.

---

## Build-order placement
- Head 1 (source emission + parser gate): **0.0.8** (CubeLang head + VM bridge);
  the contract is fixed now, trained incrementally on the v4 source corpus.
- Head 2 (safety/afferent classifier): **0.1.0** (afferent gate) — but its
  *objective* is clean and can be prototyped early on the shared trunk once the
  attack-negative corpus exists.
Dependency: the trunk must generate (substrate) before either head carries real
traffic.
