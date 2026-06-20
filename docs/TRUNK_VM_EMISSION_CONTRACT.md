# Trunk → VM emission contract

How the language cortex (cubby-lm trunk) emits programs the CubeLang VM can
**execute and verify**. Synthesized from the live code in sibling repos
`cubelang/` (Rust VM) and `opcode-vsa-rs/` (IR + VSA), June 2026. The trunk does
not reason; it *grounds language and emits opcodes*, and the VM decides.

## The key realization: the AST head IS the emission head
The dual-head trunk already has the interface. The router gates each token between
the **language head** (V_lang) and the **AST head** (V_ast = 47 special tokens).
That AST head *is* the CubeLang emission head — and 47 tokens ≈ the budget needed
for **34 executable opcodes + 8 universal roles + ~5 structural tokens**. So the
architecture is already shaped for this; it just isn't trained for it yet
(the router is currently frozen / a no-op — see
`memory/resident-dualhead-sampled-noop`). Making emission real and fixing the dead
router are the **same** work: the router learns *when* to emit a program (switch
to AST mode), the grammar-masked AST head emits *what*.

## Layer 1 — the emission target (what the AST head produces)
An ordered sequence of instruction tuples `("OPCODE", arg0, arg1, …)`:
- **OPCODE** ∈ the **executable subset only** (34): `create, assign, add, sub,
  mul, div, sum, transfer, copy, destroy, newvar, push, pop, query, store, recall,
  remember, forget, compare, if/else (cond), while, for, return, call, bind_role,
  unbind, make_array, len, index` (+ `label/jmp` as structural).
- **operands** per opcode arity (`opcode-vsa-rs/src/ir.rs::register_operand_count`)
  and the EBNF `opcode_stmt` (`opcode-vsa-rs/docs/cubelang-spec.md:2729`): a leading
  **Named register**, then immediates / the **8 roles** (AGENT…STATE) / globals
  (for jmp/call/label) / a type (CREATE).
- Parsed by `opcode-vsa-rs/src/importer.rs` into `CubeMindInstr{opcode, operands}`;
  the whole sequence is `CubeMindProgram(Vec<CubeMindInstr>)` — directly executable.

## Layer 2 — the grammar mask (the hard fence, already exists)
- **Decode-time:** constrain the AST head to (executable opcode × that opcode's
  operand schema). Every emitted program is then executable *by construction* — no
  trace-only opcode can be produced.
- **The fence is real code:** `cubelang validate <file>` / `cubelang compile
  --strict`, keyed off `is_trace_only_ext_op + unify + match`. Wire it as both the
  decode constraint and the training filter — we don't invent a grammar, we reuse
  the VM's own.

## Layer 3 — the training signal (the ladder, not PPL)
`token-CE → parses → compiles → executes → satisfies verify()`.
- **Ground truth = deterministic re-execution** (fixed memory seed `0xC0DEB00C`).
  A host compiles+runs the emitted program and compares the computed `result()` to
  the known answer (the GSM8K→opcodes path in `opcode-vsa-rs/src/training/
  gsm_program.rs` is the seed corpus).
- Enter as **eval metrics first**, then optionally as a reward term. `verify()` is
  **non-differentiable** → this is exactly where grilly's **eggroll** (gradient-free
  Evolution Strategies) fits: optimize the emission policy against the
  compile/execute/verify reward without backprop through the VM.

## Layer 4 — the lockstep invariant
The trunk's allowed opcode vocabulary == the **current** executable subset. As
trace-only opcodes graduate to computing (`cubelang/CUBELANG_FIXES.md` P0-1 work),
grow the AST vocabulary in lockstep. **Never** let the trunk learn to emit a
trace-only opcode — it would produce valid-looking programs that compute nothing,
silently breaking the ground-truth guarantee.

## Layer 5 — the VSA bridge (0.0.6) uses the *real* code space
The shared neural↔symbolic space is **dense MAP-bipolar `{-1,+1}^D`, D=4096**, with
a deterministic FNV-seeded codebook and 8 fixed-seed role vectors
(`opcode-vsa-rs/src/codebook.rs`, `hypervec.rs`). Bind = Hadamard (self-inverse),
bundle = majority sign, permute = rotate (position). Opcodes round-trip via
bind/unbind + finite **cleanup-memory nearest-neighbor** decode — guaranteed
in-vocabulary (no hallucination). The 0.0.6 VSA head must align to **this**
codebook so opcodes round-trip; it is NOT a "(k,l) block-code" scheme.

## Open items / risks to close before this is trustworthy
1. **Stale architecture docs (correct them):**
   - The docs say "(k,l) block-codes"; the implementation is **dense MAP-bipolar
     D=4096**. Fix `docs/ARCHITECTURE*.md`, `why_sparse_cubby.md`.
   - Docs say "divide-by-zero → 0"; the VM returns a **hard Error**
     (`cubelang/src/vm/engine.rs` DIV arm). Decide which is correct.
   - Docs list `for/call/return` as non-executing; they **now execute**. The real
     subset is larger than the docs claim.
2. **Pick the canonical VM.** Rust↔`opcode-vsa-rs` opcode bytes are test-enforced
   (`cubelang/tests/opcode_sync.rs`); the **Python `cubemind` VM sync is convention
   only**. Target the Rust VM (it has the executable + the `validate` fence), and
   either add the Python sync test or drop Python from the contract.
3. **The AST head/router is a no-op today.** Making emission real needs: (a) a
   paired **NL → CubeLang program** corpus (extend the GSM8K opcode builder), (b)
   training the **router** to gate lang-vs-program tokens, (c) mapping the 47 AST
   tokens to the executable opcode + role + structural set, (d) grammar-masked
   decode wired to `validate`.

## ⚠️ DEFERRED — come back to this: two emission/verification modes
Empirically tested `cubelang/examples/conversation_agent/` against the real VM:
both `check` + `compile` + `run` succeed. This proves the emission target is wider
than flat arithmetic tuples. There are **two modes**, both compile-and-run:
1. **Scalar-verifiable** (arithmetic/decision → a number): ground truth =
   `result == gold`. Airtight today (the v4 corpus, GSM8K, `conversation_min.cube`
   → `solve() -> 10`).
2. **Structural ISolver modules** (full templates with `struct`/`event`/`match`/
   `storage` → a **struct** output, e.g. `conversation_agent.cube` →
   `solve() -> null`): not externally gold-checkable (no scalar), but **self-verify**
   via the program's own `verify()` (e.g. `text != "" && confidence > 0 &&
   trace.length > 0`), which runs deterministically.
Caveat: mode-2 leans on `match`, which the VM currently compiles with ALL arms
unconditionally (P2-2) — so it executes but the branching isn't yet faithful.
**TODO when we return here:** fold the two-mode distinction into the emission ladder
(scalar-verifiable now → structural self-verify next → faithful `match` once P2-2
lands), classify the v4 sets by scalar-vs-struct return, and target the
`conversation_*` style as the mode-2 emission exemplar.

## Build-order placement
This is rung **0.0.8** (CubeLang head + VM bridge), but the *contract* can be fixed
now and the dual-head trained toward it incrementally. The dependency: the trunk
must generate (substrate) and the router must actually train (fix the no-op) before
the AST head can carry real program traffic.
