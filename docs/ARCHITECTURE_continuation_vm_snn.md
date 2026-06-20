# cubby-lm -- Architecture continuation: discipline, the VM, and the SNN input layer

> Status: continuation of the June 2026 technical pre-paper. This part brings the
> pre-paper's framing in line with the engineering reality captured in
> `docs/ARCHITECTURE.md`, `docs/why_sparse_cubby.md`, and `cubby/config.py`. The
> earlier sections describe the *target* system as if assembled; this part states
> the **build discipline** that gets there, and develops the two pillars the
> pre-paper underspecified: the **reasoning VM** (what actually separates
> reasoning from generation) and the **SNN afferent input layer**.

---

## 8. Build discipline -- why nothing ships all-at-once

The pre-paper above presents cubby-lm as a single end-to-end system trained under
one multitask objective. That is the *destination*, not the *method*, and the
distinction is load-bearing. An earlier 516M build (`cubby_516m_v4`: MinGRU +
sparse windowed attention + MoE + MTP + a fixed-codebook VSA word-vocab head, all
co-trained) reached val PPL ~7 yet generated degenerate text under both greedy
and sampled decoding. Two root causes, both of which the present build is
structured to avoid:

1. **A VSA binding head on the 65k word vocabulary.** Cosine readout against a
   fixed random bipolar codebook with a frozen temperature and untied embeddings
   is a low-rank softmax bottleneck: it yields low teacher-forced cross-entropy
   but a fragile, attractor-prone free-running distribution. The control that
   exonerates the recurrence is decisive -- a 5.6M MinGRU with a **weight-tied
   linear head** and everything else off free-runs coherent English at PPL ~7.26.
   The recurrence generates; the VSA word head was the regression.

2. **Everything-on-at-once.** A novel head, MoE with a hand-tuned balancer, MTP,
   sparse attention, and residual scaling shipped together, so the failure was
   unattributable.

The corrective is a single rule that governs the whole program:

> **One validated component at a time, at matched budget, scored on generation
> -- not perplexity alone.** Any rung that drops generation quality while PPL
> stays flat is quarantined and understood before the next is added.

This reframes three of the pre-paper's headline claims:

- **The word head is tied-linear, not VSA.** VSA-as-readout earns its keep only
  on the small, closed opcode/VM space (Section 9), where near-orthogonal codes
  separate trivially and a shared codebook lets opcodes round-trip
  neural<->symbolic. On the 65k word vocab it is the v4 failure and is deferred to
  the very last rung, only ever as a *corrected* variant (learned temperature +
  learned codebook) measured head-to-head against tied-linear at equal budget.
- **MTP is decode-time only.** It hurt at 516M during pretraining, consistent with
  the multi-token-prediction literature; it attaches post-training as a frozen
  self-speculative head, never as a co-trained pretraining objective.
- **The multitask objective is staged, not simultaneous.** L_text trains the
  substrate first; L_bytecode, the auxiliary balancing loss, and any program
  supervision enter only as the rungs that introduce their components come online,
  each gated.

Every advanced component named in the pre-paper -- MoE, Hebbian growth, segment
memory, the VSA head, the CubeLang bridge, the adapter bank, the input gate -- is
**flag-gated and off** in the substrate config, and an unbuilt component *raises*
rather than silently no-ops. The build order is forced by a dependency DAG, not
by preference: a router with no trained cortices to route to is inert; cortices
with no substrate that produces usable features are inert; the substrate is the
language model, which must generate before anything rides on it.

### The ladder (semantic-versioned rungs)

| ver | adds | gate |
|---|---|---|
| 0.0.0 | MinGRU + tied-linear + SwiGLU (small ~5.6M) | forward parity vs reference + coherent TinyStories generation |
| 0.0.1 | scale to production shape (d=1024, L=18), BBPE-65k | val PPL descends; coherent prose |
| 0.0.2 | chunked sliding-window attention (W=512, every 3rd) | linear memory in S; window-leakage parity |
| 0.0.3 | sparse MoE-MinGRU (4 experts/top-2 + shared) | load balance; per-token ~top-K |
| 0.0.4 | Hebbian growth (Oja + lateral inhibition -> spawn) | spawns on novelty; no destabilization |
| 0.0.5 | SegmentMemory (Hebbian-keyed compressive store) | 128k effective context; bounded per-chunk memory |
| 0.0.6 | VSA binding head (frozen MAP-bipolar, D=10240) | trains; coherent; PPL ~6 |
| 0.0.7 | MTP -- **decode-time only**, frozen trunk | speculative-decode lift; trunk grads 0 |
| 0.0.8 | **CubeLang head + VM + WorldManager arena** | parse->compile->execute->verify rate |
| 0.0.9 | MindForge adapter bank | adapter lifts task, no regression |
| 0.1.0 | **afferent SNN input gate** (system integration) | threat path pre-empts router; router hits right specialist; spawns on novel regime |

The two rungs in bold are the pillars the pre-paper underspecified. They are
developed next.

---

## 9. The reasoning VM -- what actually separates reasoning from generation

The pre-paper's Section 5.1 (closed-grammar bytecode synthesis) and Section 6
(factual bias handoff) describe the *output side* of reasoning separation -- the
LM emitting bytecode -- but skip the part that does the separating: a
**deterministic virtual machine that executes the bytecode over grounded
representations**, off the token axis entirely. This is the third pillar, and
without it the LM is just emitting opcodes into a void.

### 9.1 The conflation the VM exists to break

In test-time-compute reasoning models, reasoning *is* token generation: the
chain-of-thought is the reasoning substrate, so any perturbation of the textual
input perturbs the reasoning itself. Under maximum perceptual uncertainty --
confounder columns, smooth probability distributions over candidate values -- a
token-stream reasoner drowns in self-generated branching (tens of thousands of
internal tokens) and collapses toward chance, because it has nowhere to *put* a
belief state over competing rule hypotheses except into more tokens.

cubby-lm moves reasoning off the token axis. The language model handles language;
a symbolic layer handles logic, operating on **grounded representations** --
integer attributes and (k, l) block-codes -- never on tokenized text. The
principle is *ground your representations before you reason, not after*: bind
heterogeneous inputs into the shared VSA space first, then run deterministic
reasoning over the binding.

### 9.2 Mechanism: CubeLang, the VM, and the WorldManager

Three components implement the separation. All three are real repositories
(`cubelang`, `opcode-vsa-rs`, and the reasoning runtime), at differing maturity:

- **CubeLang + the VM.** CubeLang is a typed, interface-bound language; every
  program implements `ISolver`: `parse` (raw -> input), `solve` (input -> output),
  `verify` (input, output -> bool), optionally `learn`. Programs carry persistent
  `storage`, decorators, and ordinary control flow, and lower to VM bytecode
  executed deterministically under hard safety guards (bounded instruction count,
  divide-by-zero -> 0, unknown jump/call targets are no-ops). The opcode set is
  mirrored across the Python VM and the Rust implementations
  (`opcode-vsa-rs/src/ir.rs`, `cubelang/src/vm.rs`) so a program means the same
  thing in both.

- **WorldManager specialists with integer-domain detectors.** For structured
  reasoning, each scored attribute is handled by an independent specialist that
  tests candidate rules -- Constant, Progression, Arithmetic, Distribute-Three --
  with **exact integer arithmetic** and predicts the missing value. The scored
  attribute set is a structural constraint: attributes outside it are never read,
  which is *architectural* immunity to confounder noise rather than a learned
  robustness.

- **Active-Inference DecisionOracle.** When candidate rules tie, an ensemble
  maintains competing hypotheses and minimizes an Expected-Free-Energy proxy over
  N rolled-forward "world" futures combined by a top-k operator. This *is* the
  probabilistic superposition over competing world models that token reasoners
  lack -- realized in vector space (~2 MB for 128 worlds) rather than in tokens.

### 9.3 Why separation buys robustness -- a three-level taxonomy

| level | attack surface | defense |
|---|---|---|
| Semantic | which attributes are rule-governed? (confounders) | architectural -- the scored set ignores unscored columns |
| Syntactic | can you parse a smooth distribution like `<0.20::4, 0.51::5, 0.29::6>`? | **N/A by construction** -- the VM reads ground-truth integers, not tokenized text |
| Arithmetic | do rules generalize from [1,10] to [1,1000]? | exact integer induction is range-invariant |

The "N/A by construction" cell is the crux: a prompt-level perturbation is a
*category error* for a system that reasons on structured data -- the way a font
change does not affect a visual cortex. This is a stronger statement than scoring
100% against the attack: the attack does not apply. (Grounding results exist and
are strong, but are under conference submission embargo and live only in the
researcher-facing reasoning paper, not here.)

### 9.4 The honest status line

This pillar is **partially built**, and the paper must say so. The executing
subset of CubeLang -- `create`, `assign`, `add`, `sub`, `store`, `remember`,
`query`, `sum`, `compare`, `if/else` -- computes values today and is the
ground-truth-verifiable core. The extended reasoning opcodes (`predict`, `match`,
`score`, `analogy`, `temporal_bind`, `discover`, ...) currently **trace but do
not yet compute** (tracked as CUBELANG_FIXES P0-1). The consequence is a
discipline, not a disclaimer:

> The LM's emission target is the **executing subset**, grown in lockstep as the
> VM's reasoning opcodes come online. Training Cubby to fluently emit language the
> VM cannot run would make compile/execute/verify stop being ground truth.

So the bridge (rung 0.0.8) ships against the subset the VM can actually execute,
and the program-synthesis metric is never PPL -- it is the ladder
**token-CE -> parses -> compiles -> executes -> satisfies `verify()`**, entered as
eval metrics first and only later, optionally, as reward terms. Production decode
is grammar-masked to the closed opcode grammar so the head cannot emit an invalid
program.

---

## 10. The SNN afferent input layer -- the front door

The pre-paper's Section 2 introduces a spiking front-end as an *encoding*
convenience -- LIF population coding that turns spikes into embeddings. That
undersells it. In the assembled system the SNN is the **afferent input layer**:
the gate every input passes through *before* the trunk, modeled on the
dual-pathway (LeDoux) split between a fast subcortical threat route and a slower
cortical appraisal route.

### 10.1 Two paths

1. **CNS reflex path (the low road).** A spiking perceptron scores
   salience/threat on every input from its spike-rate/synchrony -- not a dense
   softmax. When the score crosses `gate_stress_threshold`, the input is escalated
   to the CNS (`live_brain`: amygdala + a 5-hormone neurochemistry model) as an
   *interrupt* that pre-empts the cortical router. Urgent, high-stress input is
   handled by the reflex system first, the way a startle response precedes
   deliberation.

2. **Cortical router path (the high road).** Otherwise the input goes to a router
   that classifies it along {modality, tone, intent, domain} and dispatches it to
   the WorldManager **specialist** that owns that regime. If no specialist owns
   it, the router triggers a Hebbian expert-on-demand spawn and registers the new
   specialist's block-code with the arena, so the next input in that regime
   already has an owner.

### 10.2 Why spiking, specifically

The gate must be cheap, always-on, and event-driven -- it scores *every* input,
so a dense matmul per token is the wrong cost model. An SNN perceptron is the
natural fit: sparse spikes, no dense projection per token, and it shares substrate
and tooling with `live_brain` (grilly-backed, event-driven). The LIF membrane
integrates incoming spikes and fires on threshold crossing; the threat signal is
read from firing statistics rather than a learned dense head.

> **Engineering correction to the pre-paper's Section 2.** The update rule printed
> there, `U[t] = H[t-1] + X[t]`, is a pure integrate-and-fire accumulator -- it
> has no leak and no reset, so as written it is not "leaky." The implemented neuron
> is a proper LIF cell: a leak term decays the membrane between events and a reset
> follows each spike. State the leak and reset explicitly, or call it IF, not LIF.

### 10.3 Why it is the *last* integration milestone, not the first

It is tempting to draw the SNN gate at the top of the diagram and conclude it is
built first. The dependency DAG says the opposite: the gate can only do its job
once the things it routes *to* exist -- the **CNS** (`live_brain`, present), the
**router** (MoE routing, rung 0.0.3), and **specialists** (Hebbian growth +
WorldManager, rung 0.0.4). A triage gate with nothing to triage to is inert. So
the gate is the first **system-integration** milestone (0.1.0), wrapping a trunk
that already generates, routes, and grows -- not a trunk rung. For substrate tests
(0.0.0) it is bypassed entirely: raw tokens go straight to the trunk, and
`enable_input_gate` stays false until 0.1.0.

`live_brain` therefore plays two distinct roles and they should not be conflated:
the **CNS reflex path** the gate escalates to under threat, and later a **modality
cortex** (vision) hooked in after the substrate is solid. It is a parallel track,
not part of the 0.0.x trunk.

---

## 11. One substrate, three readings

The reason these pillars compose rather than merely coexist is that they are three
readings of a single VSA substrate, connected through one shared Hebbian basis W:

- **Context streams *through* it.** The MinGRU recurrence is O(L); chunked
  windowed attention keeps memory linear in sequence length; SegmentMemory pushes
  the loss floor from recurrence-decay to retrieval-quality, addressed by W.
- **Memory is *indexed in* it.** Hebbian growth (residual = 1 - ||W^T y||^2 /
  ||x||^2), WorldManager arena tagging, and segment-memory addressing all read the
  same W. The first `sig_dim` rows are stable under growth, so addressing stays
  coherent as experts spawn. One novelty signal, three consumers.
- **Reasoning *runs on* it.** The CubeLang VM reasons in the same (k, l)
  block-code algebra the binding head and arena use; the program head shares the
  binding codebook. The world model the reasoner needs is the one the memory
  pillar was already building.

The afferent SNN gate sits *above* all three: it decides, per input, which slice
of this substrate fires -- the reflex CNS, or a specific cortical specialist --
and spawns a new one when the input belongs to a regime no specialist owns yet.
Routing sparsity over heads mirrors MoE sparsity over experts: inactive paths cost
~0.

That shared grounding is the whole thesis. None of the individual primitives is
novel -- windowed attention, Oja/Hebbian learning, MoE growth,
content-addressable memory, VSA binding, spiking triage, and symbolic VMs all
exist in prior work. What is specific to cubby-lm is their **connectivity through
one plasticity signal**: the basis that discovers feature axes drives neurogenesis
(a new expert), hippocampal indexing (segment memory), world-model tagging (the
arena), and -- at the top -- the afferent gate's routing. Build it one validated
rung at a time, keep the word head tied-linear and VSA where it belongs, execute
reasoning in the VM rather than the token stream, and let the SNN gate the whole
thing once there is something to gate. That is the system the pre-paper describes,
built the way it can actually be made to work.

---

*Continuation ends. The three pillars -- unlimited context, model-native VSA
memory, and reasoning separated into the VM -- plus the SNN afferent gate, are one
substrate read four ways: streamed through, indexed in, reasoned on, and gated at
the door.*
