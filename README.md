# Why Sparse Cubby

*Grillcheese Research Laboratory — May 2026*

## Abstract — in plain terms

If you use AI for real work, you have probably hit the wall that became hard to
ignore by the middle of 2026: the better the AI gets at actually thinking
through a problem, the more it costs to run — and for a lot of everyday tasks
that bill now lands *higher than simply paying a person to do it*. The newest
"reasoning" systems charge by how much they think out loud, so the harder the
question, the more they write to themselves, and the faster the meter spins. The
cost is not just high; it is unpredictable, which makes it almost impossible to
budget.

Cubby is our answer to that problem. It is an AI architecture built from the
ground up to keep the two most expensive things — *remembering* and *reasoning*
— cheap and, just as importantly, **predictable**. Cubby does its remembering
and reasoning over a single shared structure of fixed size (on the order of a
gigabyte) that does not grow every time you ask another question. Asking it to
think harder doesn't keep running up the tab the way today's think-out-loud
systems do, because the heavy lifting happens on that compact, reusable
structure instead of by generating ever more text. In practice that pushes the
per-question cost of memory and reasoning toward a small, flat floor rather than
an open-ended climb.

What this looks like in everyday use:

- **Working through long material** — a contract, a case file, a research
  binder, a whole book — and being able to come back to it again and again
  without paying more each time you do.
- **No more workarounds to stretch the context window or make your AI
  remember** — no chopping documents into pieces, no juggling external databases,
  no re-pasting old conversations. Memory and long context are built in, so the
  assistant still knows what happened months ago instead of starting from
  scratch every session.
- **Getting reliable answers to questions that follow clear rules** — "does this
  qualify?", "what does the policy say here?", "what's the total?" — where you
  need the *right* answer, not a confident-sounding guess, and you want it
  without a big bill.
- **A memory that just keeps growing.** Cubby stores what it learns using
  10,240-dimensional addresses — which sounds technical, but the practical upshot
  is simple: that scheme has more possible storage slots than there are atoms in
  the universe, so for any real use it is effectively unlimited. It can hold on
  to everything it sees, remember the past, and build on it — making it more
  useful and more accurate for you today.

Two goals follow directly from keeping costs low, and they matter to us in their
own right. The first is **accessibility**: Cubby is designed to run on ordinary,
affordable hardware rather than requiring a data-center cluster, so that capable
AI isn't reserved for the organizations that can spend the most on it. Much of
its development and validation has been done on a consumer graphics card you
could buy off a shelf. The second is being **continuously trainable**: instead
of being frozen at the moment it was built and only improvable through expensive
full retraining, Cubby is designed to keep learning and growing new skills in
small, cheap increments — adding what it needs where it needs it, while it runs.
That also means no more waiting for "version 2": improvements arrive as small
add-ons rather than as a brand-new model you have to switch to. And on the rare
occasion a genuinely major new capability does warrant a new version, it still
doesn't require retraining from scratch — new abilities attach on top of the
existing model as lightweight "adapters" and world models, and separately built
models can be combined rather than rebuilt. You keep everything Cubby already
knows and simply add to it.

Both goals point at the same outcome we care about: **lower energy use and lower
emissions.** Today's frontier AI is energy-hungry largely because of two habits
— retraining enormous models from scratch, and "thinking" by generating huge
amounts of text for every hard question. Cubby is built to avoid both: it grows
by small attachments rather than wholesale retraining, and it reasons over a
compact, reusable structure instead of burning compute to write out its
thinking. Less compute per task, on cheaper hardware, with no giant retraining
runs, adds up to a meaningfully smaller energy and carbon footprint for the same
useful work.

One thing we want to be clear about: Cubby is **not** built to replace people.
It is built to make a person's work cheaper, faster, and more dependable — to
take the parts that are tedious or repetitive off their plate and hand back time
and reliable answers, at a cost they can actually plan around. The goal is to
optimize human work, not to substitute for the human doing it.

The rest of this document explains how, in two parts. Part 1 is for a general
audience and continues in this plain-terms style. Part 2 is for AI practitioners
and researchers, with the mechanisms and the precise limits of what has and
hasn't yet been proven.

---

This document makes the case for Sparse Cubby around the three capabilities it
was built to deliver: **unlimited context**, **real memory via a
vector-symbolic architecture (VSA)**, and **better reasoning by separating
reasoning from the language model** — running it in a symbolic CubeLang VM over
grounded world models rather than inside the token stream. It is written in two
parts. Part 1 is for a general audience and frames the case against the limits
of mainstream language models. Part 2 is for AI practitioners and researchers
and explains the mechanisms behind each of the three capabilities.

> Companion documents: `cubby_unlimited_context.md` (the base model and its
> long-context results), `sparse_cubby_paper.md` (the additive, no-retrain
> extension), and `docs/papers/cubemind_iravenx_neurips2026.md` (the I-RAVEN-X
> reasoning results that ground the third pillar). This document is the "why";
> those are the "what" and "how."
>
> **Embargo note.** The I-RAVEN-X work is under NeurIPS 2026 submission
> embargo. Specific benchmark numbers and dataset internals appear only in the
> researcher-facing Part 2 and must not be circulated outside the repository
> until the embargo lifts.

---

# Part 1 — For everyone: what's broken, and what Cubby fixes

## The two problems nobody has fully solved

Today's most capable language models are extraordinary, but they share two
structural weaknesses that show up the moment you ask them to work over long
documents or to genuinely remember things.

**Problem one: context that costs too much.** A standard transformer reads text
using *attention*, where every word looks at every other word. That comparison
grows with the square of the length: double the document and the work
quadruples. This is why long inputs get expensive fast, why context windows
have hard ceilings, and why feeding a model an entire book or a months-long
conversation is either impossible or eye-wateringly costly. The industry's
usual answer is to throw bigger hardware at it. That postpones the ceiling; it
doesn't remove it.

**Problem two: no real memory.** Even inside its window, a model has no durable
memory. When the conversation scrolls past the window, the older material is
simply gone — not stored somewhere and recalled, just dropped. The model that
finishes a long session is, in a real sense, the same model that started it. It
learned nothing it can keep. Bolt-on tricks (stuffing old text back into the
prompt, external databases of text snippets) help, but they treat memory as
text to re-read rather than as something the model *understands and indexes in
its own terms*.

Cubby is an attempt to fix both at the architectural level rather than by
working around them.

## Fix one: unlimited context, at a cost that grows in a straight line

Cubby's core is *recurrence-first*. Instead of leaning on quadratic attention,
it processes a sequence with a lightweight recurrent unit (a MinGRU) whose cost
grows *linearly* with length — read twice as much, do twice the work, not four
times. Where the model does need the precise, position-aware lookups that
attention is good at, it uses a **sliding-window** version that only looks at a
nearby span of text, processed in chunks. The result is an attention path whose
memory also grows in a straight line with length instead of as a square.

The payoff is concrete and measured. An earlier implementation of the attention
layer ran out of memory at a sequence length of just 1,024 *even on a 180 GB
datacenter GPU*. The rebuilt chunked version trains at **32,000 tokens of
sequence length using about 7.5 GB** on a 96 GB card — and then processes
**128,000 tokens of effective context** by streaming through four successive
chunks, with no extra parameters and bounded memory per chunk. The bottleneck
that used to require throwing hardware at the problem is gone by design.

## Fix two: real memory the model builds in its own language

This is where Cubby goes somewhere most architectures don't. Two pieces work
together.

First, Cubby continuously asks itself *"is this new?"* As it reads, a
lightweight learning rule (a Hebbian probe, the same family of mechanism the
brain uses to strengthen connections) watches for patterns its existing
machinery can't already explain. When something genuinely novel shows up, the
model can **grow a new specialist on the spot** to handle it — adding capacity
where it's actually needed, while it runs, instead of being frozen at whatever
size it was trained at.

Second, Cubby keeps a **compressive memory** outside the model that stores
summaries of everything it has read, *indexed by the model's own evolving sense
of what matters* rather than by raw text matching. On later passes it looks up
the relevant past summaries and feeds them back in. Because the index is the
model's own feature map, the memory's addressing improves as the model's
understanding improves.

Underneath both is the **vector-symbolic architecture**. Instead of representing
knowledge only as opaque lists of numbers, Cubby can encode concepts as
structured high-dimensional vectors that can be *combined, queried, and
recalled* with well-defined algebra — bound to roles like "who," "what," and
"when," compared for similarity, and stored in a shared "world model" that the
rest of the system can search. This is the difference between a model that
*re-reads* its notes and one that has an organized, queryable memory of what it
has seen.

## Fix three: reasoning that doesn't live inside the words

There is a third weakness, and it is the deepest one. In today's reasoning
models, *thinking is text*. The model reasons by generating a long
chain-of-thought — it talks its way to an answer, token by token. That works
remarkably well when the input is clean. But it means the reasoning and the
language are the same activity, happening in the same place. So the moment the
input gets noisy, uncertain, or padded with irrelevant detail, the model's
"thinking" gets confused along with its reading. It can spend enormous effort —
tens of thousands of words of internal monologue — drowning in possibilities it
generated itself, and still land near a coin-flip.

Cubby takes reasoning *out* of the language model. The language model does what
it is good at — handling language. A separate engine does the actual reasoning:
a small, deterministic virtual machine (running programs in a purpose-built
language called CubeLang) that operates over structured "world models" rather
than over words. It finds the governing rule, checks it with exact arithmetic,
and commits to an answer — on the grounded representation of the problem, not on
the text describing it.

Why this matters: when reasoning happens on structure instead of on text,
attacks on the text can't derail it. Noise columns it was never asked to look at
are simply invisible. Fuzzy, probabilistic wording in the prompt is irrelevant,
because the engine reads the underlying values directly. And because the rule
induction is exact rather than a guess, it doesn't get *worse* as the numbers
get bigger — if anything it gets cleaner. It's the difference between a
calculator that computes through circuits and a person scribbling on paper:
smudging the paper tests handwriting, not arithmetic. On the benchmark built
specifically to break reasoning models with this kind of perceptual noise,
systems that reason inside their words fall to near-chance, while reasoning on
grounded world models stays reliable. (The specific results are in Part 2.)

This is also why the three capabilities reinforce each other rather than just
sitting side by side. The same structured "world model" that stores Cubby's
memory is the thing its reasoning engine reasons over, and the same
high-dimensional vector algebra underlies both. Memory, context, and reasoning
share one substrate.

## Why "sparse," and why it matters to you

"Sparse" is the discipline that makes all of this affordable. At any moment only
a small, relevant fraction of Cubby's machinery actually runs — the right
experts for this input, the right memory bucket, the right specialist heads.
Everything else costs essentially nothing until it's needed. Sparsity is what
lets the model carry a large, growing set of capabilities and a large memory
without paying for all of it on every token.

The practical promise is a model that can read something arbitrarily long
without falling off a cost cliff, that remembers what it read in a structured
and improvable way, and that grows new capabilities by attachment rather than by
the expense of retraining from scratch. The rest of this document explains how.

---

# Part 2 — For researchers: the mechanism case

This part assumes familiarity with transformer scaling, mixture-of-experts,
state-space/recurrent models, and vector-symbolic / HRR-style representations.
It explains how the architecture works, mechanism by mechanism.

![System overview — the Sparse Cubby trunk](figures/09_architecture.svg)

*System overview — the MoE-MinGRU trunk with multi-token-prediction heads, all sharing one frozen VSA binding codebook.*

## 1. Thesis

Sparse Cubby is a recurrence-first language model whose design decouples the
four scaling pressures that normally collide on one hardware ceiling — sequence
length, expert capacity, training cost, and working memory — and routes each to
a mechanism suited to it. The three capabilities we foreground here — **unlimited
context**, **real VSA memory**, and **reasoning separated from the LM** — are
not separate features bolted on; they fall out of a single shared substrate. A
feature axis discovered by the recurrence becomes simultaneously (a) a routing
axis for the experts, (b) a tagged specialist in a global VSA world model, and
(c) a retrieval bucket in the long-context store — without those three consumers
coordinating explicitly. And that same global VSA world model is the substrate
the reasoning layer (§4) operates over: memory and reasoning share one
representational algebra rather than living in different subsystems.

## 2. Unlimited context

### 2.1 Linear sequence core

The working core is a **MinGRU** linear-recurrent gated unit computed by a
parallel prefix scan: `x_scan = sigmoid(g) ⊙ tanh(v)`, decay
`a = 0.001 + 0.998·sigmoid(d) ∈ (0,1)`, then `h_t = a_t ⊙ h_{t-1} + x_scan_t`.
The recurrence is solved with Heinsen's log-domain associative scan, giving
O(log T) GPU depth rather than the O(T) of a naive sequential loop — at
seq=1024 × 12 layers this collapses ~12k sequential kernel launches into ~12
fused calls, the difference between roughly 1k tok/s and 100k tok/s on an H200.
Sequence cost is O(L), with no quadratic term.

### 2.2 Chunked sliding-window attention (the structural fix)

Position-precise lookups come from a sliding-window causal attention path
spliced in every `attn_every_n` layers, window `W` (default 512). The naive
implementation builds an `(S, S)` mask and hands it to SDPA — but supplying any
`attn_mask` *disables* FlashAttention fusion and forces materialization of the
full `(B, H, S, S)` scores tensor. At production shape (`S=1024, BS=24, H=16,
D=1024`, 6 attention layers) this OOMed on a 180 GB B200.

The fix processes `Q` in chunks of `W`; each chunk gathers keys/values from the
union of its rows' windows `[max(0, c_start−W+1) : c_end]`, so each per-chunk
SDPA call is a small `c_len × (≤2W−1)` operation on the FlashAttention-2 fast
path. Peak attention memory per chunk is `B·H·W·(2W−1)`, independent of `S`; the
`ceil(S/W)` chunk count is the linear-in-L factor.

**Measured (isolation-mode tiny model, bf16):** memory scales linearly with `S`
at every window — `1k → 64 MB`, `4k → 246 MB`, `16k → 968 MB`, `32k → 1929 MB`
at `W=512`; window choice is a 1.0–1.5× constant. At production shape (d=1024,
L=18, attn_every=3) block-level activation checkpointing brings the 32k
forward+backward+AdamW peak to **5.2 GB**, versus OOM for the un-checkpointed
path. A 50-step, 32k-sequence, 500M-class training run held **7.50 GB peak** out
of 95 GB with monotonic, all-finite loss — confirming correct gradient flow
through the chunked-attention and checkpointing boundaries. Correctness is
covered by `tests/test_local_attention.py` (numerical equivalence vs. a
brute-force `(S,S)` reference at six `(S,W)` settings, window-leakage, and
checkpoint-equivalence in both forward and gradient).

### 2.3 From long context to *unlimited* context

A finite recurrent state cannot hold unbounded history — that is
information-theoretic, not an engineering gap. Cubby pushes the loss floor from
"recurrence decay" up to "retrieval quality" with an external **SegmentMemory**
(§3.2). The contract is `forward_with_memory(chunk, mem, write=True,
read=True)` over successive chunks: per-chunk compute is the §2.2 O(L·W) cost
plus an O(sig_dim) lookup, and stored history sits on CPU, not VRAM. A 4 × 32k =
**128k effective context** run grows memory monotonically (1→2→3→4 entries),
holds per-chunk peak in the 5–8 GB envelope, and produces outputs that differ
measurably (~0.72 in logit space) from a fresh-memory baseline — i.e. the
injection is a real bias path, not a no-op.

## 3. Real memory via VSA

### 3.1 A single novelty primitive

Three mechanisms that look independent are the same *"is this new?"* signal,
and treating them as one is the accurate accounting:

- **Hebbian expert growth.** A `HebbianGrowthLayer` (nonlinear Oja rule +
  lateral inhibition) inside each MoE block tracks the unexplained-variance
  fraction `residual = 1 − ‖Wᵀy‖²/‖x‖²`. When `residual_ema > threshold`,
  `K < max`, and cooldown is clear, it appends the most-novel sample's
  normalized residual as a new basis row and publishes a `NoveltyEvent`; the MoE
  reacts by spawning a fresh MinGRU expert in place (gate resized, DeepSeek bias
  buffers extended, optimizer re-synced via `sync_optimizer_with_model`).
- **WorldManager arena spawning** — a block-code far from every existing
  specialist.
- **VSA cache surprise** — `1 − max_similarity` to any stored prototype.

All three read the same Hebbian basis `W`, which is what makes the substrate
coherent rather than three parallel features.

### 3.2 Hebbian-keyed compressive memory

`SegmentMemory` is an external, non-`nn.Module` store of
`(signature, value, bucket, age)` tuples with bucket-indexed retrieval:
hash a query into a bucket by `argmax(signature)`, cosine-rank within the
bucket, softmax-weight the top-k, sum the values — O(bucket_size + sig_dim),
independent of total stored count up to the per-bucket cap. The key design
choice is the **signature function**: rather than a random projection
(addressing as a hash, unrelated to the model's features), the default uses the
model's own Hebbian basis, `sig = value @ W_hebbian[:sig_dim]ᵀ`. The first
`sig_dim` rows are *stable under growth* (the rule appends at index `K+`), so
bucket assignments stay coherent across a run while addressing follows the
model's discovered feature axes. This closes the loop: the basis that grows the
experts and tags the world model also addresses the long-context store.

### 3.3 The VSA binding head and the world-model arena

The output head can replace the tied `Linear(d_model, V)` softmax with a
`Linear(d_model, D)` projection into a **frozen MAP-bipolar codebook** (default
`D=10240`), scored by cosine. The embedding stays learned; the head is untied;
the codebook is a fixed buffer. This is what makes downstream VSA reasoning
first-class: tokens live in a hyperdimensional space with HRR-style
bind/unbind, and the **WorldManager arena** is an inference-time NumPy store of
specialist "rule" vectors in `(k, l)` block-code space, fed by a fixed seeded
Gaussian projection (`d_model → k·l`, seed `0xC0DEB00C`, variance-preserving,
Johnson–Lindenstrauss). Same seed across processes ⇒ same block-code for the
same direction ⇒ distributed runs share one arena. The bridge
(`cubemind/execution/novelty_bridge.py`) duck-types the event so `model.cubby`
never imports `cubemind` — the one-way decoupling is preserved.

Crucially, the projection is *similarity-preserving but not invertible*: the
arena routes and indexes, it does not reconstruct hidden states. It is a router
and a world-index, not a representation merger — a boundary we keep explicit
because conflating the two is the easy error.

## 4. Reasoning, separated from the LM

### 4.1 The conflation that breaks reasoning models

In test-time-compute LRMs, reasoning *is* token generation: the chain-of-thought
is the reasoning substrate, so any perturbation of the textual input is a
perturbation of the reasoning itself. The I-RAVEN-X benchmark
[Sicking et al., 2026] makes this concrete and brutal — under "maximum
perceptual uncertainty" (condition c: 10 confounder attribute columns at
SNR −5.23 dB plus smooth probability distributions, `p_L = 0.51` over `[1,1000]`),
the chain-of-thought drowns in self-generated branching: o3-mini emits ~18,482
reasoning tokens per problem and still collapses toward chance. The benchmark
authors name the missing capability precisely — "probabilistic belief
maintenance over competing rule hypotheses." A system that reasons inside its
token stream does not have a place to *put* that belief state except more
tokens.

Cubby's stance is to move reasoning off the token axis entirely. The LM handles
language; reasoning happens in a symbolic layer that operates on **grounded
representations** — integer attributes and `(k, l)` block-codes — never on
tokenized text. This is the "ground your representations before you reason, not
after" (double-binding) principle: bind heterogeneous inputs into the shared VSA
space *first*, then run deterministic reasoning over the binding.

### 4.2 The mechanism: CubeLang VM + WorldManager world models

Three components implement the separation:

- **The CubeLang VM** (`cubemind/reasoning/vm.py`; 45 + 10 extended opcodes,
  mirrored in Rust at `opcode-vsa-rs/src/ir.rs` and `cubelang/src/vm.rs`).
  Reasoning is expressed as programs over block-codes — `BIND_ROLE`, `MATCH`,
  `PREDICT`, `ANALOGY`, `TEMPORAL_BIND`, `DISCOVER`, and the rest — lowered to
  VM bytecode and executed deterministically under hard safety guards
  (`max_instructions=10000`, DIV-by-zero→0, unknown jump/call targets are
  no-ops). The LM's role, where it participates at all, is to *emit* such
  programs (the CubeLang head emitting the closed 15-op `OpcodeStmt`
  grammar — the genuine VSA-VM bridge); it does not perform the reasoning.

- **WorldManager specialists with integer-domain detectors.** For structured
  reasoning (the I-RAVEN-X path), each scored attribute is handled by an
  independent specialist that tests candidate rules — Constant, Progression,
  Arithmetic, Distribute-Three — with exact integer arithmetic and predicts the
  missing value. The **scored attribute set** `S = {Type, Size, Color}` is a
  structural constraint: attributes outside `S` (Angle, Confounder0..9) are
  never read. There is no perception module in this path — the system reads the
  attribute structure directly.

- **Active Inference / DecisionOracle.** When candidate rules tie, an HMM
  ensemble maintains competing hypotheses and uses pairwise KL divergence
  between transition matrices as an Expected-Free-Energy proxy. The general case
  is "many-worlds" Active Inference: one shared HYLA hypernetwork modulated by
  N "world personality" vectors rolls forward N parallel futures, each with a
  pragmatic Q-value and an epistemic plausibility, combined by a `top_k`
  operator into a soft EFE minimization — ~2 MB for 128 worlds versus ~256 GB
  for 128 separate networks. This *is* the probabilistic superposition over
  competing world models that token LRMs lack, realized in vector space rather
  than in tokens.

### 4.3 Why this is robust: a three-level taxonomy

The separation yields three independent, non-interacting immunities:

| Level | Attack surface | Cubby's defense |
|---|---|---|
| **Semantic** | which attributes are rule-governed? (confounder noise) | Architectural: the scored attribute set ignores unscored columns. |
| **Syntactic** | can you parse `<0.20::4, 0.51::5, 0.29::6>`? (smooth distributions) | **N/A by construction** — reasoning reads ground-truth integers, not tokenized text. |
| **Arithmetic** | do rules generalize from `[1,10]` to `[1,1000]`? | Exact integer rule induction is range-invariant. |

The "N/A" cell is the crux: a prompt-level perturbation is a *category error*
for a system that reasons on structured data, the same way a font change does
not affect a visual cortex. This is stronger than scoring 100% against the
attack — the attack does not apply.

### 4.4 Grounding results (NeurIPS 2026 — embargoed; do not circulate)

> The following are from `docs/papers/cubemind_iravenx_neurips2026.md` and are
> under submission embargo. They belong in this researcher-facing section only.

On I-RAVEN-X condition (c) — maxval=1000, 10 confounders, `p_L = 0.51` — CubeMind
scores **100.0%** where o3-mini scores **17.0%** and DeepSeek R1 **23.2%**
(random chance 12.5%), solving 200 problems in **1.86 s** on an NVIDIA L4 with
**0 prompted tokens** against o3-mini's ~18,482. The smooth-distribution-only
condition is reported as **N/A by construction** (a structured-data exemption,
not a measured win — stated as such, not as 100%). On standard I-RAVEN
(7 configurations) the mean is **90.3%**, edging NVSA's 88.1% without any
gradient-trained rule detection. Accuracy *increases* with maxval
(97.5% → 99.5% → 100.0% at 10/100/1000), the opposite of LRM degradation —
evidence of genuine rule induction rather than value memorization.

**Scope.** This is the *abstract* reasoning path: in the I-RAVEN-X evaluation
there is no CNN or tokenizer, by design — a reviewer may call reading structured
JSON "cheating," and the reply is that RPM is fundamentally rule induction over
structured attributes; the text encoding is an artifact of how LRMs consume
input, not the task. The detectors are tuned for 3-row structure: on 10×10 grids
(99 context panels) accuracy is 41.0% and Distribute-Three detection does not
fire (0%), which marks where the current detector geometry applies. The claim
is robustness *at the correct representational level* — that reasoning on
grounded world models is immune to the perturbations that break token-stream
reasoning — not that the LM itself does the reasoning.

## 5. Why "sparse" is load-bearing

Sparsity is the property that lets the capability set and the memory grow
without per-token cost growing with them.

- **Sparse MoE recurrence.** `MoEMinGRULayer`: M small MinGRU experts, top-K
  routing, Switch-style load-balancing aux loss, DeepSeek-V3 auxiliary-loss-free
  bias rebalancing, always-on shared experts. Per-token compute stays near top-K
  regardless of expert count — so Hebbian growth can add experts without
  inflating the per-token bill.
- **Sparse adapter bank.** A lightweight router gates which
  reasoning/perception heads fire per sample — sparsity over *heads* mirroring
  the MoE's sparsity over *experts*; inactive heads cost ~0. Heads register
  through the runtime `add_head` registry, which is the no-retrain-improvement
  path in practice.

## 6. The no-retrain principle (Sparse Cubby++)

The spine of the extension: after the trunk converges, freeze it
(`requires_grad=False` on embed + N hybrid blocks + final RMSNorm + depth-0
head), and add every later capability as a **removable attachment trained only
on the trunk's frozen features**. This is feasible, not wishful, because the
trunk already exposes `forward_features(tokens) → (B,S,D)`, the multitask
wrapper already attaches aux heads on top of those features, MindForge adapters
are plastic by construction (fixed base projection + zero-init online-updatable
low-rank `basis_B`), and head registration is a runtime registry.

**Multi-token prediction (MTP)** is one such attachment. It follows
DeepSeek-V3's sequential form with the output projection shared with the trunk
head — on the binding-head trunk, k depths cost `k·(D·d_vsa)` projection params
and **zero** extra codebook memory (the ~1.3 GB codebook is shared, not
duplicated). Boundary masking drops depth-n targets that cross
document/turn/reasoning boundaries (ids `</s>=3, [/THINKING]=15, <|user|>=5,
<|assistant|>=6`). The frozen-trunk discipline holds by construction: freezing
and detaching the depth-0 hidden, a backward pass moves the MTP parameters and
leaves the trunk's gradients at zero. A full TinyStories run on an AMD RX 6750
XT via DirectML (fp32, 5.64M params, k=2, mtp_weight=0.3) descends monotonically
to **val PPL 13.35** in 775 steps / 20 minutes with coherent greedy prose by
step 500 — stable training to coherent language on consumer hardware *outside
the CUDA ecosystem*.

The other attachments compose the same way on the same frozen trunk, each
behind its own flag: the sparse adapter bank; a CubeLang program-synthesis head
emitting the closed 15-op `OpcodeStmt` grammar that lowers to VM bytecode (the
genuine VSA-VM bridge); looped "learn-to-learn" heads with mandatory input
injection (Yang et al., ICLR 2024); and a cross-trunk WorldManager arena that
composes independently-pretrained trunks in the neutral block-code space via
provenance kept beside the vector.

**The cost.** A frozen trunk caps every attachment at whatever the frozen
features already encode. Mitigations in the stack: Hebbian growth adds trunk
capacity *during* pretraining; MindForge `basis_B` adapts per sample at
inference; and a genuinely-plateaued head falls back to a short low-rank
LoRA-on-trunk finetune, not a full retrain.

**Versioning without retraining.** The consequence for releases is that
versioning becomes the exception, not the cadence. Routine improvement is the
adapter bank + Hebbian growth path above — no new model, no retrain. Even a
major capability jump is delivered without a from-scratch retrain via two
composable mechanisms: (i) the **mixture of world models and adapters** — new
abilities register as MindForge adapters and WorldManager specialists over the
existing frozen trunk's features; and (ii) the **mixture of trunks**
— independently-pretrained trunks compose in the neutral block-code arena
(shared seed + `(k,l)`), with a thin `TrunkFusionAdapter` as the only trained
parameter and both trunks frozen. A new version is therefore an *addition* to
what exists, not a replacement of it; prior capability is retained by
construction rather than re-learned.

## 7. Scope and measurement notes

A few properties are worth stating precisely so the mechanisms above are read at
the right level:

- **The VSA binding head is reported as an architectural property, not a
  benchmark ranking.** A binding-head run reaches eval PPL ~6 with coherent
  sentence structure; that figure stands as evidence the head trains and
  generates well, scored on its own split rather than as a head-to-head against
  a tied-softmax baseline (which would require a common split to be a fair
  comparison).
- **Compressive memory works at the representational level the architecture
  defines.** The injection is a real, bounded content bias keyed by the Hebbian
  basis; its usefulness tracks the maturity of that basis — the more the model's
  feature axes acquire semantic content through training, the more on-target the
  retrieved context becomes. The write/read/normalized-inject/Hebbian-addressing
  substrate runs continuously without instability.
- **Looped heads** emulate an iterative in-context solver *within the training
  distribution* — the same scope the source result defines.
- **Cross-trunk composition is routing and indexing, not computation-merging.**
  The Johnson–Lindenstrauss projection is similarity-preserving but
  non-invertible: the arena routes across trunks, while merging their
  computation is the job of the thin trained `TrunkFusionAdapter`.
- **Measurement hardware.** Results here are produced on consumer / single-GPU
  hardware (CPU, AMD RX 6750 XT via DirectML fp32, and a 96 GB Blackwell card) —
  the accessibility goal made concrete.

The architecture also removes equivalent mechanisms rather than counting them
twice — the temporal-memory interpolator's "fourier" and "hilbert" modes were
dropped once proven bit-exact equivalent to linear interpolation (FFT and the
Hilbert transform are linear). The same discipline governs how every figure
above is framed.

## 8. Why this combination is the point

None of the individual primitives is novel: windowed attention, Oja/Hebbian
learning, MoE growth, content-addressable memory, and VSA binding all exist in
prior work. What is specific to Cubby is their **connectivity through one shared
Hebbian basis** — expert spawn, world-model tagging, and segment-memory
addressing all read the same projection `W`. The biological analogue is
uncomplicated: plasticity discovers feature axes (Oja) that drive both
neurogenesis (a new expert) and hippocampal indexing (segment memory). Cubby's
contribution is engineering the software interface so each consumer reads the
same plasticity signal — which is precisely what turns "long context" into
*unlimited* context and "a vector store" into *real, model-native memory*.

The reasoning pillar is the same idea pushed one level further. The WorldManager
arena that the memory pillar populates with novelty-tagged specialists is the
*same* registry the reasoning specialists operate over, and the CubeLang VM
reasons in the *same* `(k, l)` block-code algebra that the binding head and the
arena use; the CubeLang head even shares the binding codebook. So the
three capabilities are not three subsystems that happen to coexist — they are
three readings of one VSA substrate: context streams *through* it, memory is
*indexed in* it, and reasoning *runs on* it. That shared grounding is why moving
reasoning off the token axis is natural here rather than bolted on: the world
model the reasoner needs is the one the rest of the architecture was already
building.

---

*In short: unlimited context, real model-native memory, and reasoning that runs
on grounded world models rather than on the token stream — three capabilities
built on one shared VSA substrate, on accessible hardware, and grown by addition
rather than retraining.*
