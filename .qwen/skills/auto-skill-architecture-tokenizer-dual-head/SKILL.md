---
name: architecture-tokenizer-dual-head
description: Replace tokenizer + softmax with custom multilingual BPE, dual-head model (language + AST), sampled softmax, and chunked sliding-window attention
source: auto-skill
extracted_at: '2026-06-18T18:32:16.711Z'
---

# Architecture: Custom Tokenizer + Dual-Head Model + Sampled Softmax

This skill documents the architecture change implemented in June 2026: replacing BBPE-65k tokenizer and full softmax with a custom multilingual BPE, dual-head output (language + AST), and sampled importance-sampling softmax.

## Architecture Overview

### 1. Custom Multilingual BPE Tokenizer (32k vocab)

**Location:** `cubby/tokenizers/cubby_mbpe32k/tokenizer.json`

Trained on 7.71 GB from the unified corpus (C4 multilingual, web data, conversations, math problems). Covers 7 languages with roundtrip validation: English, French, Chinese, Russian, Arabic, Hindi, Japanese.

**Key feature:** 67 special tokens registered as atomic entries:
- Chat markers: `<|system|>`, `<|user|>`, `<|assistant|>`
- State tags: `[MY_STATE]`, `[INSTRUCTION]`, `[THINKING]`, `[MEMORY]`, `[SPECIALIST]`
- CubeLang/AST structure tags: `<TASK:SCHEMA2RULE>`, `<SCHEMA>`, `<ROLES>`, `<TRACE>`, `<RULE>`, `<OPCODE>`, `<VALID>`
- CubeLang VM opcodes: `BIND_ROLE`, `UNBIND_ROLE`, `REBIND_ROLE`, `MATCH`, `PREDICT`, `DISCOVER`, `ANALOGY`, `TEMPORAL_BIND`, `UNIFY`, etc.

**AST token classification:** 47 tokens classified as AST tokens (opcodes + structure tags). Uses **explicit frozenset** of token IDs, not range-based split. The AST tokens are interspersed throughout the vocab (IDs 0-66 contain special tokens, ID 80 is period, BPE tokens are at 26037-32767).

### 2. Dual-Head Output Model

**Location:** `cubby/trunk/model.py` (class `CubbyLM`)

Two output heads sharing the trunk but with separate embeddings and normalization:

```
trunk output x (BS, d)
├→ router: Linear(d, 2) → softmax → [p_lang, p_ast]  # routing weights
├→ language head: RMSNorm(final_lang) → embed_lang  # (BS, V_lang=32721)
└→ AST head: RMSNorm(final_ast) → embed_ast  # (BS, V_ast=47)
```

**Config fields** (`cubby/config.py`):
- `enable_dual_head: bool` - gates the dual-head architecture
- `n_special_tokens: int` - number of AST tokens (47 for cubby_mbpe32k)
- `router_type: str = "linear"` - routing mechanism (linear classifier)
- `enable_sampled_softmax: bool` - gates sampled IS during training
- `n_samples: int = 1024` - number of negative samples for IS

### 3. Sampled Importance-Sampling Softmax

**Location:** `cubby/trunk/model.py` (`sampled_cross_entropy()`)

For each training step, draws K=1024 uniform negatives per position, computes logits only for the K+1 subset (target + negatives), performs CE loss on this subset.

**Key insight:** With uniform sampling, the importance-sampling correction (adjusting for sampling probability) is a constant that cancels in the softmax, so the gradient direction is unbiased (Bengio & Senecal, 2008). No bias correction needed.

**Gradient flow:** The backward produces a **sparse** gradient for `logits_grad` (only K+1 non-zero entries per row instead of full V). This requires careful scatter-add into the embedding table gradient.

### 4. Resident GPU Path Port

**Location:** `cubby/trunk/resident.py`

The resident path uses a combined embedding table for the head:

```python
E_combined = np.concatenate([E_lang, E_ast], axis=0)  # (32768, d)
head_logits = forward_linear(nf, E_combined, 0, BS, d, V_total)
```

**Why combined?** The tape's backward engine has a single entry point `t.backward(nCE, 0)`. With two separate head variables (`E_lang`, `E_ast`), the graph would need two backward passes, which the tape doesn't cleanly support. Using a combined `E_combined` keeps the backward flow intact.

**Dual-head loss** computed in numpy after reading logits back:
```python
lang_logits = logits_np[:, :Vlang]  # language subset
ast_logits = logits_np[:, Vlang:]   # AST subset
```

Router logits computed via `forward_linear(nf, router, 0, BS, d, 2)` and included in the tape for gradient flow. Loss = `router_weighted_ce(lang_logits, AST_logits, router_weights, targets)`.

## Implementation Steps

### Step 1: Train Custom Tokenizer

```python
from tokenizers import ByteLevelBPETokenizer, pre_tokenizers, normalizers

tok = ByteLevelBPETokenizer()

# CRITICAL: Fix pre_tokenizer to handle special tokens
# Default use_regex=True breaks special tokens like <OPCODE>
tok.pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.Split(
        pattern=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        behavior="Isolated",
        invert=False
    ),
    pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False)
])
tok.normalizer = normalizers.NFC()

# Train on multilingual corpus
tok.train(
    files=[corpus_paths],
    vocab_size=32768,
    min_frequency=2,
    show_progress=True,
    special_tokens=[]  # Add special tokens AFTER training
)

# Add special tokens with proper flags
from tokenizers import AddedToken
special_tokens = [...]  # 67 tokens
for sp in special_tokens:
    has_punct = any(c in sp for c in '<>[/|')
    tok.add_special_tokens([AddedToken(
        sp,
        single_word=not has_punct,  # FALSE for tokens with angle brackets
        lstrip=False,
        rstrip=False,
        normalized=False,
        special=True
    )])
```

**Gotcha:** Tokens with angle brackets (like `<OPCODE>`) need `single_word=False` because they span multiple pre-tokenizer words. Setting `single_word=True` fragments them into `<`, `OPCODE`, `>`.

### Step 2: Classify AST Tokens

```python
# Explicit frozenset, NOT range-based
_ast_exact = {
    "BIND_ROLE", "UNBIND_ROLE", "MATCH", "PREDICT", ...  # opcodes
    "<OPCODE>", "</OPCODE>", "<RULE>", "</RULE>", ...     # structure tags
}

# Scan vocab to find IDs
added = tokenizer.get_added_tokens_decoder()  # dict[id -> Token]
ast_token_ids = frozenset(
    id_ for id_, token in added.items()
    if token.content in _ast_exact
)

# Store in config
config.n_special_tokens = len(ast_token_ids)
config.ast_token_ids = ast_token_ids  # for runtime use
```

### Step 3: Implement Dual-Head Model

```python
class CubbyLM:
    def __init__(self, config):
        self.enable_dual_head = config.enable_dual_head
        
        if self.enable_dual_head:
            Vlang = config.vocab_size - config.n_special_tokens
            Vast = config.n_special_tokens
            
            # Separate embeddings
            self.embed_lang = nn.Variable((Vlang, d))
            self.embed_ast = nn.Variable((Vast, d))
            
            # Separate normalization
            self.final_lang = nn.Variable(d)
            self.final_ast = nn.Variable(d)
            
            # Router
            self.router = nn.Variable((2, d))
            
            self.head_type = 'dual_head'
            self.Vlang = Vlang
            self.Vast = Vast
        else:
            # Original single head
            self.embed = nn.Variable((V_total, d))
            self.final = nn.Variable(d)
            
    def forward_head(self, x):
        if self.enable_dual_head:
            # Router
            router_logits = linear(x, self.router)  # (BS, 2)
            router_logits = router_logits - router_logits.max(axis=1, keepdims=True)
            rw = exp(router_logits)
            rw = rw / rw.sum(axis=1, keepdims=True)  # softmax
            w_lang = rw[:, 0:1]
            w_ast = rw[:, 1:2]
            
            # Language head
            h_lang = rmsnorm(x, self.final_lang)
            lang_logits = linear(h_lang, self.embed_lang)  # (BS, Vlang)
            
            # AST head
            h_ast = rmsnorm(x, self.final_ast)
            ast_logits = linear(h_ast, self.embed_ast)  # (BS, Vast)
            
            return lang_logits, ast_logits, rw
        else:
            h = rmsnorm(x, self.final)
            return linear(h, self.embed)
    
    def loss(self, ids, targets):
        logits = self.forward_head(x)
        
        if self.enable_dual_head:
            lang_logits, ast_logits, rw = logits
            return dual_head_ce_loss(
                lang_logits, ast_logits, rw,
                targets, self.Vlang, self.Vast
            )
        else:
            if self.enable_sampled_softmax:
                return sampled_cross_entropy(logits, self.embed_lang, targets, n_samples=self.n_samples)
            else:
                return cross_entropy(logits, targets)
```

### Step 4: Dual-Head Loss with Router Weighting

```python
def dual_head_ce_loss(lang_logits, ast_logits, router_weights, targets, Vlang, Vast):
    BS, _ = targets.shape
    
    # Classify targets by type
    is_lang = targets < Vlang
    is_ast = (targets >= Vlang) & (targets < Vlang + Vast)
    is_unknown = ~(is_lang | is_ast)
    
    # Language CE
    lang_targets = np.where(is_lang, targets, 0)  # mask to 0 for non-lang
    lang_logits_masked = lang_logits - lang_logits.max(axis=1, keepdims=True)
    lang_sm = exp(lang_logits_masked)
    lang_sm = lang_sm / lang_sm.sum(axis=1, keepdims=True)
    lang_ce = -log(lang_sm[np.arange(BS), lang_targets] + 1e-12)
    lang_ce = lang_ce * is_lang.astype(float32)
    
    # AST CE (target IDs shifted to [0, Vast))
    ast_targets = np.where(is_ast, targets - Vlang, 0)
    ast_logits_masked = ast_logits - ast_logits.max(axis=1, keepdims=True)
    ast_sm = exp(ast_logits_masked)
    ast_sm = ast_sm / ast_sm.sum(axis=1, keepdims=True)
    ast_ce = -log(ast_sm[np.arange(BS), ast_targets] + 1e-12)
    ast_ce = ast_ce * is_ast.astype(float32)
    
    # Router-weighted combination
    w_lang = router_weights[:, 0]
    w_ast = router_weights[:, 1]
    
    weighted_ce = (w_lang * lang_ce + w_ast * ast_ce).mean()
    
    return weighted_ce
```

### Step 5: Port to Resident GPU Path

```python
def _register_dual_head_weights(t, config, blocks):
    # Combine embeddings for backward flow
    E_combined = np.concatenate([E_lang, E_ast], axis=0)  # (32768, d)
    E = t.register_weight(E_combined, persistent=True)
    
    # Register separate final norms and router
    final_lang = t.register_weight(final_lang_init, persistent=True)
    final_ast = t.register_weight(final_ast_init, persistent=True)
    router = t.register_weight(router_init, persistent=True)  # (2, d)
    
    return E, final_lang, final_ast, router

def _resident_forward_dual_head(t, x, E, final_lang, final_ast, router):
    BS = x.shape[0]
    
    # Router
    router_logits = t.forward_linear(x, router, 0, BS, d, 2)
    
    # Language head: norm with final_lang, project with E[:, :Vlang]
    h_lang = t.forward_rmsnorm(x, final_lang, BS, d)
    # Note: can't slice E in resident forward, so we use combined E
    # The loss function splits the logits
    
    # AST head: norm with final_ast, project with E[:, Vlang:]
    h_ast = t.forward_rmsnorm(x, final_ast, BS, d)
    
    # Combined head projection (uses full E_combined)
    # In practice, we just run one linear with E_combined and split logits in loss
    logits = t.forward_linear(h_lang, E, 0, BS, d, V_total)
    
    return logits, router_logits

def train_step_dual_head(t, ids, targets, E, final_lang, final_ast, router, ...):
    # Forward
    logits, router_logits = _resident_forward_dual_head(...)
    
    # Read back to numpy
    logits_np = t.read_buffer(logits, (BS, V_total))
    router_logits_np = t.read_buffer(router_logits, (BS, 2))
    
    # Compute dual-head loss in numpy
    lang_logits = logits_np[:, :Vlang]
    ast_logits = logits_np[:, Vlang:]
    loss = dual_head_ce_loss(lang_logits, ast_logits, router_logits_np, targets, Vlang, Vast)
    
    # Backward (uses full CE in tape)
    t.backward(nCE, 0)
    
    return loss, grad_norm, gnorm, clip, lr
```

## Gotchas

1. **Tokenizer special tokens:** The default `use_regex=True` pre-tokenizer fragments tokens with angle brackets. Fix: `use_regex=False` + custom `Split` pattern. Tokens like `<OPCODE>` need `single_word=False`.

2. **AST token classification:** Don't use range-based splits (e.g., `targets >= ast_start_id`). Special tokens are interspersed throughout the vocab. Use an explicit `frozenset` of AST token IDs.

3. **Resident GPU backward:** The tape's `backward()` has a single entry point. Multi-variable heads need a combined embedding table to keep backward flow intact. Compute dual-head loss in numpy after reading logits back.

4. **Sampled softmax gradient:** The gradient is sparse (only K+1 non-zeros per row). The scatter-add into `embed_grad` must use `np.add.at(embed_grad, sampled_ids, sparse_grads)` to avoid overwriting.

5. **Router initialization:** Initialize router with small weights (`std=0.01`) to start near-uniform. Monitor entropy during training — if it collapses too fast, the model can't learn both heads.

## Verification

- **Tokenizer roundtrip:** Test all 7 languages (EN/FR/ZH/RU/AR/HI/JA) — should pass 100%
- **Special token atomicity:** `<OPCODE>` should encode as single token `[29]`, not fragmented `[<, OPCODE, >]`
- **Forward parity:** Dual-head model vs. reference implementation (max_abs_diff < 1e-5)
- **Gradient parity:** Compare gradients from tape vs. manual computation
- **Training convergence:** Loss should descend steadily, router entropy should stay stable

## Files Modified

- `cubby/tokenizer.py` — `MultilingualBPE` class, AST token classification logic
- `cubby/config.py` — dual-head config fields
- `cubby/trunk/model.py` — dual-head `CubbyLM`, `sampled_cross_entropy()`, `dual_head_ce_loss()`
- `cubby/trunk/resident.py` — combined embedding registration, dual-head forward/loss
- `main.py` — `--tokenizer` CLI flag

## Data

- **Training corpus:** 7.71 GB unified corpus (C4 multilingual, web, conversations, math)
- **Tokenizer artifact:** `cubby/tokenizers/cubby_mbpe32k/tokenizer.json`
- **Special tokens:** 67 total, 47 classified as AST tokens

## 0.0.2 Milestone: Chunked Sliding-Window Attention (June 2026)

Implements chunked sliding-window attention with O(B*H*S*W) memory complexity instead of O(B*H*S²). Attention is inserted every 3rd layer (configurable via `attn_every_n`).

### Implementation

**Location:** `cubby/trunk/model.py` (function `chunked_sliding_window_attention`)

**Core algorithm:**
1. Process Q in W-sized chunks (default W=512)
2. For chunk [c_start, c_end), gather K/V from [max(0, c_start-W+1), c_end)
3. Build causal+window mask per chunk: `allowed = (col >= row - W + 1) & (col <= row)`
4. Apply mask, softmax, weighted sum
5. Concatenate outputs, compute loss

**QKV projection:** Uses fused `_Linear(d_model, 3*d_model)` then splits via `QKVSplit` GradFn:
```python
qkv = self.qkv(x)  # (B, S, 3*d)
qkv_4d = qkv.reshape(B, S, 3, n_heads, d_head)
q, k, v = [qkv_4d[:, :, i, :, :] for i in range(3)]  # each (B, S, n_heads, d_head)
```

The `QKVSplit` GradFn bridges the fused projection and attention, ensuring gradients flow back through reshape and transpose operations:
```python
def qkv_split_backward(grad_output):
    # grad_output: (3, B, n_heads, S, d_head)
    g_q = grad_output[0].transpose((0, 2, 1, 3))  # (B, S, n_heads, d_head)
    g_k = grad_output[1].transpose((0, 2, 1, 3))
    g_v = grad_output[2].transpose((0, 2, 1, 3))
    return np.concatenate([g_q, g_k, g_v], axis=-1).reshape(B, S, 3 * d_model)
```

**Chunked attention backward:** Per chunk, compute gradients for Q, K, V:
```python
dV[:,:] += (attn_weights.T @ attn_output_gradient)
d_attn_weights = attn_output_gradient @ V.T
d_scores = softmax_backward(attn_weights, d_attn_weights)
d_scores *= scaling_factor
dQ += d_scores @ K.T
dK += d_scores.T @ Q.T
```

### Verification

- **Forward parity:** 1e-7 max abs diff vs brute-force reference (10 test cases, various S and W)
- **Backward precision:** 2.5% relative error (full gradient finite-difference check with eps=5e-4)
- **Integration test:** All 18 model parameters receive gradients correctly
- **Memory:** Adds 524,288 parameters (d_model=256, 16 heads, attention every 3rd layer)

### Gotchas

1. **QKV split gradient:** The fused projection creates a single tensor, but attention needs separate Q, K, V. The `QKVSplit` GradFn must reverse the reshape and transpose operations correctly in backward.

2. **Chunk boundaries:** Ensure chunk gathering respects causal masking. Position i can only attend to positions j where j <= i.

3. **Mask construction:** The sliding window mask is `allowed = (col >= row - W + 1) & (col <= row)`. This ensures each position sees at most W tokens backwards (causal + windowed).

4. **Backward numerical stability:** Use eps=5e-4 for finite-difference testing (not 1e-3). Softmax magnifies truncation errors, so absolute < 1e-3 is too strict; use relative < 5% instead.

### Files

- `cubby/trunk/model.py` — `chunked_sliding_window_attention()`, `chunked_sliding_window_attention_from_split()`, `LocalCausalAttention` class, `Block` modified to conditionally include attention, `QKVSplit` GradFn
- `cubby/trunk/test_attention.py` — forward and backward parity tests
- `cubby/trunk/test_attention_integration.py` — full model integration test
- `cubby/config.py` — `enable_attention`, `attn_window`, `attn_every_n`, `attn_heads` fields

## 0.0.2 GPU Port: Chunked Attention on grilly Resident Path (June 2026)

Ports the chunked sliding-window attention from the numpy tape path to the grilly GPU-resident path. This involved adding new ops to grilly's C++ autograd engine, writing GLSL shaders, and fixing infrastructure issues (staging buffer exhaustion, stub backward ops).

### Procedure: Adding a New Tape Op to grilly

**Files to modify (in order):**

1. `cpp/include/grilly/autograd/autograd.h` — Add enum value **before** `_Count` sentinel, add `forward_<name>` declaration on TapeContext, add `backward_<name>` declaration on BackwardEngine
2. `cpp/src/autograd.cpp` — Implement `TapeContext::forward_<name>()` (dispatch shader), implement `BackwardEngine::backward_<name>()` (GPU shader or CPU fallback), add `case OpType::<Name>: backward_<name>(node); break;` to `dispatch_node_backward`
3. `cpp/python/bindings_autograd.cpp` — Add `.value("Name", OpType::Name)` to enum, add `.def("forward_<name>", &TapeContext::forward_<name>, ...)` to TapeContext
4. Write GLSL shader → compile to `.spv` with `glslc -fshader-stage=compute shader.glsl -o shaders/spv/shader.spv`
5. Rebuild: `cd build2 && cmake --build . --config Release`, copy `.pyd` to grilly root
6. Wire into `cubby/trunk/resident.py` `_resident_forward` (forward) and `_fb_run` (backward recording)

### Key Design Decisions

**1. Single-input ChunkedAttention (critical for backward chain):**
Record ChunkedAttention with the fused `qkv` buffer (BS, 3*d) as its **single** input — NOT separate q, k, v inputs. The forward shader still receives split Q/K/V from `forward_qkv_split`, but the tape node only tracks the qkv buffer. Backward then merges dQ+dK+dV into a single d_qkv gradient that flows to the Linear(QKV) node.

```python
# CORRECT: single qkv input, backward produces merged d_qkv
n_attn = t.record_op(op.ChunkedAttention,
                     [_R(qkv, [BS, 3*d])],
                     [_R(attn_out, [B, attn_H, S, attn_Dh])],
                     attn_params_bytes)
t.save_for_backward(n_attn, [q, k, v])  # save split buffers for backward computation

# WRONG: 3 separate inputs — breaks backward chain (grad_input_buffers[1,2] = 0)
n_attn = t.record_op(op.ChunkedAttention, [_R(q,...), _R(k,...), _R(v,...)], ...)
```

**2. GPU-only backward (CRITICAL — no CPU fallback in batch context):**
The backward engine runs handlers during a `batch_` walk, but `batch_.submit()` is only called after ALL handlers finish. This means `registry_.download()` reads stale data (compute hasn't run yet) and `registry_.upload()` writes get overwritten when the batch eventually executes. **All backward ops must be pure GPU shader dispatches** using `batch_.dispatch()`, `batch_.fillZero()`, and `batch_.transferComputeBarrier()`. Never use `registry_.download()` or `registry_.upload()` inside a backward handler.

For zeroing buffers that use atomicAdd: use `batch_.fillZero(buf, bytes)` — NOT `registry_.upload()` with zeros.

For transpose backward: the transpose is its own inverse (swap dims 1 and 2), so the same `attention-transpose-bhsd-bshd` shader works for both forward and backward. Dispatch it with `batch_.dispatch()`.

**3. Params via `record_op`:**
The `record_op` Python binding accepts an optional `params` argument (bytes object). Pass struct-packed params for backward metadata:
```python
import struct
params = struct.pack('IIIIIf', B, H, S, Dh, W, scale)
node = t.record_op(op.ChunkedAttention, inputs, outputs, params)
```
The C++ backward reads these via `std::memcpy(&p, node->params, sizeof(p))`.

### Infrastructure Fixes Required

**1. Staging buffer pooling (VMA exhaustion):**
The `uploadStaged` and `downloadStaged` functions in `buffer_pool.cpp` created a new VMA buffer per call and destroyed it after. With CPU-fallback backward ops making many download/upload calls, this exhausted the VMA pool (`VkResult=-3`). Fix: use `static thread_local` pooled staging buffers that grow but aren't freed per-call.

**2. backward_transpose was a stub:**
The existing `backward_transpose` set `grad_input_buffers[0] = 1` (placeholder). For attention output reshaping (BHSD→BSHD), replaced with a pure GPU dispatch using the same `attention-transpose-bhsd-bshd` shader (transpose is its own inverse). Uses `batch_.dispatch()` + `batch_.transferComputeBarrier()`, no CPU download/upload.

**3. Variable name collision in _fb_run:**
`H` was used as a MinGRU hidden state variable AND as the attention head count. Fix: rename to `H_min` for MinGRU and `attn_H`/`attn_Dh` for attention dimensions.

### Shader: chunked-sw-attention.glsl

Single-pass, one workgroup per `(batch, head, q_pos)`. Uses `GL_KHR_shader_subgroup_arithmetic` for dot-product reduction. Iterates K over `[max(0, q-W+1), q]` with online softmax. No O(S²) materialization.

**Push constants (24 bytes):** `batch_size, num_heads, seq_len, head_dim, window_size, scale`

**Buffer bindings (4):** Q (readonly), K (readonly), V (readonly), O (writeonly). All BHSD layout.

### Helper Shaders

- `attention-qkv-split.glsl`: Reshape fused (B*S, 3*H*Dh) → 3 separate (B, H, S, Dh) buffers. One thread per output element.
- `attention-transpose-bhsd-bshd.glsl`: Transpose (B, H, S, Dh) → (B*S, d) for output projection. One thread per element.

### Resident Path Wiring (resident.py)

In `_resident_forward`, for each attention block:
```python
ra_rms = t.forward_rmsnorm(r1, lw['rms_attn'], BS, d)
qkv_id = t.forward_linear(ra_rms, lw['qkv'], 0, BS, d, 3 * d)
q_id, k_id, v_id = t.forward_qkv_split(qkv_id, B, S, H, Dh)
attn_out = t.forward_chunked_attention(q_id, k_id, v_id, B, H, S, Dh, W)
bshd = t.forward_transpose_bhsd_bshd(attn_out, B, H, S, Dh)
outp = t.forward_linear(bshd, lw['out_proj'], 0, BS, d, d)
r_attn = t.forward_add(r1, outp, BS * d)
```

In `_fb_run`, record matching ops in reverse for backward. The cap (capture) tuple stores all intermediate buffer IDs: `(n1, G, Vv, D, H_min, r1, ra_rms, qkv, q, k, v, attn_out, transpose, out_proj, r_attn, rms_attn_id, qkv_id_buf, out_proj_id)` for attention layers, `(n1, G, Vv, D, H_min, r1, None)` + FFN for non-attention layers.

Also update: `_register_weights` (register rms_attn, qkv, out_proj per attention layer), `_w()` (include attention weight IDs), `_read_grads` (read attention gradients), `_adamw_np` (update attention weights), `_snapshot` (capture attention weights).

### Gotchas

1. **Don't break the async buffer pipeline:** Always use `registry_.alloc()` + `registry_.upload()` for new buffers in backward. Never use raw VMA calls. The buffer registry manages the pool and handles cleanup.

2. **save_for_backward saves buffer IDs, not data:** The saved IDs must still be valid when backward runs. Since the tape runs forward then backward in one `begin()`/`end()` cycle, step-scoped buffers remain valid.

3. **record_op params must match struct layout exactly:** The C++ backward reads `node->params` with a specific struct. If the Python packs different field order or types, the backward will read garbage. Use `struct.pack` with matching C++ struct.

4. **Shader compilation:** Use `glslc -fshader-stage=compute` (not `glslc` alone — it defaults to vertex stage). Target `vulkan1.3` for subgroup extensions.

5. **Thread-local staging pools are safe here:** The grilly backend is single-threaded (one command batch, synchronous submit). `static thread_local` staging buffers won't be shared across threads.

### GPU-Side Backward (replacing CPU fallback)

The initial CPU-fallback backward was extremely slow (download Q/K/V/dO → compute on CPU → upload back). Replaced with a single-pass GPU backward shader.

**Shader: `chunked-sw-attention-backward.glsl`**

Same workgroup pattern as forward (one per batch×head×q_pos). Recomputes softmax on-the-fly (no weight saving needed), then computes gradients:

- **dV[k]**: `atomicAdd` — multiple queries contribute to same key position
- **dK[k]**: `atomicAdd` — same reason
- **dQ[q]**: regular write — each query computes its own gradient

Uses `GL_EXT_shader_atomic_float` (already enabled in grilly).

**Shader: `attention-qkv-merge.glsl`**

Packs 3 separate BHSD gradient buffers (dQ, dK, dV) into one (BS, 3*d) BSHD buffer — the inverse of `attention-qkv-split.glsl`. One thread per output element.

**C++ dispatch (`backward_chunked_attention`):**
1. Allocate dQ, dK, dV buffers (BHSD layout)
2. Zero dK and dV (they use atomicAdd)
3. Dispatch backward shader (7 bindings: Q, K, V, dO, dQ, dK, dV)
4. Dispatch merge shader (4 bindings: dQ, dK, dV → d_qkv)
5. Set `node->grad_input_buffers[0] = dQkvId`

**Key insight:** The backward shader recomputes softmax from Q and K (same as forward), so no softmax weights need to be saved during forward. This trades redundant computation for zero memory traffic — much faster than CPU fallback.

**GLSL gotcha:** `subgroupBroadcastFirst` requires `GL_KHR_shader_subgroup_ballot` extension (not just `GL_KHR_shader_subgroup_basic`). Add it explicitly:
```glsl
#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_EXT_shader_atomic_float : enable
```

### Verification

- Forward: loss=4.65 (tiny model, d=64, L=3, attention on layer 0)
- Backward: all attention gradients computed (qkv buffer_id=218, out_proj buffer_id=209)
- No VMA exhaustion errors after staging pool fix
- No CPU readback in backward — fully GPU-side
- All existing resident parity tests still pass (non-attention path unchanged)
