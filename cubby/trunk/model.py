"""Cubby 0.0.0 trunk -- grilly-native, Python-tape autograd.

embed -> [RMSNorm -> MinGRU(scan) -> RMSNorm -> SwiGLU]xL -> RMSNorm -> tied head.

Everything is a tape Variable (grilly.nn.autograd), because the mixer (min_gru /
prefix_scan_causal) returns tape Variables -- so the whole trunk backprops in one
world. Ops whose tape-reduction support is uncertain (embedding gather, RMSNorm,
cross-entropy) are explicit GradFns with hand-derived backward, mirroring the
custom-GradFn pattern in grilly.nn.prefix_scan. Optimizer is numpy AdamW over the
tape params (the gc.adamw_update GPU kernel is the 0.0.1 swap).

Decay gate: min_gru uses x_scan = sigmoid(g)*tanh(v), a = 0.001 + 0.998*sigmoid(d)
(verified bit-close against grilly's own mingru parity test and a numpy reference).
"""
from __future__ import annotations

import numpy as np

from grilly.nn.autograd import Variable, GradFn, _ensure_variable
import grilly.nn.autograd as _ag
from grilly.nn.prefix_scan import min_gru

from cubby import trace
from cubby.trunk.ffn import _Linear, make_ffn
from cubby.trunk.gpu_linear import linear as _linear
from cubby.trunk.gpu_linear import slice_cols as _slice
from cubby.trunk.gpu_linear import _bridge as _bridge, _GPU as _GPU


def embedding(weight, ids) -> Variable:
    """Gather rows of `weight` (V, d) by integer `ids` (B, S) -> (B, S, d).
    Backward scatter-adds upstream grad back into the (tied) table."""
    w = _ensure_variable(weight)
    W = np.asarray(w.data, dtype=np.float32)
    ids = np.asarray(ids, dtype=np.int64)
    out = W[ids]
    if not (_ag._grad_enabled and getattr(w, "requires_grad", False)):
        return Variable(out, requires_grad=False)

    def backward_fn(grad_output):
        g = np.asarray(grad_output, dtype=np.float32)
        dW = np.zeros_like(W)
        np.add.at(dW, ids.reshape(-1), g.reshape(-1, g.shape[-1]))
        return (dW,)

    return Variable(out, requires_grad=True, grad_fn=GradFn("Embedding", backward_fn, [w]))


def rmsnorm(x, weight, eps: float = 1e-6) -> Variable:
    """RMSNorm: y = x / sqrt(mean(x^2, -1) + eps) * g. Explicit backward."""
    x_var = _ensure_variable(x)
    g_var = _ensure_variable(weight)
    X = np.asarray(x_var.data, dtype=np.float32)
    G = np.asarray(g_var.data, dtype=np.float32)
    ms = np.mean(X * X, axis=-1, keepdims=True)
    r = 1.0 / np.sqrt(ms + eps)                      # (..., 1)
    xhat = X * r
    out = xhat * G
    if not (_ag._grad_enabled and (x_var.requires_grad or g_var.requires_grad)):
        return Variable(out, requires_grad=False)
    n = X.shape[-1]

    def backward_fn(grad_output):
        dy = np.asarray(grad_output, dtype=np.float32)
        u = dy * G                                   # (..., d)
        s = np.sum(u * X, axis=-1, keepdims=True)    # (..., 1)
        dx = r * u - (r ** 3 * X / n) * s
        dG = np.sum(dy * xhat, axis=tuple(range(dy.ndim - 1)))   # -> (d,)
        return (dx.astype(np.float32), dG.astype(np.float32))

    return Variable(out, requires_grad=True,
                    grad_fn=GradFn("RMSNorm", backward_fn, [x_var, g_var]))


def cross_entropy(logits, targets) -> Variable:
    """Mean token cross-entropy. logits (B,S,V), targets int (B,S). On GPU the
    softmax/loss + (softmax-onehot)/N backward run via _bridge (offloads the
    V=65k softmax); numpy fallback otherwise. Backward: (softmax - onehot) / N."""
    z = _ensure_variable(logits)
    Z = np.asarray(z.data, dtype=np.float32)
    B, S, V = Z.shape
    t = np.asarray(targets, dtype=np.int64).reshape(-1)
    Zf = Z.reshape(-1, V)
    N = Zf.shape[0]
    if _GPU:
        per_row = np.asarray(_bridge.cross_entropy_loss(Zf, t), dtype=np.float32)
        loss = float(per_row.mean())
    else:
        m = Zf.max(axis=1, keepdims=True)
        sm = np.exp(Zf - m); sm /= sm.sum(axis=1, keepdims=True)
        loss = float(-np.log(sm[np.arange(N), t] + 1e-12).mean())
    if not (_ag._grad_enabled and z.requires_grad):
        return Variable(np.float32(loss), requires_grad=False)

    def backward_fn(grad_output):
        go = float(np.asarray(grad_output))
        if _GPU:
            sm = np.asarray(_bridge.softmax(Zf), dtype=np.float32)   # GPU softmax (offloads V=65k exp)
        else:
            mm = Zf.max(axis=1, keepdims=True)
            sm = np.exp(Zf - mm); sm /= sm.sum(axis=1, keepdims=True)
        # _bridge.cross_entropy_backward was verified to use a different convention
        # (grad diff 3.11 vs (sm-onehot)/N), so finalize in numpy -- cheap vs exp.
        d = sm.copy()
        d[np.arange(N), t] -= 1.0
        d *= (go / N)
        return (d.reshape(B, S, V).astype(np.float32),)

    return Variable(np.float32(loss), requires_grad=True,
                    grad_fn=GradFn("CrossEntropy", backward_fn, [z]))


def sampled_cross_entropy(hidden, embed_weight, targets, n_samples: int = 1024,
                          vocab_size: int | None = None) -> Variable:
    """Importance-sampled CE: avoid materialising the full (N, V) logit tensor.

    For each token position, compute logits ONLY for the true target + K random
    negatives drawn uniformly from the vocab.  CE is computed over the (K+1)
    subset.  With uniform sampling the IS correction is a constant that cancels
    in the softmax, so the gradient direction is unbiased — this is exactly
    Bengio & Senecal (2008) with q = uniform.

    Inputs
      hidden  : (N, d) tape Variable, the post-RMSNorm hidden states
      embed_weight : (V, d) tape Variable, the language embedding table
      targets : int array (N,), the true target token IDs in [0, V)
      n_samples : number of negative samples per position

    Returns a scalar tape Variable wrapping the per-position mean CE loss.
    """
    h_var = _ensure_variable(hidden)
    w_var = _ensure_variable(embed_weight)
    H = np.asarray(h_var.data, dtype=np.float32)          # (N, d)
    W = np.asarray(w_var.data, dtype=np.float32)          # (V, d)
    V = W.shape[0]
    vocab_size = vocab_size or V
    t = np.asarray(targets, dtype=np.int64).reshape(-1)    # (N,)
    N = H.shape[0]
    K = n_samples

    # ── draw negatives + build subset indices ──────────────────────────
    neg = np.random.randint(0, vocab_size, size=(N, K)).astype(np.int64)   # (N, K)
    # first column is always the true target (position 0 in the subset)
    sub_ids = np.concatenate([t[:, None], neg], axis=1)                    # (N, K+1)
    # clamp into [0, V) in case vocab_size > V (shouldn't happen but be safe)
    sub_ids = np.clip(sub_ids, 0, V - 1)

    # ── subset logits: dot product of h with sampled weight rows ───────
    W_sub = W[sub_ids]                          # (N, K+1, d)
    logits_sub = np.einsum("nd,nkd->nk", H, W_sub)   # (N, K+1)
    # target is always index 0 in each row
    targets_sub = np.zeros(N, dtype=np.int64)

    # ── forward: CE over (K+1) classes ─────────────────────────────────
    m = logits_sub.max(axis=1, keepdims=True)
    sm = np.exp(logits_sub - m);  sm /= sm.sum(axis=1, keepdims=True)
    loss = float(-np.log(sm[np.arange(N), 0] + 1e-12).mean())

    if not (_ag._grad_enabled and (h_var.requires_grad or w_var.requires_grad)):
        return Variable(np.float32(loss), requires_grad=False)

    # ── backward ───────────────────────────────────────────────────────
    # dL/dlogits_sub = (softmax - onehot_at_0) / N  — standard CE grad on subset
    d_logits = sm.copy()
    d_logits[:, 0] -= 1.0
    d_logits /= N                                            # (N, K+1)

    def backward_fn(grad_output):
        go = float(np.asarray(grad_output))
        d_logits_g = d_logits * go                           # (N, K+1)

        # grad w.r.t. hidden: scatter-add over the sampled K+1 weight rows
        # dL/dh[n] = sum_k d_logits_g[n,k] * W_sub[n,k,:]
        dh = np.einsum("nk,nkd->nd", d_logits_g, W_sub)     # (N, d)

        # grad w.r.t. weight rows: scatter-add into full (V, d)
        dW = np.zeros_like(W)                                # (V, d)
        # dL/dW_sub[n,k,:] = d_logits_g[n,k] * h[n,:]
        dW_sub = d_logits_g[:, :, None] * H[:, None, :]     # (N, K+1, d)
        # scatter-add: add each row of dW_sub to dW[sub_ids]
        np.add.at(dW, sub_ids.reshape(-1), dW_sub.reshape(-1, d))

        return (dh.astype(np.float32), dW.astype(np.float32))

    return Variable(np.float32(loss), requires_grad=True,
                    grad_fn=GradFn("SampledCE", backward_fn, [h_var, w_var]))


class MinGRUMixer:
    """Tape MinGRU: g/v/d = Linear(x); h = min_gru(g, v, d). Tape _Linear weights
    (not nn.Linear, whose C++ Parameters the tape would treat as constants)."""

    def __init__(self, d_model: int):
        self.d = d_model
        # Fused g/v/d projection: one (3d, d) matmul instead of three (d, d) --
        # all three read the same input, so concatenating the weights collapses
        # 3 GPU dispatches to 1 (the measured bottleneck is per-dispatch overhead).
        self.proj = _Linear(d_model, 3 * d_model)

    def parameters(self):
        yield from self.proj.parameters()

    def __call__(self, x):
        d = self.d
        gvd = self.proj(x)
        return min_gru(_slice(gvd, 0, d), _slice(gvd, d, 2 * d), _slice(gvd, 2 * d, 3 * d))


class Block:
    """Pre-norm: x = x + mix(norm1(x)); x = x + ffn(norm2(x))."""

    def __init__(self, cfg, idx: int):
        d = cfg.d_model
        self.idx = idx
        self.eps = cfg.rmsnorm_eps
        self.n1 = Variable(np.ones((d,), dtype=np.float32), requires_grad=True)
        self.mix = MinGRUMixer(d)
        self.n2 = Variable(np.ones((d,), dtype=np.float32), requires_grad=True)
        self.ffn = make_ffn(cfg.ffn_type, d, cfg.d_ffn, name=f"ffn{idx}")
        # enable_residual_scale (config 0.0.1, L>=18): GPT-2-style scaled init of
        # each residual branch's OUTPUT projection by 1/sqrt(2L). Keeps the
        # pre-norm residual stream bounded across deep stacks (without it L=18
        # diverges to NaN). Pure init -> backend-agnostic (the resident path reads
        # the same scaled weights, so forward/grad parity is preserved).
        if getattr(cfg, "enable_residual_scale", False):
            s = np.float32(1.0 / np.sqrt(2.0 * cfg.n_layers))
            self.mix.proj.weight.data = (np.asarray(self.mix.proj.weight.data, np.float32) * s)
            self.ffn.down.weight.data = (np.asarray(self.ffn.down.weight.data, np.float32) * s)

    def parameters(self):
        yield self.n1
        yield from self.mix.parameters()
        yield self.n2
        yield from self.ffn.parameters()

    def __call__(self, x):
        x = x + self.mix(rmsnorm(x, self.n1, self.eps))
        x = x + self.ffn(rmsnorm(x, self.n2, self.eps))
        trace.probe(f"block{self.idx}", np.asarray(x.data), topology=f"layer:{self.idx}")
        return x


class CubbyLM:
    """Trunk model.

    0.0.0 substrate (enable_dual_head=False): embed -> blocks -> final RMSNorm ->
    tied linear head.

    Dual-head architecture (enable_dual_head=True): the single trunk projects to
    two separate output heads gated by a learned router:
      - Language head: RMSNorm_lang -> h @ embed_lang[:V_lang].T
      - AST head: RMSNorm_ast -> h @ embed_ast[:V_ast].T
    The router (d -> 2, softmax) weights each head per token position. Language
    and AST embeddings are disjoint slices of the vocab — the tokenizer defines
    the boundary (ast_start_id). Sampled softmax (importance sampling) can be
    swapped in for the language head loss during training.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        d = cfg.d_model
        self.dual_head = getattr(cfg, "enable_dual_head", False)

        if self.dual_head:
            Vlang = getattr(cfg, "vocab_size", cfg.total_vocab)  # language region
            Vast = cfg.total_vocab - Vlang                         # AST region
            # language embed + output (tied)
            self.embed_lang = Variable(
                (np.random.randn(Vlang, d) * cfg.embed_init_std).astype(np.float32),
                requires_grad=True)
            self.final_lang = Variable(np.ones((d,), dtype=np.float32), requires_grad=True)
            # AST embed + output (tied)
            self.embed_ast = Variable(
                (np.random.randn(max(Vast, 1), d) * cfg.embed_init_std).astype(np.float32),
                requires_grad=True) if Vast > 0 else None
            self.final_ast = Variable(
                np.ones((d,), dtype=np.float32), requires_grad=True) if Vast > 0 else None
            # router: d -> 2 (language vs AST)
            self.router = Variable(
                (np.random.randn(cfg.router_d, d) * 0.01).astype(np.float32),
                requires_grad=True)
            self.Vlang = Vlang
            self.Vast = Vast
            # kept for compat (used by old code paths that read model.embed)
            self.embed = self.embed_lang
            self.final = self.final_lang
        else:
            V, d = cfg.total_vocab, d
            self.embed = Variable(
                (np.random.randn(V, d) * cfg.embed_init_std).astype(np.float32),
                requires_grad=True)
            self.blocks = [Block(cfg, i) for i in range(cfg.n_layers)]
            self.final = Variable(np.ones((d,), dtype=np.float32), requires_grad=True)
            self.Vlang = V
            self.Vast = 0

        self.blocks = [Block(cfg, i) for i in range(cfg.n_layers)]

    def parameters(self):
        if self.dual_head:
            yield self.embed_lang
            if self.embed_ast is not None:
                yield self.embed_ast
            yield self.final_lang
            if self.final_ast is not None:
                yield self.final_ast
            yield self.router
        else:
            yield self.embed
        for b in self.blocks:
            yield from b.parameters()
        if not self.dual_head:
            yield self.final

    def __call__(self, ids):
        """Forward pass. Returns logits tensor(s).

        Single-head: returns (B, S, V) logits array as a tape Variable.
        Dual-head: returns dict with keys 'lang_logits' (B,S,V_lang) and
        'ast_logits' (B,S,V_ast) and 'router_weights' (B,S,2).
        """
        if self.dual_head:
            # gather from combined embed: token IDs may span both regions
            # build a virtual full embed by concatenating lang + ast tables
            if self.embed_ast is not None and self.Vast > 0:
                full_embed = np.concatenate([
                    np.asarray(self.embed_lang.data, np.float32),
                    np.asarray(self.embed_ast.data, np.float32)], axis=0)
                full_var = Variable(full_embed, requires_grad=False)
            else:
                full_var = self.embed_lang
            x = embedding(full_var, ids)
        else:
            x = embedding(self.embed, ids)

        with trace.scope("trunk"):
            for b in self.blocks:
                x = b(x)

        if self.dual_head:
            # router: h @ router.T -> (B, S, 2) -> softmax
            h = np.asarray(x.data, np.float32)
            B, S, d = h.shape
            h_flat = h.reshape(-1, d)
            router_logits = h_flat @ np.asarray(self.router.data, np.float32).T  # (BS, 2)
            rl = router_logits - router_logits.max(axis=1, keepdims=True)
            rw = np.exp(rl); rw /= rw.sum(axis=1, keepdims=True)
            router_weights = rw.reshape(B, S, 2)

            # language head
            h_lang = rmsnorm(x, self.final_lang, self.cfg.rmsnorm_eps)
            lang_logits = _linear(h_lang, self.embed_lang)  # (B, S, V_lang)

            # AST head
            ast_logits = None
            if self.embed_ast is not None and self.Vast > 0:
                h_ast = rmsnorm(x, self.final_ast, self.cfg.rmsnorm_eps)
                ast_logits = _linear(h_ast, self.embed_ast)  # (B, S, V_ast)

            return {"lang_logits": lang_logits,
                    "ast_logits": ast_logits,
                    "router_weights": Variable(router_weights.astype(np.float32),
                                               requires_grad=False)}
        else:
            x = rmsnorm(x, self.final, self.cfg.rmsnorm_eps)
            logits = _linear(x, self.embed)  # tied head
            return logits

    def loss(self, ids, targets):
        """Compute loss. Routes to language head, AST head, or both."""
        out = self(ids)
        if not self.dual_head:
            if getattr(self.cfg, "enable_sampled_softmax", False):
                # sampled IS: avoid full (N, V) logits
                h = out  # not logits — but single-head returns logits
                # actually for sampled, we need hidden states, not logits
                # the single-head path needs to be re-entered — handled in train loop
                pass
            return cross_entropy(out, targets)

        # ── dual-head loss ─────────────────────────────────────────────
        lang_logits = out["lang_logits"]
        ast_logits = out["ast_logits"]
        rw = np.asarray(out["router_weights"].data, np.float32)  # (B, S, 2)

        targets_arr = np.asarray(targets, dtype=np.int64)
        B, S = targets_arr.shape
        Vlang = self.Vlang

        # classify each target: lang (id < Vlang) or AST (id >= Vlang)
        is_lang = (targets_arr < Vlang)           # (B, S) bool
        is_ast = ~is_lang

        total_loss = 0.0
        n_lang = int(is_lang.sum())
        n_ast = int(is_ast.sum())

        # language CE (for language targets only)
        if n_lang > 0:
            Ll = np.asarray(lang_logits.data, np.float32).reshape(-1, self.Vlang)
            tl = targets_arr.reshape(-1)
            lang_mask = is_lang.reshape(-1)
            Ll_sub = Ll[lang_mask]
            tl_sub = tl[lang_mask]
            if getattr(self.cfg, "enable_sampled_softmax", False):
                # sampled IS over language head
                h_lang_flat = None  # need hidden states — caller should use
                # fall back to full CE for now when sampled is requested here
                m = Ll_sub.max(axis=1, keepdims=True)
                sm = np.exp(Ll_sub - m); sm /= sm.sum(axis=1, keepdims=True)
                lang_loss = float(-np.log(sm[np.arange(len(tl_sub)), tl_sub] + 1e-12).mean())
                # weight by router's language channel
                w_lang = rw.reshape(-1, 2)[lang_mask, 0].mean()
            else:
                m = Ll_sub.max(axis=1, keepdims=True)
                sm = np.exp(Ll_sub - m); sm /= sm.sum(axis=1, keepdims=True)
                lang_loss = float(-np.log(sm[np.arange(len(tl_sub)), tl_sub] + 1e-12).mean())
                w_lang = rw.reshape(-1, 2)[lang_mask, 0].mean()
            total_loss += w_lang * lang_loss

        # AST CE (for AST targets only)
        if n_ast > 0 and ast_logits is not None:
            La = np.asarray(ast_logits.data, np.float32).reshape(-1, self.Vast)
            ta = targets_arr.reshape(-1) - Vlang  # shift to [0, Vast)
            ast_mask = is_ast.reshape(-1)
            La_sub = La[ast_mask]
            ta_sub = ta[ast_mask]
            m = La_sub.max(axis=1, keepdims=True)
            sm = np.exp(La_sub - m); sm /= sm.sum(axis=1, keepdims=True)
            ast_loss = float(-np.log(sm[np.arange(len(ta_sub)), ta_sub] + 1e-12).mean())
            w_ast = rw.reshape(-1, 2)[ast_mask, 1].mean()
            total_loss += w_ast * ast_loss

        # if no AST targets, just do language CE on all (router defaults to lang)
        if n_ast == 0 and n_lang > 0:
            total_loss = float(np.asarray(
                cross_entropy(lang_logits, targets).data))

        return Variable(np.float32(total_loss), requires_grad=False)

    def generate(self, prompt_ids, max_new_tokens=128, temperature=1.0):
        ids = list(np.asarray(prompt_ids, dtype=np.int64).reshape(-1))
        for _ in range(max_new_tokens):
            arr = np.asarray(ids, dtype=np.int64)[None, :]
            with _ag.no_grad():
                logits = np.asarray(self(arr).data)[0, -1]
            if temperature <= 0:
                nxt = int(logits.argmax())
            else:
                z = logits / temperature
                p = np.exp(z - z.max()); p /= p.sum()
                nxt = int(np.random.choice(len(p), p=p))
            ids.append(nxt)
        return ids


class AdamW:
    """Numpy AdamW over tape Variables (.data / .grad). 0.0.0 optimizer; the GPU
    gc.adamw_update kernel is the 0.0.1 swap for the scaled run."""

    def __init__(self, params, lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01):
        self.params = list(params)
        self.lr, self.eps, self.wd = lr, eps, weight_decay
        self.b1, self.b2 = betas
        self.m = [np.zeros_like(np.asarray(p.data, np.float32)) for p in self.params]
        self.v = [np.zeros_like(np.asarray(p.data, np.float32)) for p in self.params]
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        c1, c2 = 1.0 - self.b1, 1.0 - self.b2
        lr, eps, wd = self.lr, self.eps, self.wd
        for i, p in enumerate(self.params):
            g = getattr(p, "grad", None)
            if g is None:
                continue
            g = np.asarray(g, dtype=np.float32)
            # In-place updates: measured ~293ms vs ~489ms allocating (and vs
            # ~429ms GPU adamw_update, which loses to per-param round-trips).
            m_, v_ = self.m[i], self.v[i]
            m_ *= self.b1; m_ += c1 * g
            v_ *= self.b2; v_ += c2 * (g * g)
            denom = np.sqrt(v_ / bc2); denom += eps
            w = np.asarray(p.data, dtype=np.float32)
            w *= (1.0 - lr * wd)             # decoupled weight decay (AdamW)
            w -= lr * (m_ / bc1) / denom
            p.data = w


def param_count(model) -> int:
    return int(sum(np.asarray(p.data).size for p in model.parameters()))
