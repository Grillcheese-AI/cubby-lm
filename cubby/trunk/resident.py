"""Resident (GPU-resident, single-tape) execution path for the Cubby trunk.

This lives ALONGSIDE `model.py` (the numpy / Python-tape reference), per the port
discipline: it runs CubbyLM's architecture fully on grilly's resident TapeContext
(register_weight + forward_* + record_op + adamw_update -- the stack validated in
grilly/experimental/resident_train/train_trunk_lm.py), and is gated on FORWARD
PARITY then GRADIENT PARITY vs model.py before it can become the default.

It reads parameters straight out of a built `CubbyLM` (the Variables in model.py),
so the two share identical weights -- the only thing under test is whether the
resident kernels reproduce the numpy trunk.

Two cubby-specific details vs the train_trunk_lm reference:
  - the MinGRU projection is FUSED (one (3d,d) weight) and sliced G|V|D; the
    resident path has no Slice op, so the fused weight is split into three (d,d)
    blocks (rows [0:d]|[d:2d]|[2d:3d]) and run as three forward_linears
    (mathematically identical).
  - cubby SwiGLU is down(silu(gu[:f]) * gu[f:2f]) -- silu on the FIRST half --
    but the resident forward_swiglu computes x1*silu(x2) (silu on the SECOND
    half). So the gate_up weight's two row-blocks are SWAPPED at registration so
    the resident output is up*silu(gate) == silu(gate)*up.
"""
from __future__ import annotations

import sys

import numpy as np

# grilly_core.<abi>.pyd lives at the grilly repo root (copied there by rebuild.ps1);
# the editable install exposes `import grilly` but not the raw extension, so put the
# repo root on the path for the resident TapeContext API.
_GRILLY_ROOT = r"C:\Users\grill\Documents\GitHub\grilly"
if _GRILLY_ROOT not in sys.path:
    sys.path.insert(0, _GRILLY_ROOT)
import grilly_core as gc

# cubby-lm root on the path so `from cubby import trace` works when this file is
# run as a script (as a package import it's already importable).
_CUBBY_ROOT = r"C:\Users\grill\Documents\GitHub\cubby-lm"
if _CUBBY_ROOT not in sys.path:
    sys.path.insert(0, _CUBBY_ROOT)
from cubby import trace

_SPV = _GRILLY_ROOT + r"\shaders\spv"


def make_device():
    dev = gc.Device(); dev.load_shaders(_SPV); return dev


def _R(buf, shape, rg=True):
    r = gc.TensorRef(); r.buffer_id = buf; r.set_shape(shape); r.requires_grad = rg; return r


def _f32(a):
    return np.ascontiguousarray(np.asarray(a, dtype=np.float32))


def force_numpy_reference():
    """Make the model.py path pure-numpy so the resident grilly_core.Device() is
    the ONLY Vulkan context. Two contexts (the resident one + model.py's
    grilly.nn.autograd / _bridge GPU backward) nondeterministically corrupt each
    other. Forces gpu_linear+model off the bridge AND short-circuits autograd's
    LAZY gpu-backward singleton so model.loss().backward() never inits a 2nd
    device (e.g. MinGRU backward). Call before building/using any CubbyLM here."""
    import cubby.trunk.gpu_linear as _gl
    import cubby.trunk.model as _m
    import grilly.nn.autograd as _ag
    _gl._GPU = False
    _m._GPU = False
    _ag._gpu_backward_ops_checked = True       # pretend "already checked"
    _ag._gpu_backward_ops_cache = None         # -> no GPU backend -> CPU/numpy


class ResidentTrunk:
    """Resident forward (and, incrementally, backward/optimizer) for a CubbyLM.

    Built from the model's parameter Variables. Weights are registered ONCE as
    persistent resident buffers; forward() re-runs each call over a fresh batch.
    """

    def __init__(self, model, dev=None):
        cfg = model.cfg
        self.d = int(cfg.d_model)
        self.L = int(cfg.n_layers)
        self.dff = int(cfg.d_ffn)
        self.dual_head = getattr(cfg, 'enable_dual_head', False) and \
                         hasattr(model, 'embed_ast') and model.embed_ast is not None
        self.Vlang = int(cfg.vocab_size) if self.dual_head else int(cfg.total_vocab)
        self.Vast = int(cfg.total_vocab) - self.Vlang if self.dual_head else 0
        self.V = int(cfg.total_vocab)  # total vocab (Vlang + Vast)
        self.model = model
        self.dev = dev or make_device()
        self.t = gc.TapeContext(self.dev)
        self._register_weights()

    _WKEYS = ('n1', 'WG', 'WV', 'WD', 'n2', 'gate_up', 'down')

    # --- weight registration (persistent resident + Adam moments) ---------------
    def _register_weights(self):
        """ALL weights -- incl. the tied embedding E -- are PERSISTENT resident,
        each with persistent Adam m/v, so resident AdamW updates them in place.
        E is used as both the embedding table and the tied head weight; its grad
        (head-weight grad + embedding scatter) is assembled fully on-GPU in
        train_step (embedding_scatter_add), so E joins self.opt with no host path.

        Dual-head: register combined embedding E_combined = [E_lang; E_ast] as
        a single resident weight, plus router (d -> 2) as a separate weight.

        Attention: when enable_attention is set, register per-layer rms_attn,
        qkv (d->3d fused), and out_proj (d->d) weights. has_attn flags which
        layers include attention (every attn_every_n-th from block 0)."""
        t, d, dff = self.t, self.d, self.dff
        m = self.model
        cfg = getattr(m, 'cfg', None)

        def regw(arr):
            a = _f32(arr)
            return dict(w=t.register_weight(a.copy()),
                        m=t.register_weight(np.zeros(a.size, np.float32)),
                        v=t.register_weight(np.zeros(a.size, np.float32)), n=int(a.size))

        if self.dual_head:
            # Combined embedding table (V_total, d) = concat of E_lang + E_ast
            E_combined = np.concatenate([
                np.asarray(m.embed_lang.data, np.float32),
                np.asarray(m.embed_ast.data, np.float32)
            ], axis=0)
            self.E = regw(E_combined)
            # Router: linear (d -> 2)
            self.router = regw(np.asarray(m.router.data, np.float32))
        else:
            self.E = regw(m.embed.data)                                  # (V, d) tied, resident
            self.router = None
        self.final = regw(m.final.data)

        # Attention config
        self.has_attn = bool(getattr(cfg, 'enable_attention', False))
        self.attn_every_n = int(getattr(cfg, 'attn_every_n', 3)) if self.has_attn else 0
        self.attn_n_heads = int(getattr(cfg, 'attn_heads', getattr(cfg, 'attn_n_heads', 8)))
        self.attn_window = int(getattr(cfg, 'attn_window', 1024))
        self.attn_d_head = self.d // self.attn_n_heads

        self.layers = []
        for li, b in enumerate(m.blocks):
            W = _f32(b.mix.proj.weight.data); gu = _f32(b.ffn.gate_up.weight.data)
            layer = dict(
                n1=regw(b.n1.data), WG=regw(W[0:d]), WV=regw(W[d:2 * d]), WD=regw(W[2 * d:3 * d]),
                n2=regw(b.n2.data),
                gate_up=regw(np.concatenate([gu[dff:2 * dff], gu[0:dff]], axis=0)),
                down=regw(b.ffn.down.weight.data),
                has_attn=False,
            )
            if self.has_attn and hasattr(b, 'attn') and getattr(b, 'has_attn', False):
                layer['has_attn'] = True
                layer['rms_attn'] = regw(b.rms_attn.data)
                layer['qkv'] = regw(b.attn.qkv.weight.data)          # (3d, d)
                layer['out_proj'] = regw(b.attn.out_proj.weight.data) # (d, d)
            self.layers.append(layer)
        # flat list for the optimizer sweep + GPU grad-norm (E included)
        head_weights = [self.E, self.final]
        # Router is not in the tape backward graph (loss weighting is done in
        # numpy), so it doesn't get a gradient from t.backward(). We exclude
        # it from self.opt to avoid the sum_squares crash. Router weights
        # will be updated separately if needed.
        layer_weights = []
        for li in range(self.L):
            for k in self._WKEYS:
                layer_weights.append(self.layers[li][k])
            if self.layers[li].get('has_attn', False):
                layer_weights.append(self.layers[li]['rms_attn'])
                layer_weights.append(self.layers[li]['qkv'])
                layer_weights.append(self.layers[li]['out_proj'])
        self.opt = head_weights + layer_weights

    def _w(self):
        """Extract weight buffer IDs per layer for the forward pass."""
        w = dict(E=self.E['w'], final=self.final['w'],
                 layers=[
                     {**{k: self.layers[li][k]['w'] for k in self._WKEYS},
                      'has_attn': self.layers[li].get('has_attn', False),
                      **({'rms_attn': self.layers[li]['rms_attn']['w'],
                           'qkv': self.layers[li]['qkv']['w'],
                           'out_proj': self.layers[li]['out_proj']['w']}
                         if self.layers[li].get('has_attn', False) else {})}
                     for li in range(self.L)])
        if self.router is not None:
            w['router'] = self.router['w']
        return w

    # --- inference forward (+ optional cubby.trace) -----------------------------
    def forward_ids(self, ids):
        """Resident forward over ids (B, S) -> (logits_buffer_id, B, S). Records
        nothing. Emits cubby.trace per block ONLY when a tracer is active (reads
        block outputs back) so production (trace OFF) pays zero overhead."""
        t = self.t; t.begin()
        emb, logits, cap, nf, B, S, _router = _resident_forward(
            t, self._w(), ids, self.d, self.dff, self.V, self.L)
        tr = trace.current()
        if tr.level > trace.Level.OFF:
            BS = B * S
            with tr.scope("trunk"):
                for li, c in enumerate(cap):
                    r2 = t.read_buffer(c[-1], [BS, self.d]).reshape(B, S, self.d)
                    tr.probe("block%d" % li, r2, topology="layer:%d" % li)
        return logits, B, S

    def logits(self, ids):
        lid, B, S = self.forward_ids(ids)
        return self.t.read_buffer(lid, [B * S, self.V]).reshape(B, S, self.V)

    # --- resident forward+backward -> grads mapped back to model.py params ------
    def grads(self, ids, targets):
        """Resident forward+backward over the persistent weights (E step-scoped).
        Returns grads in model.py layout (see _read_grads)."""
        t = self.t; t.begin()
        w = self._w()
        attn_H = self.attn_n_heads if self.has_attn else 0
        emb, logits_np, BS, _, _, _, _, _ = _fb_run(
            t, w, ids, targets, self.d, self.dff, self.V, self.L,
            attn_heads=attn_H, attn_window=self.attn_window)
        return _read_grads(t, w, emb, ids, BS, self.d, self.dff, self.V, self.L)

    # --- one resident training step (fully on-GPU: scatter + norm + AdamW) -------
    def train_step(self, ids, targets, step, lr=3e-4, betas=(0.9, 0.95), eps=1e-8,
                   wd=0.01, max_grad_norm=1.0, n_samples=1024, use_sampled=False):
        """Forward+backward then resident AdamW with GLOBAL grad-norm clipping --
        ALL on-GPU, no grad readback. ALL params (incl. the tied embedding E)
        update resident: E's grad = tied-head weight grad + embedding scatter,
        assembled in place by embedding_scatter_add; the global norm is a GPU
        reduction. clip + mean-CE 1/B are folded into the kernel grad_scale (scaled
        BEFORE Adam m/v, since Adam normalizes by sqrt(v)). Returns (mean-CE loss,
        pre-clip global grad-norm, skipped). Skips the step on a non-finite norm.
        
        Dual-head: computes CE separately for language and AST token subsets,
        weighted by the mean router probability for each head.
        Sampled softmax: if `use_sampled`, draws K=n_samples negatives per
        position and computes CE only over the K+1 subset (Bengio & Senecal 2008).
        """
        d, V = self.d, self.V
        ids = np.asarray(ids, dtype=np.int64); B, S = ids.shape; BS = B * S
        t = self.t; t.begin()
        w = self._w()
        Vlang = self.Vlang if self.dual_head else None
        Vast = self.Vast if self.dual_head else None
        attn_H = self.attn_n_heads if self.has_attn else 0
        emb, logits_np, _, router_logits_np, _, _, _, _ = _fb_run(
            t, w, ids, targets, d, self.dff, V, self.L, Vlang, Vast,
            attn_heads=attn_H, attn_window=self.attn_window)
        tgt = np.asarray(targets, np.int64).reshape(-1)

        if self.dual_head and router_logits_np is not None:
            loss, rw = _dual_head_ce(logits_np, tgt, router_logits_np, Vlang, Vast,
                                     n_samples=n_samples, use_sampled=use_sampled,
                                     d_model=d)
        else:
            if use_sampled:
                loss = _sampled_ce(logits_np, tgt, V, n_samples=n_samples, d=d)
            else:
                loss = _ce(logits_np, tgt)
            rw = None

        # E grad: scatter the embedding-output grad INTO E's grad buffer (which
        # already holds the tied-head weight grad) -- fully resident, in place.
        ids_u32 = t.register_input_u32(ids.reshape(-1).astype(np.uint32))
        t.embedding_scatter_add(t.get_grad_buffer(emb), ids_u32,
                                t.get_grad_buffer(self.E['w']), BS, d)

        # global L2 norm over MEAN grads (raw/BS), ALL params on-GPU (one scalar).
        sq = t.sum_squares([t.get_grad_buffer(p['w']) for p in self.opt],
                           [p['n'] for p in self.opt])
        gnorm = float(np.sqrt(float(sq))) / BS
        if not np.isfinite(gnorm):
            return loss, gnorm, True                                # poisoned grad -> SKIP
        clip = min(1.0, max_grad_norm / (gnorm + 1e-6)) if max_grad_norm else 1.0
        gscale = clip / BS                                          # clip + mean-CE, in-kernel

        b1, b2 = betas; b1t, b2t = b1 ** step, b2 ** step
        t.forward_begin()
        for p in self.opt:                                          # resident AdamW, one batch
            t.adamw_update(p['w'], t.get_grad_buffer(p['w']), p['m'], p['v'], p['n'],
                           lr, b1, b2, eps, wd, b1t, b2t, False, gscale)
        t.forward_submit()
        return loss, gnorm, False

    def train_step_sft(self, ids, targets, mask, step, lr=3e-4, betas=(0.9, 0.95),
                       eps=1e-8, wd=0.01, max_grad_norm=1.0):
        """One resident step with PROMPT-MASKED (completion-only) loss: CE is
        computed and backpropped ONLY where mask==1 (the program tokens), so the
        model learns p(program | instruction) and never wastes capacity generating
        the NL instruction. Custom dlogits are seeded at the head node (see
        _fb_run dlogits_fn). Returns (masked-mean CE, pre-clip gnorm, skipped)."""
        d, V = self.d, self.V
        ids = np.asarray(ids, dtype=np.int64); B, S = ids.shape; BS = B * S
        tgt = np.asarray(targets, np.int64).reshape(-1)
        m = np.asarray(mask, np.float32).reshape(-1)
        n_comp = float(m.sum())
        if n_comp < 1.0:
            return 0.0, 0.0, True
        box = {}
        def dlogits_fn(logits_np):
            z = logits_np - logits_np.max(axis=1, keepdims=True)
            sm = np.exp(z); sm /= sm.sum(axis=1, keepdims=True)
            nll = -np.log(sm[np.arange(BS), tgt] + 1e-12)
            box['loss'] = float((nll * m).sum() / n_comp)        # masked-mean CE
            sm[np.arange(BS), tgt] -= 1.0                        # softmax - onehot
            sm *= m[:, None]                                     # zero prompt/pad (raw sum)
            return sm
        t = self.t; t.begin()
        w = self._w()
        attn_H = self.attn_n_heads if self.has_attn else 0
        emb, _, _, _, _, _, _, _ = _fb_run(
            t, w, ids, targets, d, self.dff, V, self.L,
            attn_heads=attn_H, attn_window=self.attn_window, dlogits_fn=dlogits_fn)
        loss = box.get('loss', 0.0)
        ids_u32 = t.register_input_u32(ids.reshape(-1).astype(np.uint32))
        t.embedding_scatter_add(t.get_grad_buffer(emb), ids_u32,
                                t.get_grad_buffer(self.E['w']), BS, d)
        sq = t.sum_squares([t.get_grad_buffer(p['w']) for p in self.opt],
                           [p['n'] for p in self.opt])
        gnorm = float(np.sqrt(float(sq))) / n_comp               # raw grads are sum/comp
        if not np.isfinite(gnorm):
            return loss, gnorm, True
        clip = min(1.0, max_grad_norm / (gnorm + 1e-6)) if max_grad_norm else 1.0
        gscale = clip / n_comp                                   # mean over COMPLETION tokens
        b1, b2 = betas; b1t, b2t = b1 ** step, b2 ** step
        t.forward_begin()
        for p in self.opt:
            t.adamw_update(p['w'], t.get_grad_buffer(p['w']), p['m'], p['v'], p['n'],
                           lr, b1, b2, eps, wd, b1t, b2t, False, gscale)
        t.forward_submit()
        return loss, gnorm, False

    # --- autoregressive generation (resident forward) ---------------------------
    def generate(self, prompt_ids, max_new_tokens=80, temperature=0.8, seed=0,
                 rep_penalty=1.0):
        rng = np.random.default_rng(seed)
        ids = list(np.asarray(prompt_ids, dtype=np.int64).reshape(-1))
        n_prompt = len(ids)
        for _ in range(max_new_tokens):
            lg = self.logits(np.asarray(ids, dtype=np.int64)[None, :])[0, -1]
            if not np.isfinite(lg).all():
                break                                       # diverged trunk -> stop, don't crash
            if rep_penalty != 1.0 and len(ids) > n_prompt:  # CTRL-style repetition penalty
                gen = np.unique(np.asarray(ids[n_prompt:], dtype=np.int64))
                pos = lg[gen] > 0
                lg[gen[pos]] /= rep_penalty
                lg[gen[~pos]] *= rep_penalty
            if temperature <= 0:
                nxt = int(lg.argmax())
            else:
                z = lg / temperature; z -= z.max(); p = np.exp(z); p /= p.sum()
                nxt = int(rng.choice(len(p), p=p))
            ids.append(nxt)
        return ids


def _register_layer_ids(t, P, d, dff, L, requires_grad=True):
    """Register a numpy param dict (model.py layout) as resident buffers, applying
    the gvd split and the swiglu half-swap. Returns (E_id, final_id, layers)."""
    reg = (lambda a: t.register_weight(_f32(a))) if False else (lambda a: t.register_input(_f32(a), requires_grad))
    E = reg(P['embed']); final = reg(P['final'])
    layers = []
    for li in range(L):
        Wp = _f32(P['proj'][li]); gu = _f32(P['gate_up'][li])
        layer = dict(
            n1=reg(P['n1'][li]), WG=reg(Wp[0:d]), WV=reg(Wp[d:2 * d]), WD=reg(Wp[2 * d:3 * d]),
            n2=reg(P['n2'][li]),
            gate_up=reg(np.concatenate([gu[dff:2 * dff], gu[0:dff]], axis=0)),
            down=reg(P['down'][li]),
            has_attn=P['has_attn'][li],
        )
        if layer['has_attn']:
            layer['rms_attn'] = reg(P['rms_attn'][li])
            layer['qkv_w'] = reg(P['qkv_w'][li])
            layer['out_proj_w'] = reg(P['out_proj_w'][li])
        layers.append(layer)
    return E, final, layers


def _resident_forward(t, w, ids, d, dff, V, L,
                      attn_heads=None, attn_window=1024):
    """Resident forward over weight buffer ids w={E,final,layers[,router]}. Caller has
    t.begin(). Returns (emb_id, logits_id, cap, nf_id, B, S, router_logits_id).
    cap[li] holds every intermediate buffer id (n1,G,Vv,D,H,r1,n2,gu,h,ff,r2)
    for the backward record. router_logits_id is None if no router in w."""
    E_id, final_id, layers = w['E'], w['final'], w['layers']
    router_id = w.get('router')
    ids = np.asarray(ids, dtype=np.int64); B, S = ids.shape; BS = B * S
    H = attn_heads or 1
    Dh = d // H if attn_heads else d
    ids_u32 = t.register_input_u32(ids.reshape(-1).astype(np.uint32))
    t.forward_begin()
    emb = t.forward_embedding(ids_u32, E_id, B, S, V, d)
    x = emb; cap = []
    for lw in layers:
        # MinGRU block
        n1 = t.forward_rmsnorm(x, lw['n1'], BS, d)
        G = t.forward_linear(n1, lw['WG'], 0, BS, d, d)
        Vv = t.forward_linear(n1, lw['WV'], 0, BS, d, d)
        D = t.forward_linear(n1, lw['WD'], 0, BS, d, d)
        H_min = t.forward_mingru(G, Vv, D, B, S, d)
        r1 = t.forward_add(x, H_min, BS * d)
        # Attention block (if this layer has it)
        if lw.get('has_attn', False):
            ra_rms = t.forward_rmsnorm(r1, lw['rms_attn'], BS, d)
            qkv_id = t.forward_linear(ra_rms, lw['qkv'], 0, BS, d, 3 * d)
            q_id, k_id, v_id = t.forward_qkv_split(qkv_id, B, S, H, Dh)
            attn_out = t.forward_chunked_attention(q_id, k_id, v_id, B, H, S, Dh, attn_window)
            bshd = t.forward_transpose_bhsd_bshd(attn_out, B, H, S, Dh)
            outp = t.forward_linear(bshd, lw['out_proj'], 0, BS, d, d)
            r_attn = t.forward_add(r1, outp, BS * d)
            cap.append((n1, G, Vv, D, H_min, r1, ra_rms, qkv_id, q_id, k_id, v_id,
                        attn_out, bshd, outp, r_attn,
                        lw['rms_attn'], lw['qkv'], lw['out_proj']))
            x_pre_ffn = r_attn
        else:
            cap.append((n1, G, Vv, D, H_min, r1, None))
            x_pre_ffn = r1
        # FFN block
        n2 = t.forward_rmsnorm(x_pre_ffn, lw['n2'], BS, d)
        gu = t.forward_linear(n2, lw['gate_up'], 0, BS, d, 2 * dff)
        h = t.forward_swiglu(gu, BS, dff)
        ff = t.forward_linear(h, lw['down'], 0, BS, dff, d)
        r2 = t.forward_add(x_pre_ffn, ff, BS * d)
        # store FFN intermediates at the end of cap tuple for both attn/non-attn
        cap[-1] = cap[-1] + (n2, gu, h, ff, r2)
        x = r2
    nf = t.forward_rmsnorm(x, final_id, BS, d)
    logits = t.forward_linear(nf, E_id, 0, BS, d, V)
    router_logits_id = None
    if router_id is not None:
        router_logits_id = t.forward_linear(nf, router_id, 0, BS, d, 2)
    t.forward_submit()
    return emb, logits, cap, nf, B, S, router_logits_id


def _fb_run(t, w, ids, targets, d, dff, V, L, Vlang=None, Vast=None,
            attn_heads=0, attn_window=0, dlogits_fn=None):
    """Resident forward + backward (records the matching graph, one backward()).
    Returns (emb_id, logits_numpy, BS, router_logits_np, nf_id, logits_id, Vlang, Vast).
    Grad buffers are then readable via t.get_grad_buffer(weight_id) / get_grad_buffer(emb_id).
    Vlang/Vast partition the combined logits into language and AST regions for
    dual-head loss computation.
    attn_heads / attn_window: only needed when any layer has has_attn=True.

    dlogits_fn: optional `(logits_np) -> dlogits (BS, V)` for a CUSTOM loss (e.g.
    completion-only / prompt-masked SFT). When given, the head node's output
    gradient is seeded DIRECTLY with dlogits and the GPU CrossEntropy node is
    skipped -- so the loss can be any host-computed function of the logits."""
    op = gc.OpType
    E_id, final_id, layers = w['E'], w['final'], w['layers']
    router_id = w.get('router')
    emb, logits, cap, nf, B, S, router_logits_id = _resident_forward(
        t, w, ids, d, dff, V, L)
    BS = B * S
    logits_np = t.read_buffer(logits, [BS, V])
    router_logits_np = None
    if router_logits_id is not None:
        router_logits_np = t.read_buffer(router_logits_id, [BS, 2])
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    tgt_id = t.register_input(targets.astype(np.float32), False)

    x = emb
    for li, lw in enumerate(layers):
        has_attn = lw.get('has_attn', False)
        if has_attn:
            # Unpack all 18 elements for attention layers
            n1, G, Vv, D, H_min, r1, ra_rms, qkv, q, k, v, attn_out, transpose, out_proj, r_attn, rms_attn_id, qkv_id_buf, out_proj_id = cap[li][:18]
            ffn_input = r_attn
            attn_H = attn_heads
            attn_Dh = d // attn_heads if attn_heads else d
        else:
            # Unpack 6 elements + 5 FFN elements for non-attention layers
            n1, G, Vv, D, H_min, r1 = cap[li][:6]
            ffn_input = r1

        # MinGRU block
        nn1 = t.record_op(op.RMSNorm, [_R(x, [BS, d]), _R(lw['n1'], [d])], [_R(n1, [BS, d])]); t.save_for_backward(nn1, [x, lw['n1']])
        nG = t.record_op(op.Linear, [_R(n1, [BS, d]), _R(lw['WG'], [d, d])], [_R(G, [BS, d])]); t.save_for_backward(nG, [n1, lw['WG']])
        nV = t.record_op(op.Linear, [_R(n1, [BS, d]), _R(lw['WV'], [d, d])], [_R(Vv, [BS, d])]); t.save_for_backward(nV, [n1, lw['WV']])
        nD = t.record_op(op.Linear, [_R(n1, [BS, d]), _R(lw['WD'], [d, d])], [_R(D, [BS, d])]); t.save_for_backward(nD, [n1, lw['WD']])
        nM = t.record_op(op.MinGRU, [_R(G, [B, S, d]), _R(Vv, [B, S, d]), _R(D, [B, S, d])], [_R(H_min, [B, S, d])]); t.save_for_backward(nM, [G, Vv, D, H_min])
        t.record_op(op.Add, [_R(x, [BS, d]), _R(H_min, [BS, d])], [_R(r1, [BS, d])])

        # Attention block (if enabled)
        if has_attn:
            # RMSNorm before attention
            nra = t.record_op(op.RMSNorm, [_R(r1, [BS, d]), _R(lw['rms_attn'], [d])], [_R(ra_rms, [BS, d])]); t.save_for_backward(nra, [r1, lw['rms_attn']])
            # QKV projection (fused: output is (BS, 3*d))
            nqkv = t.record_op(op.Linear, [_R(ra_rms, [BS, d]), _R(lw['qkv'], [d, 3*d])], [_R(qkv, [BS, 3*d])]); t.save_for_backward(nqkv, [ra_rms, lw['qkv']])
            # ChunkedAttention: record with qkv as single input so backward
            # produces a single d_qkv gradient that flows to the Linear(QKV) node.
            # q/k/v (from forward_qkv_split) are saved for the backward computation.
            # Pass attention params (B, H, S, Dh, W, scale) as bytes for backward.
            import struct as _struct
            _attn_scale = 1.0 / float(attn_Dh) ** 0.5
            _attn_params = _struct.pack('IIIIIf', B, attn_H, S, attn_Dh, attn_window, _attn_scale)
            n_attn = t.record_op(op.ChunkedAttention,
                                 [_R(qkv, [BS, 3*d])],
                                 [_R(attn_out, [B, attn_H, S, attn_Dh])],
                                 _attn_params)
            t.save_for_backward(n_attn, [q, k, v])
            # Transpose BHSD -> BSHD = (BS, d)
            n_trans = t.record_op(op.Transpose, [_R(attn_out, [B, attn_H, S, attn_Dh])], [_R(transpose, [BS, d])])
            t.save_for_backward(n_trans, [attn_out])
            # Output projection
            n_out = t.record_op(op.Linear, [_R(transpose, [BS, d]), _R(lw['out_proj'], [d, d])], [_R(out_proj, [BS, d])]); t.save_for_backward(n_out, [transpose, lw['out_proj']])
            # Residual connection
            t.record_op(op.Add, [_R(r1, [BS, d]), _R(out_proj, [BS, d])], [_R(r_attn, [BS, d])])

        # FFN block (uses ffn_input which is r_attn for attention layers, r1 otherwise)
        n2, gu, h, ff, r2 = cap[li][-5:]
        nn2 = t.record_op(op.RMSNorm, [_R(ffn_input, [BS, d]), _R(lw['n2'], [d])], [_R(n2, [BS, d])]); t.save_for_backward(nn2, [ffn_input, lw['n2']])
        ngu = t.record_op(op.Linear, [_R(n2, [BS, d]), _R(lw['gate_up'], [2 * dff, d])], [_R(gu, [BS, 2 * dff])]); t.save_for_backward(ngu, [n2, lw['gate_up']])
        nsw = t.record_op(op.SwiGLU, [_R(gu, [BS, 2 * dff])], [_R(h, [BS, dff])]); t.save_for_backward(nsw, [gu])
        ndn = t.record_op(op.Linear, [_R(h, [BS, dff]), _R(lw['down'], [d, dff])], [_R(ff, [BS, d])]); t.save_for_backward(ndn, [h, lw['down']])
        t.record_op(op.Add, [_R(r1, [BS, d]), _R(ff, [BS, d])], [_R(r2, [BS, d])])
        x = r2
    nFin = t.record_op(op.RMSNorm, [_R(x, [BS, d]), _R(final_id, [d])], [_R(nf, [BS, d])]); t.save_for_backward(nFin, [x, final_id])
    nHead = t.record_op(op.Linear, [_R(nf, [BS, d]), _R(E_id, [V, d])], [_R(logits, [BS, V])]); t.save_for_backward(nHead, [nf, E_id])
    if router_logits_id is not None:
        nRouter = t.record_op(op.Linear, [_R(nf, [BS, d]), _R(router_id, [2, d])], [_R(router_logits_id, [BS, 2])]); t.save_for_backward(nRouter, [nf, router_id])
    if dlogits_fn is not None:
        # Custom loss (e.g. prompt-masked SFT): seed the head node's output grad
        # directly with host-computed dlogits, skipping the GPU CrossEntropy node.
        dl = np.ascontiguousarray(dlogits_fn(logits_np), dtype=np.float32)
        t.backward(nHead, t.register_input(dl, False))
    else:
        nCE = t.record_op(op.CrossEntropy, [_R(logits, [BS, V])], [_R(0, [1], False)]); t.save_for_backward(nCE, [logits, tgt_id])
        t.backward(nCE, 0)
    return emb, logits_np, BS, router_logits_np, nf, logits, Vlang, Vast


def _read_grads(t, w, emb, ids, BS, d, dff, V, L):
    """Read grads after _fb_run, mapped to model.py layout (mean-CE /BS): proj
    re-concats the 3 split gvd blocks, gate_up un-swaps, embed merges head + scatter."""
    E_id, final_id, layers = w['E'], w['final'], w['layers']
    ids = np.asarray(ids, dtype=np.int64)

    def gr(b, sh): return t.read_buffer(t.get_grad_buffer(b), sh) / BS
    out = dict(n1=[], n2=[], proj=[], gate_up=[], down=[], has_attn=[],
               rms_attn=[], qkv=[], out_proj=[])
    for lw in layers:
        out['n1'].append(gr(lw['n1'], [d]))
        out['n2'].append(gr(lw['n2'], [d]))
        out['proj'].append(np.concatenate([gr(lw['WG'], [d, d]), gr(lw['WV'], [d, d]), gr(lw['WD'], [d, d])], axis=0))
        gsw = gr(lw['gate_up'], [2 * dff, d])
        out['gate_up'].append(np.concatenate([gsw[dff:2 * dff], gsw[0:dff]], axis=0))
        out['down'].append(gr(lw['down'], [d, dff]))
        
        has_attn = lw.get('has_attn', False)
        out['has_attn'].append(has_attn)
        if has_attn:
            out['rms_attn'].append(gr(lw['rms_attn'], [d]))
            out['qkv'].append(gr(lw['qkv'], [d, 3*d]))
            out['out_proj'].append(gr(lw['out_proj'], [d, d]))
    
    out['final'] = gr(final_id, [d])
    emb_g = gr(emb, [BS, d])
    dE = np.zeros((V, d), np.float32); np.add.at(dE, ids.reshape(-1), emb_g)
    out['embed'] = dE + gr(E_id, [V, d])
    return out


def _fb_grads(t, w, ids, targets, d, dff, V, L):
    """forward+backward + read grads (model.py layout). Returns (grads, logits)."""
    emb, logits_np, BS, *_ = _fb_run(t, w, ids, targets, d, dff, V, L)
    return _read_grads(t, w, emb, ids, BS, d, dff, V, L), logits_np


def _snapshot(model):
    """Init params from a CubbyLM in model.py layout. COPIES (model.py AdamW
    mutates p.data in place; a view would alias the trained weights)."""
    cp = lambda a: np.array(a, dtype=np.float32, copy=True)
    out = dict(
        embed=cp(model.embed.data), final=cp(model.final.data),
        n1=[cp(b.n1.data) for b in model.blocks],
        n2=[cp(b.n2.data) for b in model.blocks],
        proj=[cp(b.mix.proj.weight.data) for b in model.blocks],
        gate_up=[cp(b.ffn.gate_up.weight.data) for b in model.blocks],
        down=[cp(b.ffn.down.weight.data) for b in model.blocks],
        has_attn=[getattr(b, 'has_attn', False) for b in model.blocks],
        rms_attn=[], qkv_w=[], out_proj_w=[],
    )
    for b in model.blocks:
        if getattr(b, 'has_attn', False):
            out['rms_attn'].append(cp(b.rms_attn.data))
            out['qkv'].append(cp(b.attn.qkv.weight.data))
            out['out_proj'].append(cp(b.attn.out_proj.weight.data))
    return out


def _ce(logits, tgt):
    z = logits - logits.max(1, keepdims=True); e = np.exp(z); sm = e / e.sum(1, keepdims=True)
    return float(-np.log(sm[np.arange(len(tgt)), tgt] + 1e-12).mean())


def _sampled_ce(logits, tgt, V, n_samples=1024, d=1024):
    """Importance-sampled CE (Bengio & Senecal 2008): for each position, draw
    K=n_samples uniform negatives from [0, V), compute CE only over the K+1
    subset (target + negatives). With uniform sampling, the IS correction is a
    constant that cancels in softmax -- gradient direction is unbiased.
    
    Args:
        logits: (BS, V) full logits from GPU
        tgt: (BS,) target token IDs
        V: total vocab size
        n_samples: number of negative samples per position
        d: model dimension (unused, kept for API compat)
    
    Returns: scalar CE loss over the sampled subsets
    """
    BS = logits.shape[0]
    K = n_samples
    
    # Draw K negatives per position
    neg = np.random.randint(0, V, size=(BS, K), dtype=np.int64)
    
    # Build subset indices: [target, neg1, neg2, ..., negK] for each position
    # Shape: (BS, K+1)
    subset_ids = np.concatenate([tgt[:, None], neg], axis=1)
    
    # Gather logits for the subset
    # For each position i, we need logits[i, subset_ids[i, :]]
    batch_idx = np.arange(BS)[:, None]
    subset_logits = logits[batch_idx, subset_ids]  # (BS, K+1)
    
    # CE over the subset (target is always at index 0)
    z = subset_logits - subset_logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    sm = e / e.sum(axis=1, keepdims=True)
    target_logits = sm[:, 0]  # probability of the true target in the subset
    loss = float(-np.log(target_logits + 1e-12).mean())
    return loss


def _dual_head_ce(logits, tgt, router_logits, Vlang, Vast,
                  n_samples=1024, use_sampled=False, d_model=1024):
    """Dual-head CE loss with router weighting.
    
    Args:
        logits: (BS, V_total) combined language+AST logits from GPU
        tgt: (BS,) target token IDs
        router_logits: (BS, 2) router raw logits
        Vlang: language vocab size
        Vast: AST vocab size
        n_samples: number of negatives for sampled softmax
        use_sampled: whether to use sampled softmax
        d_model: model dimension
    
    Returns:
        loss: combined loss (language + AST, weighted by router)
        rw: (BS, 2) router weights (softmax over router_logits)
    """
    BS = logits.shape[0]
    
    # Compute router weights
    rl = router_logits - router_logits.max(axis=1, keepdims=True)
    rw = np.exp(rl)
    rw = rw / rw.sum(axis=1, keepdims=True)  # (BS, 2)
    
    # Classify tokens: language (id < Vlang) vs AST (id >= Vlang)
    is_lang = (tgt < Vlang)
    is_ast = (tgt >= Vlang)
    
    # Language head loss — only for language tokens (id < Vlang)
    lang_logits = logits[:, :Vlang]  # (BS, Vlang)
    n_lang = int(is_lang.sum())
    if n_lang > 0:
        lang_mask = is_lang
        if use_sampled:
            lang_loss = _sampled_ce(lang_logits[lang_mask], tgt[lang_mask], Vlang,
                                    n_samples=n_samples, d=d_model)
        else:
            lang_loss = _ce(lang_logits[lang_mask], tgt[lang_mask])
    else:
        lang_loss = 0.0
    
    # Weight language loss by mean router weight for language (channel 0)
    w_lang = rw[:, 0].mean()
    lang_weighted = w_lang * lang_loss
    
    # AST head loss — only for AST tokens (id >= Vlang)
    ast_loss = 0.0
    n_ast = int(is_ast.sum())
    if n_ast > 0 and Vast > 0:
        ast_logits = logits[:, Vlang:]  # (BS, Vast)
        ast_tgt = tgt[is_ast] - Vlang  # shift to [0, Vast)
        ast_logits_sub = ast_logits[is_ast]
        if use_sampled:
            ast_loss = _sampled_ce(ast_logits_sub, ast_tgt, Vast,
                                   n_samples=min(n_samples, Vast), d=d_model)
        else:
            ast_loss = _ce(ast_logits_sub, ast_tgt)
    
    # Weight AST loss by mean router weight for AST (channel 1)
    w_ast = rw[:, 1].mean()
    ast_weighted = w_ast * ast_loss
    
    # Combined loss
    loss = lang_weighted + ast_weighted
    return loss, rw


def _adamw_np(P, g, m, v, step, lr, b1, b2, eps, wd):
    """In-place AdamW over a model-layout param dict, matching model.py AdamW."""
    b1c, b2c = 1.0 - b1 ** step, 1.0 - b2 ** step

    def upd(w, gg, mm, vv):
        mm *= b1; mm += (1 - b1) * gg
        vv *= b2; vv += (1 - b2) * (gg * gg)
        w *= (1 - lr * wd)
        w -= lr * (mm / b1c) / (np.sqrt(vv / b2c) + eps)

    for k in ('embed', 'final'):
        upd(P[k], g[k], m[k], v[k])
    for k in ('n1', 'n2', 'proj', 'gate_up', 'down'):
        for i in range(len(P[k])):
            upd(P[k][i], g[k][i], m[k][i], v[k][i])
    # Attention weights
    for i in range(len(P.get('has_attn', []))):
        if P['has_attn'][i]:
            upd(P['rms_attn'][i], g['rms_attn'][i], m['rms_attn'][i], v['rms_attn'][i])
            upd(P['qkv'][i], g['qkv'][i], m['qkv'][i], v['qkv'][i])
            upd(P['out_proj'][i], g['out_proj'][i], m['out_proj'][i], v['out_proj'][i])


def loss_curve_match(dev, steps=30):
    """Train two trunks from IDENTICAL init on the SAME batch: model.py (numpy
    fwd/bwd + numpy AdamW) vs numpy params driven by the RESIDENT kernels for
    grads + the same numpy AdamW. The only difference is the forward/backward
    source, so the loss curves must track -- proving the resident path drives
    identical learning. (resident AdamW + persistent weights are gated separately
    in grilly; here AdamW is held identical to isolate the trunk.)"""
    force_numpy_reference()                                 # numpy reference (one GPU ctx)
    from cubby.config import make_config
    from cubby.trunk.model import CubbyLM, AdamW

    np.random.seed(1)
    cfg = make_config("0.0.0", vocab_size=256, d_model=128, n_layers=3, d_ffn=256)
    d, L, dff, V = cfg.d_model, cfg.n_layers, cfg.d_ffn, cfg.total_vocab
    B, S = 4, 8
    ids = np.random.randint(0, V, (B, S)).astype(np.int64)
    tgt = np.random.randint(0, V, (B, S)).astype(np.int64)   # fixed batch -> memorize
    lr, b1, b2, eps, wd = 0.01, 0.9, 0.95, 1e-8, 0.0

    ref_model = CubbyLM(cfg)
    P0 = _snapshot(ref_model)

    # reference: model.py numpy training
    opt = AdamW(ref_model.parameters(), lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd)
    ref = []
    for _ in range(steps):
        loss = ref_model.loss(ids, tgt)
        ref.append(float(loss.data))
        opt.zero_grad(); loss.backward(); opt.step()

    # resident: numpy params from the same init, grads from the resident kernels
    import copy as _copy
    t = gc.TapeContext(dev)
    P = _copy.deepcopy(P0)
    m = {k: (np.zeros_like(P[k]) if not isinstance(P[k], list) else [np.zeros_like(a) for a in P[k]]) for k in P}
    v = {k: (np.zeros_like(P[k]) if not isinstance(P[k], list) else [np.zeros_like(a) for a in P[k]]) for k in P}
    res = []
    for step in range(1, steps + 1):
        t.begin()
        E, final, layers = _register_layer_ids(t, P, d, dff, L)
        g, logits = _fb_grads(t, dict(E=E, final=final, layers=layers), ids, tgt, d, dff, V, L)
        res.append(_ce(logits, tgt.reshape(-1)))
        _adamw_np(P, g, m, v, step, lr, b1, b2, eps, wd)
    return ref, res


def _load_token_stream(data, tok, max_tokens, seed=0):
    """Tokenize up to `max_tokens` ids from `data`, reading only what's needed.

      - .json : list[str] of documents (TinyStories-style). Story order is
                permuted by `seed`, so different seeds draw different subsets.
      - .txt  : raw concatenated UTF-8 corpus (e.g. the 134 GB unified pretrain
                file). Sampled at random *document-aligned* byte offsets so a large
                heterogeneous corpus contributes its full source mix (multilingual /
                math / agent / personas), not just whichever source sits at the
                head. Only ~max_tokens of text is ever read. Different seeds yield
                near-disjoint subsets -> a free held-out split for eval.

    Returns list[int] of token ids (length ~max_tokens).
    """
    import os, json
    ext = os.path.splitext(data)[1].lower()
    sb = []
    if ext == ".json":
        with open(data, "r", encoding="utf-8") as f:
            stories = json.load(f)
        for i in np.random.default_rng(seed).permutation(len(stories)):
            sb.extend(tok.encode(stories[int(i)] + "\n"))
            if len(sb) >= max_tokens:
                break
        return sb[:max_tokens]
    # raw text corpus: random document-aligned probes
    size = os.path.getsize(data)
    rng = np.random.default_rng(seed)
    CHUNK = 1 << 20                                          # 1 MB per probe
    with open(data, "rb") as f:
        if size <= 4 * CHUNK:                               # small file: just stream it
            return tok.encode(f.read().decode("utf-8", "ignore"))[:max_tokens]
        while len(sb) < max_tokens:
            f.seek(int(rng.integers(0, size - CHUNK)))
            raw = f.read(CHUNK).decode("utf-8", "ignore")
            nl = raw.find("\n")                             # drop partial leading doc
            if nl != -1:
                raw = raw[nl + 1:]
            sb.extend(tok.encode(raw))
    return sb[:max_tokens]


def _parse_sources(data):
    """Parse a data spec into [(path, weight), ...]. Accepts a single path, or a
    weighted mix `pathA:0.9,pathB:0.1` (weights renormalized). A Windows drive
    colon (`D:\\..`) is not mistaken for a weight -- only a trailing `:<number>`
    counts."""
    out = []
    for p in (s for s in str(data).split(",") if s.strip()):
        head, sep, tail = p.rpartition(":")
        if sep and tail.replace(".", "", 1).strip().isdigit():
            out.append((head.strip(), float(tail)))
        else:
            out.append((p.strip(), 1.0))
    tot = sum(w for _, w in out) or 1.0
    return [(p, w / tot) for p, w in out]


def _read_source(path, tok, n_tokens, seed):
    """Read ~n_tokens token ids from one corpus file, reading only what's needed.
    Supports .json (list[str]), .jsonl (one {'text':..}/line), .txt (raw). Large
    files (.jsonl/.txt) are sampled at random *record-aligned* byte offsets so a
    multi-GB heterogeneous corpus contributes its mix, not just its head."""
    import os, json
    if not os.path.isabs(path) and not os.path.exists(path):
        path = os.path.join(_CUBBY_ROOT, path)             # resolve vs repo root
    ext = os.path.splitext(path)[1].lower()
    rng = np.random.default_rng(seed)
    sb = []
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            docs = json.load(f)
        for i in rng.permutation(len(docs)):
            sb.extend(tok.encode(str(docs[int(i)]) + "\n"))
            if len(sb) >= n_tokens:
                break
        return sb[:n_tokens]
    size = os.path.getsize(path)
    if ext == ".jsonl":
        with open(path, "rb") as f:
            small = size <= (8 << 20)
            while len(sb) < n_tokens:
                if not small:
                    f.seek(int(rng.integers(0, size - (1 << 20))))
                    f.readline()                           # discard partial line
                    lines = f.read(1 << 20).decode("utf-8", "ignore").split("\n")[:-1]
                else:
                    lines = f.read().decode("utf-8", "ignore").split("\n")
                for ln in lines:
                    if not ln.strip():
                        continue
                    try:
                        o = json.loads(ln)
                    except Exception:
                        continue
                    t = o.get("text") or o.get("content") or ""
                    if t:
                        sb.extend(tok.encode(t + "\n"))
                if small:
                    break
        return sb[:n_tokens]
    # .txt raw concatenated corpus
    CHUNK = 1 << 20
    with open(path, "rb") as f:
        if size <= 4 * CHUNK:
            return tok.encode(f.read().decode("utf-8", "ignore"))[:n_tokens]
        while len(sb) < n_tokens:
            f.seek(int(rng.integers(0, size - CHUNK)))
            raw = f.read(CHUNK).decode("utf-8", "ignore")
            nl = raw.find("\n")                            # drop partial leading doc
            if nl != -1:
                raw = raw[nl + 1:]
            sb.extend(tok.encode(raw))
    return sb[:n_tokens]


def _load_token_stream(data, tok, max_tokens, seed=0):
    """Tokenize ~max_tokens ids from one or more weighted sources. A mix
    (`pathA:0.9,pathB:0.1`) is concatenated by token budget; the random-window
    batch sampler then draws from each region in proportion to its weight.
    Different seeds yield near-disjoint subsets -> a free held-out split."""
    import os
    srcs = _parse_sources(data)
    sb = []
    for k, (path, w) in enumerate(srcs):
        n = max(1, int(round(w * max_tokens)))
        got = _read_source(path, tok, n, seed + k * 1009)
        sb.extend(got)
        if len(srcs) > 1:
            print("[data] %-34s w=%.2f  %d tok" % (os.path.basename(path), w, len(got)), flush=True)
    return sb


def _load_sft_examples(data, tok, max_examples=200000):
    """Load (prompt_ids, program_ids) pairs from v4-style jsonl(s) for prompt-masked
    SFT. prompt = `[INSTRUCTION]\\n<nl>\\n[/INSTRUCTION]\\n`; completion = the
    `cubelang_program` SOURCE. `data` is a path or comma-mix (resolved per-source)."""
    import os, json
    exs = []
    for path, _w in _parse_sources(data):
        if not os.path.isabs(path) and not os.path.exists(path):
            path = os.path.join(_CUBBY_ROOT, path)
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    e = json.loads(ln)
                except Exception:
                    continue
                prog = e.get("cubelang_program")
                if not prog:
                    continue
                nl = e.get("prompt") or e.get("question") or e.get("text") or ""
                p_ids = tok.encode("[INSTRUCTION]\n%s\n[/INSTRUCTION]\n" % nl)
                c_ids = tok.encode(prog)
                if c_ids:
                    exs.append((p_ids, c_ids))
                if len(exs) >= max_examples:
                    return exs
    return exs


def train_cubby_resident(version="0.0.0", steps=600, data="tinystory_50k.json",
                         B=8, S=64, lr=3e-4, max_tokens=4000000, sample_every=200,
                         prompt="Once upon a time", gen_tokens=60, dev=None,
                         warmup=0, max_grad_norm=1.0,
                         ckpt_path=None, ckpt_every=100, resume=True, max_consec_skips=10,
                         tokenizer="bbpe65k", identity_probe=""):
    """Train a CubbyLM via the RESIDENT backend (persistent weights + resident
    AdamW + E host path); sample periodically. The default `main.py train`
    backend. `tokenizer` selects the tokenizer: "bbpe65k" (default), "multilingual_bpe"/
    "mbpe32k" (our 32k custom BPE + AST tokens), or "byte".
    Returns (ResidentTrunk, tokenizer)."""
    import json
    import time as _time
    force_numpy_reference()                                 # resident path owns the GPU
    from cubby.config import make_config
    from cubby.trunk.model import CubbyLM, param_count
    from cubby.tokenizer import make_tokenizer

    import os
    np.random.seed(0)
    cfg = make_config(version)
    tok = make_tokenizer(tokenizer)
    # If using multilingual_bpe, the tokenizer knows the language/AST split.
    # Set config vocab_size = lang_vocab_size and n_special_tokens = n_ast_tokens
    # so total_vocab = vocab_size + n_special_tokens = full tokenizer vocab.
    if hasattr(tok, 'lang_vocab_size') and hasattr(tok, 'n_ast_tokens'):
        cfg.vocab_size = tok.lang_vocab_size
        cfg.n_special_tokens = tok.n_ast_tokens
    sb = _load_token_stream(data, tok, max_tokens, seed=0)
    # Remap path: model vocab smaller than the tokenizer (e.g. 'tiny' preset
    # with vocab=10000 on a 65k tokenizer). Compress IDs into dense [0, vocab)
    # by corpus frequency so targets never exceed the head.
    if cfg.total_vocab < tok.vocab_size:
        from cubby.tokenizer import RemapTokenizer
        tok = RemapTokenizer(tok, cfg.total_vocab, sb)
        sb = tok.encode_base(sb)
        print("[remap] V %d->%d  coverage=%.3f%% (rare ids -> <unk>)"
              % (tok.base.vocab_size, tok.vocab_size, tok.coverage * 100), flush=True)
    # For multilingual_bpe, total_vocab = lang_vocab_size + n_ast_tokens = tok.vocab_size
    if not hasattr(tok, 'lang_vocab_size'):
        assert tok.vocab_size == cfg.total_vocab, (tok.vocab_size, cfg.total_vocab)
    stream = np.asarray(sb, dtype=np.int64)
    rng = np.random.default_rng(0)
    def batch():
        ix = rng.integers(0, len(stream) - S - 1, size=B)
        return (np.stack([stream[i:i + S] for i in ix]).astype(np.int64),
                np.stack([stream[i + 1:i + 1 + S] for i in ix]).astype(np.int64))
    def _gen(p):
        s = tok.decode(rt.generate(tok.encode(p), max_new_tokens=gen_tokens))
        return s.encode("ascii", "backslashreplace").decode("ascii")          # cp1252 console
    def sample():
        return _gen(prompt)

    from cubby.trunk import checkpoint as _ckpt
    if ckpt_path is None:
        ckpt_path = os.path.join(_CUBBY_ROOT, "ckpt_%s.grl" % version)

    model = CubbyLM(cfg)
    start_step = 0
    if resume and os.path.exists(ckpt_path):
        _ms, _meta = _ckpt.load_checkpoint(ckpt_path)
        if _ckpt.checkpoint_matches(_meta, cfg):
            _ckpt.apply_model_state(model, _ms)            # restore BEFORE ResidentTrunk()
            start_step = int(_meta.get("step", 0))
            rng = _ckpt.restore_rng(_meta)                 # continue the SAME data stream
            print("[resume] %s @ step %d" % (ckpt_path, start_step), flush=True)
        else:
            print("[resume] shape mismatch, ignoring %s" % ckpt_path, flush=True)
    rt = ResidentTrunk(model, dev or make_device())
    print("[resident] V=%d d=%d L=%d d_ffn=%d params=%d stream=%d B=%d S=%d"
          % (cfg.total_vocab, cfg.d_model, cfg.n_layers, cfg.d_ffn,
             param_count(model), len(stream), B, S), flush=True)
    print("[hp] lr=%.1e warmup=%d clip=%s betas=(0.9,0.95) wd=0.01" % (lr, warmup, max_grad_norm), flush=True)
    guard = _ckpt.SkipGuard(max_consecutive=max_consec_skips)
    step = start_step
    def _save(tag):
        try:
            _ckpt.save_checkpoint(ckpt_path, rt, step=step, rng=rng, version=version,
                                  lr=lr, warmup=warmup, max_grad_norm=max_grad_norm)
            print("[ckpt] %s @ step %d -> %s" % (tag, step, ckpt_path), flush=True)
        except Exception as _e:
            print("[ckpt] save FAILED (%s): %r" % (tag, _e), flush=True)

    # Honor the preset's sampled-softmax flag on the resident path (else the
    # full (N, V) logit tensor is materialized; tolerable at V<=32k, but the
    # flag must actually take effect when set).
    use_sampled = bool(getattr(cfg, "enable_sampled_softmax", False))
    n_samples = int(getattr(cfg, "n_samples", 1024))
    if use_sampled:
        print("[ce] sampled softmax: K=%d negatives/token" % n_samples, flush=True)
    t0 = _time.perf_counter(); nskip = 0
    try:
        for step in range(start_step + 1, steps + 1):
            ids, tgt = batch()
            lr_t = lr * min(1.0, step / warmup) if warmup else lr  # linear LR warmup
            loss, gnorm, skipped = rt.train_step(ids, tgt, step, lr=lr_t, max_grad_norm=max_grad_norm,
                                                 use_sampled=use_sampled, n_samples=n_samples)
            nskip += int(skipped)
            print("[%4d/%d] ce=%.3f ppl=%.1f gnorm=%.2e lr=%.1e (%.2f it/s)%s"
                  % (step, steps, loss, np.exp(loss), gnorm, lr_t,
                     step / (_time.perf_counter() - t0), "  [skipped]" if skipped else ""), flush=True)
            if guard.update(skipped):                             # K-in-a-row -> divergence
                print("[abort] %d consecutive non-finite grads" % guard.max_consecutive, flush=True)
                _save("diverge"); break
            if sample_every and step % sample_every == 0:
                print("  sample:", repr(sample()), flush=True)
                if identity_probe:
                    print("  ident :", repr(_gen(identity_probe)), flush=True)
            if ckpt_every and step % ckpt_every == 0:
                _save("periodic")
    except KeyboardInterrupt:
        print("[interrupt] flushing checkpoint", flush=True); _save("interrupt"); raise
    except Exception as _e:                                       # OOM / any in-step failure
        print("[error] step %d: %r -> emergency checkpoint" % (step, _e), flush=True)
        _save("emergency"); raise
    else:
        _save("final")
    if nskip:
        print("[warn] %d step(s) skipped (non-finite grad)" % nskip, flush=True)
    print("[done] %.1fs  final sample: %r" % (_time.perf_counter() - t0, sample()), flush=True)
    return rt, tok


def train_cubby_sft(version="mbpe_emit", steps=2000, data="", B=8, S=256, lr=3e-4,
                    max_examples=200000, sample_every=200,
                    sample_prompt="What is 5 plus 3?", gen_tokens=160, dev=None,
                    warmup=50, max_grad_norm=1.0, ckpt_path=None, ckpt_every=200,
                    resume=True, max_consec_skips=10, tokenizer="mbpe32k"):
    """Prompt-masked SFT: train p(program | instruction) on (instruction, program)
    pairs with COMPLETION-ONLY loss (loss masked to the program tokens). The trunk
    emits `.cube` SOURCE; the compiler is the gate. See
    docs/TRUNK_VM_EMISSION_CONTRACT.md."""
    import os, time as _time
    force_numpy_reference()
    from cubby.config import make_config
    from cubby.trunk.model import CubbyLM, param_count
    from cubby.tokenizer import make_tokenizer
    from cubby.trunk import checkpoint as _ckpt
    np.random.seed(0)
    cfg = make_config(version)
    tok = make_tokenizer(tokenizer)
    if hasattr(tok, 'lang_vocab_size') and hasattr(tok, 'n_ast_tokens'):
        cfg.vocab_size = tok.lang_vocab_size
        cfg.n_special_tokens = tok.n_ast_tokens
    exs = _load_sft_examples(data, tok, max_examples)
    if not exs:
        raise SystemExit("no SFT examples loaded from %r" % data)
    rng = np.random.default_rng(0)

    def batch():
        idx = rng.integers(0, len(exs), size=B)
        ib = np.zeros((B, S), np.int64); tb = np.zeros((B, S), np.int64)
        mb = np.zeros((B, S), np.float32)
        for bi, ei in enumerate(idx):
            p, c = exs[int(ei)]
            full = (p + c)[: S + 1]
            Lp = len(p)
            for i in range(len(full) - 1):
                ib[bi, i] = full[i]; tb[bi, i] = full[i + 1]
                mb[bi, i] = 1.0 if (i + 1) >= Lp else 0.0   # loss only on program tokens
        return ib, tb, mb

    def sample():
        out = rt.generate(tok.encode("[INSTRUCTION]\n%s\n[/INSTRUCTION]\n" % sample_prompt),
                          max_new_tokens=gen_tokens, temperature=0.0)
        # skip_special=False: render opcode/role AST tokens as literal source
        return tok.decode(out, skip_special=False).encode("ascii", "backslashreplace").decode("ascii")

    if ckpt_path is None:
        ckpt_path = os.path.join(_CUBBY_ROOT, "ckpt_%s.grl" % version)
    model = CubbyLM(cfg)
    start_step = 0
    if resume and os.path.exists(ckpt_path):
        ms, meta = _ckpt.load_checkpoint(ckpt_path)
        if _ckpt.checkpoint_matches(meta, cfg):
            _ckpt.apply_model_state(model, ms); start_step = int(meta.get("step", 0))
            print("[resume] %s @ step %d" % (ckpt_path, start_step), flush=True)
    rt = ResidentTrunk(model, dev or make_device())
    print("[sft] %s V=%d d=%d L=%d examples=%d B=%d S=%d params=%d" % (
        version, cfg.total_vocab, cfg.d_model, cfg.n_layers, len(exs), B, S, param_count(model)), flush=True)
    print("[hp] lr=%.1e warmup=%d clip=%s (prompt-masked, completion-only)" % (lr, warmup, max_grad_norm), flush=True)
    guard = _ckpt.SkipGuard(max_consecutive=max_consec_skips)
    step = start_step

    def _save(tag):
        try:
            _ckpt.save_checkpoint(ckpt_path, rt, step=step, rng=rng, version=version,
                                  lr=lr, warmup=warmup, max_grad_norm=max_grad_norm)
            print("[ckpt] %s @ step %d -> %s" % (tag, step, ckpt_path), flush=True)
        except Exception as _e:
            print("[ckpt] save FAILED (%s): %r" % (tag, _e), flush=True)

    t0 = _time.perf_counter(); nskip = 0
    try:
        for step in range(start_step + 1, steps + 1):
            ib, tb, mb = batch()
            lr_t = lr * min(1.0, step / warmup) if warmup else lr
            loss, gnorm, skipped = rt.train_step_sft(ib, tb, mb, step, lr=lr_t, max_grad_norm=max_grad_norm)
            nskip += int(skipped)
            print("[%4d/%d] ce=%.3f ppl=%.1f gnorm=%.2e lr=%.1e (%.2f it/s)%s" % (
                step, steps, loss, np.exp(min(loss, 20)), gnorm, lr_t,
                step / (_time.perf_counter() - t0), "  [skipped]" if skipped else ""), flush=True)
            if guard.update(skipped):
                print("[abort] divergence", flush=True); _save("diverge"); break
            if sample_every and step % sample_every == 0:
                print("  sample:", repr(sample()), flush=True)
            if ckpt_every and step % ckpt_every == 0:
                _save("periodic")
    except KeyboardInterrupt:
        print("[interrupt]", flush=True); _save("interrupt"); raise
    except Exception as _e:
        print("[error] step %d: %r" % (step, _e), flush=True); _save("emergency"); raise
    else:
        _save("final")
    print("[done] %.1fs  final sample: %r" % (_time.perf_counter() - t0, sample()), flush=True)
    return rt, tok


def eval_full_softmax_ppl(version="mbpe_v33", data="tinystory_50k.json",
                          tokenizer="mbpe32k", ckpt_path=None, dev=None,
                          n_batches=40, B=4, S=128, eval_tokens=400000, seed=98765):
    """UNWEIGHTED full-softmax language-head perplexity -- the comparable PPL the
    router-weighted *sampled* training loss does NOT report.

    Training prints ~0.5x the true language NLL: the untrained router halves the
    language weight (w_lang ~ 0.5) and TinyStories has no AST mass, so the loss is
    `w_lang * sampled_lang_NLL`. This evaluates the language head with the COMPLETE
    (BS, Vlang) softmax over a held-out tail slice, unweighted -- directly
    comparable to the tiny-scale full-softmax PPL. AST targets are excluded.

    Run this when the training process is IDLE: two Vulkan contexts sharing 12 GB
    VRAM will contend (see the resident trunk's persistent weights + Adam state).
    Loads `ckpt_path` (default ckpt_<version>.grl).
    """
    import json, os
    force_numpy_reference()                                 # resident path owns the GPU
    from cubby.config import make_config
    from cubby.trunk.model import CubbyLM
    from cubby.tokenizer import make_tokenizer
    from cubby.trunk import checkpoint as _ckpt

    cfg = make_config(version)
    tok = make_tokenizer(tokenizer)
    if hasattr(tok, 'lang_vocab_size') and hasattr(tok, 'n_ast_tokens'):
        cfg.vocab_size = tok.lang_vocab_size
        cfg.n_special_tokens = tok.n_ast_tokens
    Vlang = int(cfg.vocab_size)

    # seed 7777 != the trainer's seed 0 -> near-disjoint held-out sample
    stream = np.asarray(_load_token_stream(data, tok, eval_tokens, seed=7777), dtype=np.int64)

    if ckpt_path is None:
        ckpt_path = os.path.join(_CUBBY_ROOT, "ckpt_%s.grl" % version)
    if not os.path.exists(ckpt_path):
        raise SystemExit("no checkpoint at %s -- train first" % ckpt_path)
    model = CubbyLM(cfg)
    ms, meta = _ckpt.load_checkpoint(ckpt_path)
    if not _ckpt.checkpoint_matches(meta, cfg):
        raise SystemExit("checkpoint shape mismatch for version %s" % version)
    _ckpt.apply_model_state(model, ms)
    rt = ResidentTrunk(model, dev or make_device())

    rng = np.random.default_rng(seed)
    tot_nll, tot_tok = 0.0, 0
    for _ in range(n_batches):
        ix = rng.integers(0, len(stream) - S - 1, size=B)
        ids = np.stack([stream[i:i + S] for i in ix]).astype(np.int64)
        tgt = np.stack([stream[i + 1:i + 1 + S] for i in ix]).astype(np.int64).reshape(-1)
        logits = rt.logits(ids)[:, :, :Vlang].reshape(B * S, Vlang)
        lang = tgt < Vlang                                   # exclude AST targets
        lg, tt = logits[lang], tgt[lang]
        lg = lg - lg.max(axis=1, keepdims=True)
        lse = np.log(np.exp(lg).sum(axis=1))
        tot_nll += float((lse - lg[np.arange(len(tt)), tt]).sum())
        tot_tok += int(len(tt))
    ce = tot_nll / max(tot_tok, 1)
    print("[eval] %s @ step %d  full-softmax lang-head CE=%.4f  PPL=%.2f  "
          "(%d tokens, Vlang=%d)" % (version, int(meta.get("step", 0)), ce,
                                     float(np.exp(ce)), tot_tok, Vlang), flush=True)
    return ce, float(np.exp(ce))


def generate_from_checkpoint(version="mbpe_v33", prompt="The ", tokenizer="mbpe32k",
                             ckpt_path=None, dev=None, max_new_tokens=80,
                             temperature=0.8, seed=0, skip_special=True, rep_penalty=1.0):
    """Load a checkpoint and free-run from `prompt`. Spot-check coherence and the
    Grilly identity -- for the latter, prompt in the chat format the identity
    corpus uses, e.g.:
        <|system|>\\nYou are Grilly, a helpful assistant.\\n<|user|>\\nWho are you?\\n<|assistant|>\\n
    Run when the trainer is idle (shared 12 GB VRAM)."""
    import os
    force_numpy_reference()                                 # resident path owns the GPU
    from cubby.config import make_config
    from cubby.trunk.model import CubbyLM
    from cubby.tokenizer import make_tokenizer
    from cubby.trunk import checkpoint as _ckpt
    cfg = make_config(version)
    tok = make_tokenizer(tokenizer)
    if hasattr(tok, 'lang_vocab_size') and hasattr(tok, 'n_ast_tokens'):
        cfg.vocab_size = tok.lang_vocab_size
        cfg.n_special_tokens = tok.n_ast_tokens
    if ckpt_path is None:
        ckpt_path = os.path.join(_CUBBY_ROOT, "ckpt_%s.grl" % version)
    if not os.path.exists(ckpt_path):
        raise SystemExit("no checkpoint at %s -- train first" % ckpt_path)
    model = CubbyLM(cfg)
    ms, meta = _ckpt.load_checkpoint(ckpt_path)
    if not _ckpt.checkpoint_matches(meta, cfg):
        raise SystemExit("checkpoint shape mismatch for version %s" % version)
    _ckpt.apply_model_state(model, ms)
    rt = ResidentTrunk(model, dev or make_device())
    out = rt.generate(tok.encode(prompt), max_new_tokens=max_new_tokens,
                      temperature=temperature, seed=seed, rep_penalty=rep_penalty)
    text = tok.decode(out, skip_special=skip_special).encode("ascii", "backslashreplace").decode("ascii")
    print("[gen] %s @ step %d  T=%.2f  prompt=%r\n%s"
          % (version, int(meta.get("step", 0)), temperature, prompt, text), flush=True)
    return text


def train_demo(dev, steps=150, tokens=400000):
    """Train a real CubbyLM via the RESIDENT backend on TinyStories, then
    generate -- the train->generate gate before flipping the default. Also
    asserts cubby.trace emission works from the resident forward."""
    import json
    force_numpy_reference()                                 # only the resident path owns the GPU
    from cubby.config import make_config
    from cubby.trunk.model import CubbyLM
    from cubby.tokenizer import make_tokenizer

    np.random.seed(0)
    cfg = make_config("0.0.0", d_model=256, n_layers=6, d_ffn=512)   # V=65536
    tok = make_tokenizer("bbpe65k")
    assert tok.vocab_size == cfg.total_vocab, (tok.vocab_size, cfg.total_vocab)
    with open(r"C:\Users\grill\Documents\GitHub\cubby-lm\tinystory_50k.json", "r", encoding="utf-8") as f:
        stories = json.load(f)
    sb = []
    for s in stories:
        sb.extend(tok.encode(s + "\n"))
        if len(sb) >= tokens:
            break
    stream = np.asarray(sb[:tokens], dtype=np.int64)
    B, S = 8, 64
    rng = np.random.default_rng(0)
    def batch():
        ix = rng.integers(0, len(stream) - S - 1, size=B)
        return (np.stack([stream[i:i + S] for i in ix]).astype(np.int64),
                np.stack([stream[i + 1:i + 1 + S] for i in ix]).astype(np.int64))

    model = CubbyLM(cfg)
    rt = ResidentTrunk(model, dev)
    print("=== RESIDENT CubbyLM training on TinyStories (V=%d d=%d L=%d, B=%d S=%d) ==="
          % (cfg.total_vocab, cfg.d_model, cfg.n_layers, B, S))
    print("    starting CE ~ ln(V) = %.3f.  step  ce  ppl" % np.log(cfg.total_vocab))
    import time as _time
    t0 = _time.perf_counter()
    for step in range(1, steps + 1):
        ids, tgt = batch()
        loss, gnorm, _ = rt.train_step(ids, tgt, step, lr=3e-3)
        if step == 1 or step % 25 == 0:
            print("  %4d   %.4f   %.1f   gnorm=%.2e" % (step, loss, np.exp(loss), gnorm))
    dt = _time.perf_counter() - t0

    # cubby.trace gate: a traced resident forward emits one record per block
    sink = trace.MemorySink()
    with trace.trace_to(sink, "audit"):
        trace.set_step(0)
        rt.logits(batch()[0][:1])
    nblocks = len([r for r in sink.records if r.component.startswith("block")])
    print("\n  cubby.trace: %d/%d block records emitted from the resident forward"
          % (nblocks, cfg.n_layers))

    # train->generate gate
    seed = tok.encode("Once upon a time")
    out = rt.generate(seed, max_new_tokens=60, temperature=0.8, seed=0)
    sample = tok.decode(out).encode("ascii", "backslashreplace").decode("ascii")  # cp1252 console
    print("  sample: %r" % sample)
    print("\n  %d steps, %.0f ms/step. CE descended -> resident-trained CubbyLM generates."
          % (steps, 1e3 * dt / steps))
    ok = nblocks == cfg.n_layers
    print("RESIDENT TRAIN+GEN:", "PASS" if ok else "FAIL")
    return ok


def gradient_parity(model, ids, targets, dev=None):
    """Per-parameter max_abs_diff between resident grads and model.py's tape grads."""
    for p in model.parameters():
        p.grad = None
    loss = model.loss(np.asarray(ids, np.int64), np.asarray(targets, np.int64))
    loss.backward()
    rt = ResidentTrunk(model, dev)
    rg = rt.grads(ids, targets)

    def rel(a, b):
        a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
        return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-9))

    results = {}
    results['embed'] = rel(rg['embed'], model.embed.grad)
    results['final'] = rel(rg['final'], model.final.grad)
    for li, b in enumerate(model.blocks):
        results['n1[%d]' % li] = rel(rg['n1'][li], b.n1.grad)
        results['n2[%d]' % li] = rel(rg['n2'][li], b.n2.grad)
        results['proj[%d]' % li] = rel(rg['proj'][li], b.mix.proj.weight.grad)
        results['gate_up[%d]' % li] = rel(rg['gate_up'][li], b.ffn.gate_up.weight.grad)
        results['down[%d]' % li] = rel(rg['down'][li], b.ffn.down.weight.grad)
    return results


def forward_parity(model, ids, dev=None):
    """max_abs_diff between the resident forward and model.py's numpy forward."""
    import grilly.nn.autograd as _ag
    rt = ResidentTrunk(model, dev)
    res = rt.logits(ids)
    with _ag.no_grad():
        ref = np.asarray(model(np.asarray(ids, dtype=np.int64)).data, dtype=np.float32)
    return float(np.abs(res - ref).max()), res, ref


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, r"C:\Users\grill\Documents\GitHub\cubby-lm")

    if "train" in sys.argv:                              # resident train->generate gate
        _steps = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 150
        ok = train_demo(make_device(), steps=_steps)
        sys.stdout.flush()
        os._exit(0 if ok else 1)

    from cubby.config import make_config
    from cubby.trunk.model import CubbyLM

    import grilly.nn.autograd as _ag
    force_numpy_reference()                              # only the resident path owns the GPU
    np.random.seed(0)
    # small but real 0.0.0-shaped trunk for a fast parity check
    cfg = make_config("0.0.0", vocab_size=512, d_model=128, n_layers=4, d_ffn=256)
    model = CubbyLM(cfg)
    ids = np.random.randint(0, cfg.total_vocab, size=(2, 8)).astype(np.int64)
    tgts = np.random.randint(0, cfg.total_vocab, size=(2, 8)).astype(np.int64)
    dev = make_device()
    rt = ResidentTrunk(model, dev)                       # ONE resident trunk for both gates

    print("=== STEP-5 PARITY: resident trunk vs model.py (numpy/Python-tape) ===")
    print("    cfg: V=%d d=%d L=%d d_ffn=%d  ids=(2,8)\n" %
          (cfg.total_vocab, cfg.d_model, cfg.n_layers, cfg.d_ffn))

    # forward parity
    res = rt.logits(ids)
    with _ag.no_grad():
        ref = np.asarray(model(ids).data, dtype=np.float32)
    diff = float(np.abs(res - ref).max())
    print("  [forward] logits max_abs_diff = %.3e  %s"
          % (diff, "PASS" if diff < 1e-3 else "FAIL"))

    # gradient parity (model.py tape backward as reference)
    for p in model.parameters():
        p.grad = None
    model.loss(ids, tgts).backward()
    rg = rt.grads(ids, tgts)
    def _rel(a, b):
        a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
        return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-9))
    gp = {'embed': _rel(rg['embed'], model.embed.grad), 'final': _rel(rg['final'], model.final.grad)}
    for li, b in enumerate(model.blocks):
        gp['n1[%d]' % li] = _rel(rg['n1'][li], b.n1.grad)
        gp['n2[%d]' % li] = _rel(rg['n2'][li], b.n2.grad)
        gp['proj[%d]' % li] = _rel(rg['proj'][li], b.mix.proj.weight.grad)
        gp['gate_up[%d]' % li] = _rel(rg['gate_up'][li], b.ffn.gate_up.weight.grad)
        gp['down[%d]' % li] = _rel(rg['down'][li], b.ffn.down.weight.grad)
    print("  [gradient] per-param rel_err vs model.py tape:")
    worst = 0.0
    for k in sorted(gp, key=lambda s: (s.split('[')[0], s)):
        worst = max(worst, gp[k])
        print("    %-12s rel_err=%.3e  %s" % (k, gp[k], "PASS" if gp[k] < 1e-2 else "FAIL"))
    ok = diff < 1e-3 and worst < 1e-2
    print("\n  worst gradient rel_err = %.3e" % worst)
    print("STEP-5 PARITY:", "PASS" if ok else "FAIL")

    # loss-curve match: resident-driven training vs model.py numpy training
    print("\n=== STEP-5 loss-curve match: resident-grad training vs model.py (numpy) ===")
    ref, res = loss_curve_match(dev, steps=30)
    print("  step   model.py   resident   |diff|")
    worst_l = 0.0
    for s in range(0, len(ref), 5):
        dlt = abs(ref[s] - res[s]); worst_l = max(worst_l, dlt)
        print("  %4d   %.5f    %.5f   %.2e" % (s, ref[s], res[s], dlt))
    worst_l = max(abs(a - b) for a, b in zip(ref, res))
    descended = res[-1] < 0.5 * res[0]
    ok_l = worst_l < 1e-2 and descended
    print("  worst |loss diff| over all %d steps = %.3e ; descended %.3f->%.3f"
          % (len(ref), worst_l, res[0], res[-1]))
    print("LOSS-CURVE MATCH:", "PASS" if ok_l else "FAIL")

    print("\nSTEP-5 OVERALL:", "PASS" if (ok and ok_l) else "FAIL")
    sys.stdout.flush()
    os._exit(0 if (ok and ok_l) else 1)   # skip Python/C++ finalizers (GPU teardown order)
