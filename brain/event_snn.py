"""Event-driven SNN feed-forward, GPU-backed by grilly.spike_propagate_batch.

EventDrivenSynapsis is a drop-in for cubemind.brain.Synapsis. For SPARSE spike
input it routes propagation through grilly's batched event-driven kernel
(weights resident, one GPU submit) instead of a dense GEMV; for dense input it
falls back to dense matmul automatically. Numerically identical to dense to fp32.

Propagation:  out[m, post] = sum over fired pre of  spike[m, pre] * W[post, pre]
which equals  x @ W.T  (W stored (out, in) like cubemind's Synapsis).

The grilly op computes  out[m, post] = sum_pre x[m, pre] * Wop[pre, post]  with
Wop = W.T (in, out), and carries the spike MAGNITUDE per fired index, so multi-
bit GIF spikes are handled exactly (not just binary).
"""
from __future__ import annotations

import math
import numpy as np

# ── grilly GPU primitive (optional) ──────────────────────────────────────
# Import grilly first: its __init__ registers the compiled `grilly_core`
# module (PyTorch-style), so no .pyd needs to be copied into site-packages.
_core = None
_dev = None
try:
    import grilly  # noqa: F401  — registers grilly_core, sets up the package
    import grilly_core as _core
    from grilly.backend import _bridge as _grilly_bridge
    if _grilly_bridge.is_available():
        _dev = _grilly_bridge._get_device()
    else:
        _core = None
except Exception:
    _core = None


def grilly_available() -> bool:
    return _core is not None and _dev is not None


def _compact_batch(x_flat: np.ndarray):
    """Vectorized compaction of M spike vectors -> concatenated fired lists.

    No Python loop: np.nonzero returns indices in row-major order (grouped by
    row), so counts/offsets follow from bincount + exclusive prefix sum.

    Args:
        x_flat: (M, N_in) spike values (0 = silent). Sparse along N_in.
    Returns:
        idx  (uint32, total_fired) concatenated fired pre-indices,
        off  (uint32, M)           start offset of each vector in idx,
        cnt  (uint32, M)           fired count per vector,
        vals (float32, total_fired) spike magnitude at each fired index.
    """
    M = x_flat.shape[0]
    rows, cols = np.nonzero(x_flat)                      # row-major (row-grouped)
    cnt = np.bincount(rows, minlength=M).astype(np.uint32)
    off = np.zeros(M, np.uint32)
    if M > 1:
        np.cumsum(cnt[:-1], out=off[1:])                # exclusive prefix sum
    idx = cols.astype(np.uint32)
    vals = x_flat[rows, cols].astype(np.float32)
    return idx, off, cnt, vals


# ═══════════════════════════════════════════════════════════════════════════
# EventDrivenSynapsis — drop-in for cubemind.brain.Synapsis
# ═══════════════════════════════════════════════════════════════════════════
class EventDrivenSynapsis:
    """Spike-driven linear transform; event-driven GPU path for sparse input.

    Same constructor/forward contract as cubemind.brain.Synapsis. `mode`:
      'auto'   -> event-driven when input density <= sparse_threshold, else dense
      'sparse' -> always event-driven (grilly)
      'dense'  -> always dense matmul (numpy / grilly linear)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        target_firing_rate: float = 0.1,
        enable_stdp: bool = False,
        stdp_lr: float = 0.001,
        trace_decay: float = 0.95,
        seed: int = 42,
        mode: str = "auto",
        sparse_threshold: float = 0.25,
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.enable_stdp = enable_stdp
        self.stdp_lr = stdp_lr
        self.trace_decay = trace_decay
        self.mode = mode
        self.sparse_threshold = sparse_threshold

        rng = np.random.default_rng(seed)
        std = 1.0 / math.sqrt(in_features * max(target_firing_rate, 0.01))
        # (out, in), matching cubemind Synapsis
        self.weight = rng.normal(0, std, (out_features, in_features)).astype(np.float32)
        self.bias = np.zeros(out_features, dtype=np.float32)

        self._pre_trace = None
        self._post_trace = None
        self._W_op = None      # cached (in, out) transpose for the GPU op
        self._W_dirty = True

        # bookkeeping: which path the last forward used
        self.last_path = None

    def _w_op(self) -> np.ndarray:
        if self._W_dirty or self._W_op is None:
            self._W_op = np.ascontiguousarray(self.weight.T)  # (in, out)
            self._W_dirty = False
        return self._W_op

    def _propagate_sparse(self, x_flat: np.ndarray) -> np.ndarray:
        """Event-driven propagation via grilly. x_flat: (M, in) -> (M, out)."""
        M = x_flat.shape[0]
        idx, off, cnt, vals = _compact_batch(x_flat)
        out = _core.spike_propagate_batch(
            _dev,
            idx.view(np.float32), off.view(np.float32), cnt.view(np.float32),
            self._w_op(), vals,
            int(self.in_features), int(self.out_features), int(M),
        )
        return np.asarray(out, dtype=np.float32) + self.bias

    def forward(self, x: np.ndarray, state=None):
        squeezed = False
        if x.ndim == 2:
            x = x[np.newaxis, :]
            squeezed = True

        flat = x.reshape(-1, self.in_features).astype(np.float32)

        use_sparse = False
        if self.mode == "sparse":
            use_sparse = grilly_available()
        elif self.mode == "auto" and grilly_available():
            density = np.count_nonzero(flat) / max(flat.size, 1)
            use_sparse = density <= self.sparse_threshold

        if use_sparse:
            try:
                out_flat = self._propagate_sparse(flat)
                self.last_path = "sparse"
            except Exception:
                out_flat = flat @ self.weight.T + self.bias
                self.last_path = "dense_fallback"
        else:
            out_flat = flat @ self.weight.T + self.bias
            self.last_path = "dense"

        output = out_flat.reshape(x.shape[:-1] + (self.out_features,)).astype(np.float32)

        if self.enable_stdp:
            self._stdp_update(x, output)
            self._W_dirty = True

        if squeezed:
            output = output[0]
        return output, None

    def _stdp_update(self, pre_spikes: np.ndarray, post_spikes: np.ndarray) -> None:
        batch, seq_len, _ = pre_spikes.shape
        if self._pre_trace is None or self._pre_trace.shape[0] != batch:
            self._pre_trace = np.zeros((batch, self.in_features), dtype=np.float32)
            self._post_trace = np.zeros((batch, self.out_features), dtype=np.float32)
        dW = np.zeros_like(self.weight)
        for t in range(seq_len):
            self._pre_trace *= self.trace_decay
            self._post_trace *= self.trace_decay
            self._pre_trace += pre_spikes[:, t, :]
            self._post_trace += post_spikes[:, t, :]
            dW += np.mean(
                self._post_trace[:, :, None] * self._pre_trace[:, None, :], axis=0)
        self.weight += self.stdp_lr * dW / max(seq_len, 1)
        norms = np.linalg.norm(self.weight, axis=1, keepdims=True)
        self.weight /= np.maximum(norms, 1e-6)

    def reset_traces(self) -> None:
        self._pre_trace = None
        self._post_trace = None

    @property
    def param_count(self) -> int:
        return self.weight.size + self.bias.size


# ═══════════════════════════════════════════════════════════════════════════
# MiniGIF — compact multi-bit adaptive LIF (bundled stand-in for GIFNeuron)
# ═══════════════════════════════════════════════════════════════════════════
class MiniGIF:
    """Lightweight multi-bit integrate-and-fire neuron producing sparse spikes.

    Stand-in so EventDrivenSNNFFN runs without importing cubemind. Swap in
    cubemind.brain.GIFNeuron for the full dynamics; the FFN only needs
    forward(h:(M,T,dim)) -> (spikes:(M,T,dim), None).
    """

    def __init__(self, dim, L=8, tau=10.0, threshold=1.0, alpha=0.01, seed=0):
        self.dim = dim
        self.L = L
        self.decay = math.exp(-1.0 / max(tau, 1e-3))
        self.threshold = float(threshold)
        self.alpha = float(alpha)

    def forward(self, h):
        M, T, D = h.shape
        v = np.zeros((M, D), dtype=np.float32)
        thr = np.full((M, D), self.threshold, dtype=np.float32)
        out = np.zeros_like(h, dtype=np.float32)
        for t in range(T):
            v = v * self.decay + h[:, t, :]
            s = np.floor(np.clip(v / np.maximum(thr, 1e-6), 0.0, self.L))
            out[:, t, :] = s
            v -= s * thr                       # soft reset
            thr += self.alpha * (s > 0)        # spike-frequency adaptation
        return out.astype(np.float32), None


# ═══════════════════════════════════════════════════════════════════════════
# EventDrivenSNNFFN — drop-in for cubemind.brain.SNNFFN
# ═══════════════════════════════════════════════════════════════════════════
class EventDrivenSNNFFN:
    """Synapsis -> GIF -> Synapsis -> GIF -> mean-pool, event-driven synapses.

    Layer-2 synapse sees sparse GIF spikes, so its propagation runs on grilly's
    event-driven kernel; layer-1 sees the dense (continuous) input and stays
    dense. `mode` is forwarded to both synapses ('auto' does the right thing).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        num_timesteps: int = 4,
        L: int = 8,
        tau: float = 10.0,
        threshold: float = 1.0,
        alpha: float = 0.01,
        enable_stdp: bool = False,
        stdp_lr: float = 0.001,
        dropout_rate: float = 0.0,
        seed: int = 42,
        mode: str = "auto",
        neuron_cls=None,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or input_dim
        self.num_timesteps = num_timesteps
        self.dropout_rate = dropout_rate
        nc = neuron_cls or MiniGIF

        self.syn1 = EventDrivenSynapsis(input_dim, hidden_dim, enable_stdp=enable_stdp,
                                        stdp_lr=stdp_lr, seed=seed, mode=mode)
        self.neuron1 = nc(hidden_dim, L=L, tau=tau, threshold=threshold,
                          alpha=alpha, seed=seed + 1)
        self.syn2 = EventDrivenSynapsis(hidden_dim, self.output_dim, enable_stdp=enable_stdp,
                                        stdp_lr=stdp_lr, seed=seed + 2, mode=mode)
        self.neuron2 = nc(self.output_dim, L=L, tau=tau, threshold=threshold,
                          alpha=alpha, seed=seed + 3)
        self._rng = np.random.default_rng(seed)

    def forward(self, x: np.ndarray) -> np.ndarray:
        squeezed = False
        if x.ndim == 2:
            x = x[np.newaxis, :]
            squeezed = True
        batch, seq_len, _ = x.shape

        x_expanded = np.repeat(x.reshape(batch * seq_len, 1, self.input_dim),
                               self.num_timesteps, axis=1)
        h1, _ = self.syn1.forward(x_expanded)      # dense (continuous input)
        spikes1, _ = self.neuron1.forward(h1)
        h2, _ = self.syn2.forward(spikes1)         # sparse -> event-driven
        spikes2, _ = self.neuron2.forward(h2)

        output = spikes2.mean(axis=1)
        if self.dropout_rate > 0:
            mask = (self._rng.random(output.shape) > self.dropout_rate).astype(np.float32)
            output = output * mask / max(1.0 - self.dropout_rate, 1e-6)
        output = output.reshape(batch, seq_len, self.output_dim)
        if squeezed:
            output = output[0]
        return output.astype(np.float32)
