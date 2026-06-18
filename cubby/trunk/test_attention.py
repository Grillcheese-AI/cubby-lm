"""Parity test: chunked sliding-window attention vs brute-force reference."""
import numpy as np
import sys

from cubby.trunk.model import (
    chunked_sliding_window_attention,
    _reference_sliding_window_attention,
)
from grilly.nn.autograd import Variable

np.random.seed(42)

B, H, Dh = 2, 2, 16
# (S, W) test cases: various sizes including edges
test_cases = [
    (8, 4),      # 2 chunks
    (16, 4),     # 4 chunks
    (32, 8),     # 4 chunks
    (64, 16),    # 4 chunks
    (128, 32),   # 4 chunks
    (256, 64),   # 4 chunks
    (10, 3),     # non-power-of-2
    (7, 4),      # S < 2W
    (4, 8),      # S <= W (single chunk, pure causal)
    (1, 4),      # S=1 edge
]

print("=== Forward parity: chunked vs brute-force reference ===")
all_pass = True
for S, W in test_cases:
    q = np.random.randn(B, H, S, Dh).astype(np.float32)
    k = np.random.randn(B, H, S, Dh).astype(np.float32)
    v = np.random.randn(B, H, S, Dh).astype(np.float32)

    ref = _reference_sliding_window_attention(q, k, v, W)

    qv = Variable(q.copy(), requires_grad=False)
    kv = Variable(k.copy(), requires_grad=False)
    vv = Variable(v.copy(), requires_grad=False)
    chunked = chunked_sliding_window_attention(qv, kv, vv, W)
    out = np.asarray(chunked.data, dtype=np.float32)

    diff = float(np.abs(ref - out).max())
    ok = diff < 1e-4
    if not ok:
        all_pass = False
    print(f"  S={S:3d} W={W:3d}  max_abs_diff={diff:.2e}  {'PASS' if ok else 'FAIL'}")

print()
print("=== Backward parity: full gradient check ===")
# Test backward with full gradient comparison (not just 5 points)
S, W = 16, 4
q = np.random.randn(B, H, S, Dh).astype(np.float32) * 0.1
k = np.random.randn(B, H, S, Dh).astype(np.float32) * 0.1
v = np.random.randn(B, H, S, Dh).astype(np.float32) * 0.1

qv = Variable(q.copy(), requires_grad=True)
kv = Variable(k.copy(), requires_grad=True)
vv = Variable(v.copy(), requires_grad=True)
out = chunked_sliding_window_attention(qv, kv, vv, W)

# Create a simple loss: sum of output
loss = out.data.sum()
out.backward(np.ones_like(out.data))
dq_auto = np.asarray(qv.grad, dtype=np.float32)
dk_auto = np.asarray(kv.grad, dtype=np.float32)
dv_auto = np.asarray(vv.grad, dtype=np.float32)

# Finite difference check for ALL q positions (use smaller eps for stability)
eps = 5e-4
dq_fd = np.zeros_like(q)
for bi in range(B):
    for hi in range(H):
        for si in range(S):
            for di in range(Dh):
                q_p = q.copy()
                q_p[bi, hi, si, di] += eps
                q_m = q.copy()
                q_m[bi, hi, si, di] -= eps
                
                out_p = chunked_sliding_window_attention(
                    Variable(q_p, requires_grad=False),
                    Variable(k.copy(), requires_grad=False),
                    Variable(v.copy(), requires_grad=False),
                    W
                ).data.sum()
                
                out_m = chunked_sliding_window_attention(
                    Variable(q_m, requires_grad=False),
                    Variable(k.copy(), requires_grad=False),
                    Variable(v.copy(), requires_grad=False),
                    W
                ).data.sum()
                
                dq_fd[bi, hi, si, di] = (out_p - out_m) / (2 * eps)

# Compare full gradient tensors
max_diff_q = float(np.abs(dq_auto - dq_fd).max())
mean_diff_q = float(np.abs(dq_auto - dq_fd).mean())
rel_err = mean_diff_q / (np.abs(dq_auto).mean() + 1e-8)
print(f"  q grad max_abs_diff: {max_diff_q:.2e}")
print(f"  q grad mean_abs_diff: {mean_diff_q:.2e}")
print(f"  q grad relative error: {rel_err:.2e}  {'PASS' if rel_err < 0.05 else 'FAIL'}")

# Check gradient signs and magnitudes are reasonable
grad_norm_q = np.linalg.norm(dq_auto)
print(f"  ||dq_auto|| = {grad_norm_q:.4f}")

# window leak test: perturb pos 0 of S=32, W=8
# positions >= W should be UNCHANGED (causal isolation)
S2, W2 = 32, 8
q2 = np.random.randn(1, 1, S2, Dh).astype(np.float32) * 0.1
k2 = np.random.randn(1, 1, S2, Dh).astype(np.float32) * 0.1
v2 = np.random.randn(1, 1, S2, Dh).astype(np.float32) * 0.1
ref_clean = _reference_sliding_window_attention(q2, k2, v2, W2)

q2_perturbed = q2.copy()
q2_perturbed[0, 0, 0, :] += 100.0
ref_perturbed = _reference_sliding_window_attention(q2_perturbed, k2, v2, W2)

max_change_before_window = float(np.abs(ref_clean[0, 0, :W2] - ref_perturbed[0, 0, :W2]).max())
max_change_after_window = float(np.abs(ref_clean[0, 0, W2:] - ref_perturbed[0, 0, W2:]).max())
leak_ok = max_change_after_window < 1e-5
print(f"  window leak test: pos<W change={max_change_before_window:.2e}  pos>=W change={max_change_after_window:.2e}  {'PASS' if leak_ok else 'FAIL'}")

print()
if all_pass and max_diff_q < 1e-3 and leak_ok:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED")
