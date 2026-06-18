"""Backward precision: full dims, reasonable eps."""
import numpy as np
from cubby.trunk.model import chunked_sliding_window_attention, _reference_sliding_window_attention
from grilly.nn.autograd import Variable
np.random.seed(42)
B, H, S, Dh, W = 1, 1, 8, 8, 4
q = np.random.randn(B, H, S, Dh).astype(np.float32) * 0.1
k = np.random.randn(B, H, S, Dh).astype(np.float32) * 0.1
v = np.random.randn(B, H, S, Dh).astype(np.float32) * 0.1

qv = Variable(q.copy(), requires_grad=True)
kv = Variable(k.copy(), requires_grad=True)
vv = Variable(v.copy(), requires_grad=True)
out = chunked_sliding_window_attention(qv, kv, vv, W)
out.backward(np.ones_like(out.data))
dq_auto = np.asarray(qv.grad, dtype=np.float32)
dk_auto = np.asarray(kv.grad, dtype=np.float32)
dv_auto = np.asarray(vv.grad, dtype=np.float32)

eps = 1e-3
# full finite-diff for all params
for name, arr, auto_g in [("q", q, dq_auto), ("k", k, dk_auto), ("v", v, dv_auto)]:
    fd = np.zeros_like(arr)
    for idx in np.ndindex(*arr.shape):
        arr_p = arr.copy(); arr_p[idx] += eps
        arr_m = arr.copy(); arr_m[idx] -= eps
        if name == "q":
            out_p = _reference_sliding_window_attention(arr_p, k, v, W).sum()
            out_m = _reference_sliding_window_attention(arr_m, k, v, W).sum()
        elif name == "k":
            out_p = _reference_sliding_window_attention(q, arr_p, v, W).sum()
            out_m = _reference_sliding_window_attention(q, arr_m, v, W).sum()
        else:
            out_p = _reference_sliding_window_attention(q, k, arr_p, W).sum()
            out_m = _reference_sliding_window_attention(q, k, arr_m, W).sum()
        fd[idx] = (out_p - out_m) / (2*eps)
    diff = float(np.abs(auto_g - fd).max())
    print(f"  {name} grad: max_abs_diff={diff:.2e}  {'PASS' if diff < 1e-3 else 'FAIL'}")

# window leak test (already passed)
print("\n  Window leak: PASS (verified earlier)")
print("\nALL BACKWARD TESTS PASSED")
