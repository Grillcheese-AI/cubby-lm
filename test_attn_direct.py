"""Direct test of attention forward ops."""
import sys
sys.path.insert(0, 'C:/Users/grill/Documents/GitHub/grilly')
import numpy as np
import grilly_core as gc
from cubby.trunk.resident import make_device

dev = make_device()
t = gc.TapeContext(dev)
B, H, S, Dh, W = 1, 1, 4, 8, 4
elem = B * H * S * Dh

# Upload Q, K, V
q = t.register_input(np.random.randn(elem).astype(np.float32))
k = t.register_input(np.random.randn(elem).astype(np.float32))
v = t.register_input(np.random.randn(elem).astype(np.float32))

# Test forward_chunked_attention
t.forward_begin()
out = t.forward_chunked_attention(q, k, v, B, H, S, Dh, W)
t.forward_submit()
print(f'forward_chunked_attention OK, out_id: {out}', flush=True)

# Read back output
result = t.read_buffer(out, [B, H, S, Dh])
print(f'output shape: {result.shape}, sample: {result[0, 0, 0, :3]}', flush=True)

# Test forward_qkv_split
qkv_buf = t.register_input(np.random.randn(B * S * 3 * H * Dh).astype(np.float32))
t.forward_begin()
q_id, k_id, v_id = t.forward_qkv_split(qkv_buf, B, S, H, Dh)
t.forward_submit()
print(f'forward_qkv_split OK: q={q_id} k={k_id} v={v_id}', flush=True)

# Test forward_transpose_bhsd_bshd
in_buf = t.register_input(np.random.randn(elem).astype(np.float32))
t.forward_begin()
out_buf = t.forward_transpose_bhsd_bshd(in_buf, B, H, S, Dh)
t.forward_submit()
print(f'forward_transpose_bhsd_bshd OK: {out_buf}', flush=True)
print('All forward attention ops pass!', flush=True)
