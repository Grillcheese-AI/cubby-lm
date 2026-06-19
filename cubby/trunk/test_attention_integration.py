"""Integration test: full model with attention enabled (0.0.2 config)."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from cubby.config import make_config
from cubby.trunk.model import CubbyLM

np.random.seed(42)

# Test with attention config
cfg = make_config('0.0.2', vocab_size=512, d_model=256, n_layers=6, d_ffn=512, seq_len=64)
print(f"Config: enable_attention={cfg.enable_attention}, attn_heads={cfg.attn_heads}, "
      f"attn_window={cfg.attn_window}, attn_every_n={cfg.attn_every_n}")
print(f"Expected attention layers: {[i for i in range(cfg.n_layers) if i % cfg.attn_every_n == 0]}")

model = CubbyLM(cfg)

# Verify attention is inserted in the right layers
for i, b in enumerate(model.blocks):
    print(f"  Block {i}: has_attn={b.has_attn}")

# Forward + backward pass
B, S = 2, 64
ids = np.random.randint(0, cfg.total_vocab, (B, S)).astype(np.int64)
tgts = np.random.randint(0, cfg.total_vocab, (B, S)).astype(np.int64)

loss = model.loss(ids, tgts)
print(f"\nForward: loss={float(loss.data):.4f}")

# Backward
for p in model.parameters():
    p.grad = None
loss.backward()

# Check all params got gradients
n_with_grad = 0
n_total = 0
for i, b in enumerate(model.blocks):
    n_total += 2  # n1, n2
    if b.n1.grad is not None:
        n_with_grad += 1
    if b.n2.grad is not None:
        n_with_grad += 1
    if b.has_attn:
        n_total += 1  # rms_attn
        if b.rms_attn.grad is not None:
            n_with_grad += 1
        # QKV + out_proj each have 1 param
        for pname in ['attn.qkv', 'attn.out_proj']:
            n_total += 1
            p_obj = getattr(b, 'attn')
            if pname == 'attn.qkv':
                g = p_obj.qkv.weight.grad
            else:
                g = p_obj.out_proj.weight.grad
            if g is not None:
                n_with_grad += 1

print(f"Params with grad: {n_with_grad}/{n_total}")
all_have_grad = all(b.n1.grad is not None and b.n2.grad is not None and b.ffn.down.weight.grad is not None
                    and (not b.has_attn or b.rms_attn.grad is not None)
                    and (not b.has_attn or b.attn.qkv.weight.grad is not None)
                    and (not b.has_attn or b.attn.out_proj.weight.grad is not None)
                    for b in model.blocks)
print(f"All trainables have grad: {'PASS' if all_have_grad else 'FAIL'}")

# Compare with attention OFF
cfg_no_attn = make_config('0.0.0', vocab_size=512, d_model=256, n_layers=6, d_ffn=512, seq_len=64)
model_no_attn = CubbyLM(cfg_no_attn)
loss_ref = model_no_attn.loss(ids, tgts)
print(f"\nAttention OFF: loss={float(loss_ref.data):.4f}")
print(f"  Attention ON adds {sum(p.data.size for b in model.blocks if b.has_attn for p in b.attn.parameters())} params (attention)")
print(f"\nINTEGRATION TEST COMPLETE")
