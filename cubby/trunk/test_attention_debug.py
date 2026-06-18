"""Debug gradient flow through attention layers."""
import numpy as np
from cubby.config import make_config
from cubby.trunk.model import CubbyLM

np.random.seed(42)

cfg = make_config('0.0.2', vocab_size=512, d_model=256, n_layers=6, d_ffn=512, seq_len=64)
model = CubbyLM(cfg)

B, S = 2, 64
ids = np.random.randint(0, cfg.total_vocab, (B, S)).astype(np.int64)
tgts = np.random.randint(0, cfg.total_vocab, (B, S)).astype(np.int64)

loss = model.loss(ids, tgts)
print(f"Forward: loss={float(loss.data):.4f}")

# Clear gradients
for p in model.parameters():
    p.grad = None

# Backward
loss.backward()

# Check each block's gradients
for i, b in enumerate(model.blocks):
    print(f"\nBlock {i} (has_attn={b.has_attn}):")
    print(f"  n1.grad: {b.n1.grad is not None}")
    print(f"  mix.proj.weight.grad: {b.mix.proj.weight.grad is not None}")

    if b.has_attn:
        print(f"  rms_attn.grad: {b.rms_attn.grad is not None}")
        print(f"  attn.qkv.weight.grad: {b.attn.qkv.weight.grad is not None}")
        print(f"  attn.out_proj.weight.grad: {b.attn.out_proj.weight.grad is not None}")
    else:
        print(f"  (no attention)")

    print(f"  n2.grad: {b.n2.grad is not None}")
    print(f"  ffn.gate_up.weight.grad: {b.ffn.gate_up.weight.grad is not None}")
    print(f"  ffn.down.weight.grad: {b.ffn.down.weight.grad is not None}")

print("\nDone")
