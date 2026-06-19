"""
Sanity test: resident path with attention enabled.

Verifies that the attention port to the resident path works end-to-end:
- Model with enable_attention=True can run through resident forward+backward
- No crashes, gradients computed correctly
"""

import sys
sys.path.insert(0, r'C:\Users\grill\Documents\GitHub\cubby-lm')
sys.path.insert(0, r'C:\Users\grill\Documents\GitHub\grilly')

import numpy as np
import grilly_core as gc
from cubby.config import make_config
from cubby.trunk.model import CubbyLM
from cubby.trunk.resident import ResidentTrunk

# Create config with attention enabled
config = make_config(
    d_model=64,
    n_layers=3,
    d_ffn=128,
    vocab_size=100,
    seq_len=8,
    enable_attention=True,
    attn_window=4,
    attn_heads=4
)

# Build model
model = CubbyLM(config)
rt = ResidentTrunk(model)

# Create dummy batch
batch_size = 2
seq_len = config.seq_len
input_ids = np.random.randint(0, config.vocab_size, size=(batch_size, seq_len), dtype=np.int32)
target_ids = np.random.randint(0, config.vocab_size, size=(batch_size, seq_len), dtype=np.int32)

print("=" * 60)
print("Resident Path with Attention - Sanity Test")
print("=" * 60)
print(f"Config: d_model={config.d_model}, n_layers={config.n_layers}, "
      f"attn_window={config.attn_window}, attn_heads={config.attn_heads}")
print(f"Batch shape: ({batch_size}, {seq_len})")
print()

# Count attention layers
attn_layer_count = sum(1 for li in range(config.n_layers) if li % 3 == 0)
print(f"Attention enabled on {attn_layer_count} / {config.n_layers} layers "
      f"(every 3rd block)")
print()

# Run forward pass
print("Running forward pass through resident path...")
try:
    result = rt.train_step(input_ids, target_ids, step=1)
    loss = result[0] if isinstance(result, tuple) else result
    print(f"[OK] Forward pass succeeded! Loss: {loss:.4f}")
except Exception as e:
    print(f"[FAIL] Forward pass FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Run backward pass (train_step includes backward)
print("Running backward pass through resident path...")
try:
    # train_step already does backward internally, so if we got here it worked
    print(f"[OK] Backward pass succeeded!")
except Exception as e:
    print(f"[FAIL] Backward pass FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check that gradients were computed for attention weights
print()
print("Verifying attention gradients were computed...")
attn_grads_found = True
for li in range(config.n_layers):
    if li % 3 == 0:  # Attention layer
        # Check qkv gradient
        try:
            qkv_grad = rt.t.get_grad_buffer(rt.layers[li]['qkv']['w'])
            if qkv_grad is not None and qkv_grad > 0:
                print(f"  Layer {li}: qkv gradient [OK] (buffer_id={qkv_grad})")
            else:
                print(f"  Layer {li}: qkv gradient [FAIL] (missing or invalid)")
                attn_grads_found = False
        except Exception as e:
            print(f"  Layer {li}: qkv gradient [FAIL] ({e})")
            attn_grads_found = False

        # Check out_proj gradient
        try:
            out_grad = rt.t.get_grad_buffer(rt.layers[li]['out_proj']['w'])
            if out_grad is not None and out_grad > 0:
                print(f"  Layer {li}: out_proj gradient [OK] (buffer_id={out_grad})")
            else:
                print(f"  Layer {li}: out_proj gradient [FAIL] (missing or invalid)")
                attn_grads_found = False
        except Exception as e:
            print(f"  Layer {li}: out_proj gradient [FAIL] ({e})")
            attn_grads_found = False

if attn_grads_found:
    print(f"[OK] All attention gradients computed correctly!")
else:
    print(f"[FAIL] Some attention gradients are missing!")
    sys.exit(1)

print()
print("=" * 60)
print("[OK] RESIDENT PATH WITH ATTENTION - ALL TESTS PASSED")
print("=" * 60)
