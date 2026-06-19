"""Test tiny_mbpe config with attention + dual head."""
import sys, io
sys.stdout = sys.stderr = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, r'C:\Users\grill\Documents\GitHub\grilly')
sys.path.insert(0, r'C:\Users\grill\Documents\GitHub\cubby-lm')
import numpy as np
from cubby.config import make_config
from cubby.trunk.model import CubbyLM
from cubby.trunk.resident import ResidentTrunk
from cubby.tokenizer import make_tokenizer

tok = make_tokenizer('multilingual_bpe')
cfg = make_config('tiny_mbpe', vocab_size=tok.lang_vocab_size, n_special_tokens=tok.n_ast_tokens)
print(f'Config: V={cfg.total_vocab} d={cfg.d_model} L={cfg.n_layers} attn={cfg.enable_attention} dual={cfg.enable_dual_head}')
print(f'  attn_heads={cfg.attn_heads} attn_window={cfg.attn_window} attn_every_n={cfg.attn_every_n}')

m = CubbyLM(cfg)
rt = ResidentTrunk(m)
print(f'Resident: V={rt.V} d={rt.d} L={rt.L} has_attn={rt.has_attn} attn_n_heads={rt.attn_n_heads}')
print(f'  attn layers: {[li for li in range(rt.L) if rt.layers[li].get("has_attn", False)]}')

ids = np.random.randint(0, cfg.total_vocab, (4, 128), dtype=np.int64)
tgts = np.random.randint(0, cfg.total_vocab, (4, 128), dtype=np.int64)
print(f'Batch: {ids.shape}')

# Debug: check which weights have grad=0 before calling train_step
print('Checking weight registration:')
for i, p in enumerate(rt.opt):
    name = '?'
    if p is rt.E: name = 'E'
    elif p is rt.final: name = 'final'
    elif p is getattr(rt, 'router', None): name = 'router'
    else:
        for li in range(rt.L):
            for k in rt._WKEYS:
                if p is rt.layers[li][k]: name = f'L{li}.{k}'
            if rt.layers[li].get('has_attn', False):
                for k in ['rms_attn', 'qkv', 'out_proj']:
                    if p is rt.layers[li][k]: name = f'L{li}.{k}'
    print(f'  opt[{i}]: {name} w={p["w"]} n={p["n"]}')

result = rt.train_step(ids, tgts, step=1)
loss = result[0] if isinstance(result, tuple) else result
gnorm = result[1] if isinstance(result, tuple) and len(result) > 1 else '?'
print(f'train_step OK: loss={loss:.4f} gnorm={gnorm}')
print('SUCCESS')