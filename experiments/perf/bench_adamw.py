import time
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
import grilly
from grilly.backend import _bridge
from cubby.config import make_config
from cubby.trunk.model import CubbyLM

cfg = make_config("0.0.0", vocab_size=256, d_model=512, n_layers=8, d_ffn=2048, seq_len=128)
m = CubbyLM(cfg)
params = list(m.parameters())
rng = np.random.default_rng(0)
W = [np.asarray(p.data, np.float32).copy() for p in params]
G = [rng.standard_normal(w.shape).astype(np.float32) * 0.01 for w in W]
M = [np.zeros_like(w) for w in W]
V = [np.zeros_like(w) for w in W]
nparams = sum(w.size for w in W)
print(f"{len(W)} param tensors, {nparams/1e6:.1f}M params")

lr, b1, b2, eps, wd = 3e-3, 0.9, 0.999, 1e-8, 0.01

def numpy_step(t):
    bc1 = 1 - b1**t; bc2 = 1 - b2**t
    for i in range(len(W)):
        M[i] = b1*M[i] + (1-b1)*G[i]
        V[i] = b2*V[i] + (1-b2)*(G[i]*G[i])
        W[i] = W[i] - lr*((M[i]/bc1)/(np.sqrt(V[i]/bc2)+eps) + wd*W[i])

def gpu_step(t):
    b1t = b1**t; b2t = b2**t
    for i in range(len(W)):
        r = _bridge.adamw_update(W[i], G[i], M[i], V[i], lr, b1, b2, eps, wd, b1t, b2t, False)
        W[i] = np.asarray(r["weights"], np.float32)
        M[i] = np.asarray(r["m"], np.float32)
        V[i] = np.asarray(r["v"], np.float32)

def inplace_step(t):
    bc1 = 1 - b1**t; bc2 = 1 - b2**t
    c1 = 1 - b1; c2 = 1 - b2
    for i in range(len(W)):
        m_, v_, w_, g_ = M[i], V[i], W[i], G[i]
        m_ *= b1; m_ += c1 * g_
        v_ *= b2; v_ += c2 * (g_ * g_)
        denom = np.sqrt(v_ / bc2); denom += eps
        w_ *= (1 - lr * wd)
        w_ -= lr * (m_ / bc1) / denom

numpy_step(1); gpu_step(1); inplace_step(1)  # warmup
t = time.time()
for k in range(20): numpy_step(k+2)
tn = (time.time()-t)/20
t = time.time()
for k in range(20): gpu_step(k+2)
tg = (time.time()-t)/20
t = time.time()
for k in range(20): inplace_step(k+2)
ti = (time.time()-t)/20
print(f"numpy AdamW (alloc):   {tn*1000:.1f} ms/step")
print(f"gpu   AdamW:           {tg*1000:.1f} ms/step")
print(f"numpy AdamW (inplace): {ti*1000:.1f} ms/step")
print("winner:", min([("numpy", tn), ("gpu", tg), ("inplace", ti)], key=lambda x: x[1])[0])
