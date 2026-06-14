import cProfile, pstats, io
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
import grilly  # noqa
from cubby.config import make_config
from cubby.trunk.model import CubbyLM, AdamW

cfg = make_config("0.0.0", vocab_size=256, d_model=512, n_layers=8, d_ffn=2048, seq_len=128)
m = CubbyLM(cfg)
opt = AdamW(m.parameters(), lr=3e-3)
rng = np.random.default_rng(0)

def step():
    ids = rng.integers(0, 256, size=(16, 128))
    tgt = rng.integers(0, 256, size=(16, 128))
    loss = m.loss(ids, tgt)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return float(loss.data)

step()  # warmup (shaders/pipelines)

pr = cProfile.Profile()
pr.enable()
for _ in range(5):
    step()
pr.disable()

s = io.StringIO()
st = pstats.Stats(pr, stream=s).sort_stats("tottime")
st.print_stats(18)
# keep only the table rows
for line in s.getvalue().splitlines():
    if "function calls" in line or "tottime" in line or "{" in line or "cubby" in line or "autograd" in line:
        print(line)
