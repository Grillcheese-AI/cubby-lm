"""TinyStories training for the 0.0.0 trunk (the coherence gate).

Tokenizer-driven: `--tok byte` (vocab 256 control) or `--tok bbpe65k` (the real
subword tokenizer -- valid wordpieces, so no intra-word garble). Concatenates
stories into one token stream, trains next-token prediction, prints the loss
curve and a decoded sample every `sample_every` steps. Forward is CPU numpy
(Variable matmul); the MinGRU scan is the GPU kernel. Keep the model small here
-- this is a learning/coherence smoke, not the scaled GPU run (0.0.1).
"""
from __future__ import annotations

import json
import time

import numpy as np

import grilly  # noqa: F401
from cubby.config import make_config
from cubby.tokenizer import make_tokenizer
from cubby.trunk.model import CubbyLM, AdamW, param_count


def load_stream(path, tokenizer, max_tokens):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ids = []
    for story in data:
        ids.extend(tokenizer.encode(story + "\n"))
        if len(ids) >= max_tokens:
            break
    return np.asarray(ids[:max_tokens], dtype=np.int64)


def get_batch(stream, B, S, rng):
    ix = rng.integers(0, len(stream) - S - 1, size=B)
    ids = np.stack([stream[i:i + S] for i in ix])
    tgt = np.stack([stream[i + 1:i + 1 + S] for i in ix])
    return ids, tgt


def sample(model, tokenizer, seed_text, n=80, temperature=0.8):
    seed = np.asarray(tokenizer.encode(seed_text), dtype=np.int64)
    out = model.generate(seed, max_new_tokens=n, temperature=temperature)
    return tokenizer.decode(out)


def train(steps=600, B=16, S=96, d_model=128, n_layers=4, d_ffn=256, lr=3e-3,
          ffn_type="swiglu", tok="byte", tok_path=None, max_tokens=600_000,
          sample_every=150, seed=0, data_path="tinystory_50k.json"):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    tokenizer = make_tokenizer(tok, tok_path)
    cfg = make_config("0.0.0", vocab_size=tokenizer.vocab_size, d_model=d_model,
                      n_layers=n_layers, d_ffn=d_ffn, seq_len=S, ffn_type=ffn_type)
    model = CubbyLM(cfg)
    opt = AdamW(model.parameters(), lr=lr)
    stream = load_stream(data_path, tokenizer, max_tokens)
    print(f"[setup] tok={tok}(V={tokenizer.vocab_size}) ffn={ffn_type} "
          f"params={param_count(model):,} stream={len(stream):,} toks "
          f"d={d_model} L={n_layers} B={B} S={S}", flush=True)

    t0, ema = time.time(), None
    for step in range(1, steps + 1):
        ids, tgt = get_batch(stream, B, S, rng)
        loss = model.loss(ids, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        lv = float(loss.data)
        ema = lv if ema is None else 0.98 * ema + 0.02 * lv
        if step % 25 == 0 or step == 1:
            print(f"[{step:4d}/{steps}] loss={lv:.3f} ema={ema:.3f} "
                  f"({step/(time.time()-t0):.1f} it/s)", flush=True)
        if step % sample_every == 0:
            print("  sample:", repr(sample(model, tokenizer, "Once upon a time")), flush=True)
    print(f"[done] {time.time()-t0:.1f}s  final ema loss={ema:.3f}", flush=True)
    return model


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", default="byte")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--S", type=int, default=96)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--ffn", default="swiglu")
    ap.add_argument("--sample_every", type=int, default=150)
    a = ap.parse_args()
    train(steps=a.steps, B=a.B, S=a.S, d_model=a.d, n_layers=a.L,
          ffn_type=a.ffn, tok=a.tok, sample_every=a.sample_every)
