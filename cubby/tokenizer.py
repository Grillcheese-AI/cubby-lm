"""Tokenizers for Cubby.

- ByteTokenizer : vocab 256, the validated control (no deps); assembles words
  byte-by-byte, so small/short runs produce phonetic misspellings.
- BPETokenizer  : wraps a HuggingFace `tokenizers` JSON -- the production
  BBPE-65k v3 artifact. Each id is a wordpiece, so generation emits valid
  fragments (no intra-word garble); errors become word-choice/grammar, not
  spelling. This is the tokenizer the ladder targets (+ AST special tokens later).

Uniform interface: .encode(text)->list[int], .decode(ids)->str, .vocab_size.
"""
from __future__ import annotations

BBPE65K_PATH = r"E:\AITEMP\grillcheese_bbpe65k_v3.json"


class ByteTokenizer:
    vocab_size = 256
    kind = "byte"

    def encode(self, text: str):
        return list(text.encode("utf-8", "ignore"))

    def decode(self, ids):
        return bytes(int(i) & 0xFF for i in ids).decode("utf-8", "replace")


class BPETokenizer:
    kind = "bpe"

    def __init__(self, path: str = BBPE65K_PATH):
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(path)
        self.vocab_size = self.tok.get_vocab_size()
        self.path = path

    def encode(self, text: str):
        return self.tok.encode(text).ids

    def decode(self, ids):
        return self.tok.decode([int(i) for i in ids])


class RemapTokenizer:
    """Wraps a base tokenizer, compressing its ids into a dense ``[0, vocab_size)``
    space built from corpus frequency. The ``vocab_size-1`` most frequent base ids
    get dedicated dense slots; every rarer id folds to dense 0 (``<unk>``).

    This lets a small model vocab ride the full BBPE-65k wordpieces when the corpus
    only touches a few thousand of them -- e.g. TinyStories uses 11.6k unique ids and
    the top-10k cover 99.96% of all occurrences, so a 10k dense vocab is lossless in
    practice while cutting the tied head/embedding params ~6x. Generation quality is
    unchanged (still wordpieces); only the id space is renumbered.
    """
    kind = "remap"

    def __init__(self, base, vocab_size: int, corpus_ids):
        import numpy as np
        self.base = base
        self.vocab_size = int(vocab_size)
        vals, counts = np.unique(np.asarray(corpus_ids), return_counts=True)
        keep = vals[np.argsort(-counts)[: self.vocab_size - 1]]   # reserve dense 0 = <unk>
        self.fwd = {int(b): i + 1 for i, b in enumerate(keep)}    # base id -> dense
        self.inv = {i + 1: int(b) for i, b in enumerate(keep)}    # dense -> base id
        self.coverage = float(counts[np.isin(vals, keep)].sum() / counts.sum())

    def encode_base(self, base_ids):
        f = self.fwd
        return [f.get(int(b), 0) for b in base_ids]

    def encode(self, text: str):
        return self.encode_base(self.base.encode(text))

    def decode(self, ids):
        inv = self.inv
        return self.base.decode([inv[i] for i in (int(x) for x in ids) if i in inv])


def make_tokenizer(kind: str = "byte", path: str | None = None):
    if kind == "byte":
        return ByteTokenizer()
    if kind in ("bpe", "bbpe65k"):
        return BPETokenizer(path or BBPE65K_PATH)
    raise ValueError(f"unknown tokenizer kind {kind!r}")
