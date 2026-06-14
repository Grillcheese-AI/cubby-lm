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


def make_tokenizer(kind: str = "byte", path: str | None = None):
    if kind == "byte":
        return ByteTokenizer()
    if kind in ("bpe", "bbpe65k"):
        return BPETokenizer(path or BBPE65K_PATH)
    raise ValueError(f"unknown tokenizer kind {kind!r}")
