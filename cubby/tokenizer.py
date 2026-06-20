"""Tokenizers for Cubby.

- ByteTokenizer      : vocab 256, the validated control (no deps); assembles words
                       byte-by-byte, so small/short runs produce phonetic misspellings.
- BPETokenizer       : wraps a HuggingFace `tokenizers` JSON -- the production
                       BBPE-65k v3 artifact. Each id is a wordpiece, so generation
                       emits valid fragments (no intra-word garble); errors become
                       word-choice/grammar, not spelling.
- MultilingualBPE    : our custom byte-level multilingual BPE, trained on the
                       unified corpus (C4 multilingual, Wiki EN/FR, math, agent data).
                       ~32k vocab with AST/CubeLang special tokens registered as
                       atomic entries so they never fragment.

Uniform interface: .encode(text)->list[int], .decode(ids)->str, .vocab_size.
"""
from __future__ import annotations

from pathlib import Path

BBPE65K_PATH = r"E:\AITEMP\grillcheese_bbpe65k_v3.json"
# default path for the multilingual BPE artifact (built by cubby/tools/train_tokenizer.py)
_MultilingualBPE_DIR = Path(__file__).parent / "tokenizers" / "cubby_mbpe32k"
MultilingualBPE_PATH = _MultilingualBPE_DIR / "tokenizer.json"


class ByteTokenizer:
    vocab_size = 256
    kind = "byte"

    def encode(self, text: str):
        return list(text.encode("utf-8", "ignore"))

    def decode(self, ids, skip_special=True):
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

    def decode(self, ids, skip_special=True):
        return self.tok.decode([int(i) for i in ids], skip_special_tokens=skip_special)


class MultilingualBPE:
    """Custom byte-level multilingual BPE (~32k vocab).

    Loaded from a HuggingFace tokenizer.json artifact (our own, built by
    cubby/tools/train_tokenizer.py on the unified training corpus). AST/CubeLang
    special tokens are registered as atomic entries so the language head and the
    AST head see disjoint, non-overlapping IDs — the AST region is the tail end
    of the vocab, starting at `ast_start_id`.
    """
    kind = "multilingual_bpe"

    def __init__(self, path: str | Path | None = None):
        from tokenizers import Tokenizer, AddedToken
        path = path or MultilingualBPE_PATH
        self.tok = Tokenizer.from_file(str(path))
        self.vocab_size = self.tok.get_vocab_size()
        self.path = str(path)
        # scan added tokens
        added = self.tok.get_added_tokens_decoder()
        self._added_tokens = {id_: tok.content for id_, tok in added.items()}
        self._name_to_id = {tok.content: id_ for id_, tok in added.items()}
        self.n_special_tokens = len(self._added_tokens)
        # AST tokens: CubeLang opcodes + AST structure tags. These are a small,
        # explicit set — NOT a range, since special tokens are interleaved with
        # chat/structural markers at the front of the vocab.
        _ast_exact = {
            # CubeLang VM opcodes
            "BIND_ROLE", "UNBIND_ROLE", "REBIND_ROLE",
            "MATCH", "PREDICT", "DISCOVER",
            "ANALOGY", "TEMPORAL_BIND", "UNIFY",
            "SUM", "SELECT", "GROUP", "SORT", "JOIN",
            "MERGE", "SPLIT", "FILTER", "MAP_ROLES", "REDUCE",
            "INST", "GEN", "NEWVAR", "SKIP",
            "BIND", "UNBIND", "QUERY",
            # AST structure tags
            "<OPCODE>", "</OPCODE>",
            "<TASK:SCHEMA2RULE>",
            "<SCHEMA>", "</SCHEMA>",
            "<ROLES>", "</ROLES>",
            "<TRACE>", "</TRACE>",
            "<RULE>", "</RULE>",
            "<VALID>", "</VALID>",
            # agent/domain concepts used in AST framing
            "AGENT", "ACTION", "OBJECT", "QUANTITY",
            "SOURCE", "DESTINATION", "CONTEXT", "STATE",
        }
        self.ast_token_ids = frozenset(
            id_ for id_, content in self._added_tokens.items()
            if content in _ast_exact
        )
        self.n_ast_tokens = len(self.ast_token_ids)
        # language vocab = everything not in ast_token_ids (most of the 32k BPE)
        self.lang_vocab_size = self.vocab_size - self.n_ast_tokens

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text).ids

    def decode(self, ids, skip_special=True) -> str:
        return self.tok.decode([int(i) for i in ids], skip_special_tokens=skip_special)

    def is_ast_token(self, token_id: int) -> bool:
        return int(token_id) in self.ast_token_ids

    def token_name(self, token_id: int) -> str | None:
        return self._added_tokens.get(int(token_id))


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

    def decode(self, ids, skip_special=True):
        inv = self.inv
        return self.base.decode([inv[i] for i in (int(x) for x in ids) if i in inv], skip_special)


class DualHeadRemap:
    """Wraps a tokenizer whose AST/CubeLang token ids are scattered (interleaved at
    the front of the vocab) so they become a CONTIGUOUS TAIL [V-n_ast, V).

    The dual-head trunk assumes `id < Vlang` == language and `id >= Vlang` == AST
    (separate embed_lang/embed_ast tables, a contiguous logit split, and the
    `tgt >= Vlang` loss classification). MultilingualBPE breaks that: its
    `ast_token_ids` sit at low ids (~20-66) while `lang_vocab_size` is just a COUNT,
    so `tgt >= Vlang` selects the 47 *rarest BPE fragments* instead of the AST
    tokens -- the AST head trains on garbage and opcodes route to the language head.

    Fix without touching the (parity-validated) model math: a bijective INVOLUTION
    that swaps the n_ast AST ids with the n_ast highest ids. Applied in both
    encode and decode (self-inverse), it is transparent everywhere else and makes
    the contiguity assumption true. Requires a retrain (token ids change).
    """
    kind = "multilingual_bpe"

    def __init__(self, base):
        self.base = base
        V, A = base.vocab_size, sorted(base.ast_token_ids)
        n = len(A)
        tail = list(range(V - n, V))
        self.swap = {}
        for a, t in zip(A, tail):
            if a != t:
                self.swap[a] = t
                self.swap[t] = a                       # involution
        self.vocab_size = V
        self.n_ast_tokens = n
        self.n_special_tokens = getattr(base, "n_special_tokens", n)
        self.lang_vocab_size = V - n
        self.ast_token_ids = frozenset(tail)

    def _map(self, ids):
        s = self.swap
        return [s.get(int(i), int(i)) for i in ids]

    def encode(self, text):
        return self._map(self.base.encode(text))

    def decode(self, ids, skip_special=True):
        return self.base.decode(self._map(ids), skip_special)

    def is_ast_token(self, token_id):
        return int(token_id) in self.ast_token_ids


def make_tokenizer(kind: str = "byte", path: str | None = None):
    if kind == "byte":
        return ByteTokenizer()
    if kind in ("bpe", "bbpe65k"):
        return BPETokenizer(path or BBPE65K_PATH)
    if kind in ("multilingual_bpe", "mbpe", "mbpe32k"):
        # Remap so AST ids are a contiguous tail -> the dual-head's id-range
        # split (id >= Vlang == AST) is then correct.
        return DualHeadRemap(MultilingualBPE(path))
    raise ValueError(f"unknown tokenizer kind {kind!r}")
