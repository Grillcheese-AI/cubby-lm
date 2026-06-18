"""Train a small byte-level multilingual BPE tokenizer for Cubby.

Reads from the unified training corpus (D:\\grillcheese_training_data\\unified\\*.jsonl),
extracts the `text` field, and trains a ByteLevelBPETokenizer at a target vocab size
(default 32k) with AST/CubeLang/chat special tokens registered as atomic entries.

Output: a tokenizer.json artifact at cubby/tokenizers/<name>/tokenizer.json.

Usage:
    python cubby/tools/train_tokenizer.py --vocab-size 32768 --max-gb 10
    python cubby/tools/train_tokenizer.py --vocab-size 32768 --max-gb 20 --name cubby_mbpe32k_v2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

CUBBY_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(r"D:\grillcheese_training_data\unified")


# ── Special tokens (carried from the proven BBPE-65k v3 set) ───────────
# These are already present as atomic tokens in training data; registering
# them here guarantees they never fragment across BPE merges.
SPECIAL_TOKENS = [
    # structural
    "<pad>", "<unk>", "<s>", "</s>",
    # chat / role
    "<|system|>", "<|user|>", "<|assistant|>", "<|tool|>",
    "<|image|>", "<|audio|>",
    # state markup
    "[MY_STATE]", "[/MY_STATE]",
    "[INSTRUCTION]", "[/INSTRUCTION]",
    "[THINKING]", "[/THINKING]",
    "[MEMORY]", "[/MEMORY]",
    "[SPECIALIST]", "[/SPECIALIST]",
    # AST / CubeLang structure tags
    "<TASK:SCHEMA2RULE>",
    "<SCHEMA>", "</SCHEMA>",
    "<ROLES>", "</ROLES>",
    "<TRACE>", "</TRACE>",
    "<RULE>", "</RULE>",
    "<OPCODE>", "</OPCODE>",
    "<VALID>", "</VALID>",
    # CubeLang VM opcodes (45 base + 10 extended)
    "BIND_ROLE", "UNBIND_ROLE", "REBIND_ROLE",
    "MATCH", "PREDICT", "DISCOVER",
    "ANALOGY", "TEMPORAL_BIND", "UNIFY",
    "SUM", "SELECT", "GROUP", "SORT", "JOIN",
    "MERGE", "SPLIT", "FILTER", "MAP_ROLES", "REDUCE",
    "INST", "GEN", "NEWVAR", "SKIP",
    "BIND", "UNBIND", "QUERY",
    # Agent / domain concepts
    "AGENT", "ACTION", "OBJECT", "QUANTITY",
    "SOURCE", "DESTINATION", "CONTEXT", "STATE",
]


def _extract_texts(data_dir: Path, max_gb: float) -> list[str]:
    """Collect JSONL files, sorted smallest-first, accumulating up to max_gb."""
    files = sorted(data_dir.glob("*.jsonl"), key=os.path.getsize)
    total_gb = sum(f.stat().st_size for f in files) / 1e9
    chosen, running = [], 0.0
    for f in files:
        if running + f.stat().st_size / 1e9 > max_gb and running > 0:
            break
        chosen.append(f)
        running += f.stat().st_size / 1e9
    print(f"Training data: {len(chosen)} files, {running:.1f} / {total_gb:.1f} GB")
    return chosen


def _stream_texts_to_file(files: list[str], out_path: str, max_lines: int | None = None) -> int:
    """Extract `text` fields from JSONL files and write as one-text-per-line."""
    n = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for path in files:
            with open(path, "r", encoding="utf-8") as fin:
                for line in fin:
                    try:
                        obj = json.loads(line)
                        t = obj.get("text", "")
                        if t:
                            fout.write(t + "\n")
                            n += 1
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
                    if max_lines and n >= max_lines:
                        return n
    return n


def train_tokenizer(
    vocab_size: int = 32768,
    data_dir: Path = DATA_ROOT,
    max_gb: float = 10.0,
    name: str = "cubby_mbpe32k",
    min_frequency: int = 2,
    max_lines: int | None = None,
) -> Path:
    from tokenizers import ByteLevelBPETokenizer, AddedToken

    out_dir = CUBBY_ROOT / "cubby" / "tokenizers" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Extract text to a flat .txt file ────────────────────────────
    jsonl_files = _extract_texts(data_dir, max_gb)
    tmp_txt = out_dir / "_train_corpus.txt"
    print(f"Streaming texts to {tmp_txt}...")
    t0 = time.time()
    n_lines = _stream_texts_to_file([str(f) for f in jsonl_files], str(tmp_txt), max_lines)
    print(f"  {n_lines:,} lines in {time.time()-t0:.1f}s  ({tmp_txt.stat().st_size/1e9:.2f} GB)")

    # ── 2. Train byte-level BPE ────────────────────────────────────────
    from tokenizers import pre_tokenizers, decoders, normalizers

    tok = ByteLevelBPETokenizer()
    print(f"Training BPE (vocab_size={vocab_size}, min_frequency={min_frequency})...")
    t0 = time.time()
    tok.train(
        files=[str(tmp_txt)],
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=True,
        special_tokens=[],  # don't register here -- add as AddedToken below for atomicity
    )
    train_s = time.time() - t0
    print(f"  Training done in {train_s/60:.1f} min  |  actual vocab: {tok.get_vocab_size()}")

    # ── 3. Fix pre-tokenizer to match BBPE-65k config ──────────────────
    # The default ByteLevelBPETokenizer uses ByteLevel pre-tokenizer with
    # use_regex=True, which splits punctuation before special-token matching.
    # We switch to Split(regex, Isolated) + ByteLevel(use_regex=False) so that
    # special tokens with angle brackets (<OPCODE>, <|user|>, etc.) stay atomic.
    from tokenizers import pre_tokenizers, normalizers
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(
            pattern=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            behavior="Isolated",
            invert=False,
        ),
        pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False),
    ])
    tok.normalizer = normalizers.NFC()

    # ── 4. Register special tokens as AddedToken (atomic, single-piece) ─
    from tokenizers import AddedToken
    print(f"  Registering {len(SPECIAL_TOKENS)} special tokens as AddedToken...")
    tokens_to_add = []
    for sp in SPECIAL_TOKENS:
        # tokens mixing angle brackets / punctuation + letters need
        # single_word=False so they can span multiple pre-tokenizer words
        has_punct = any(c in sp for c in '<>[/|')
        tokens_to_add.append(AddedToken(
            sp, single_word=not has_punct, lstrip=False, rstrip=False,
            normalized=False, special=True,
        ))
    n_added = tok.add_special_tokens(tokens_to_add)
    print(f"  Added {n_added} special tokens  |  total vocab: {tok.get_vocab_size()}")

    # ── 5. Save ────────────────────────────────────────────────────────
    out_path = out_dir / "tokenizer.json"
    tok.save(str(out_path))
    print(f"Saved: {out_path}")

    # ── 6. Validate roundtrip on multiple scripts ──────────────────────
    _validate(tok)

    # ── 7. Write report ────────────────────────────────────────────────
    report = {
        "vocab_size": vocab_size,
        "actual_vocab_size": tok.get_vocab_size(),
        "min_frequency": min_frequency,
        "n_special_tokens": len(SPECIAL_TOKENS),
        "train_gb": max_gb,
        "train_files": [f.name for f in jsonl_files],
        "train_lines": n_lines,
        "train_minutes": train_s / 60,
        "data_dir": str(data_dir),
        "output_path": str(out_path),
    }
    with open(out_dir / "train_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Report: {out_dir / 'train_report.json'}")

    # cleanup corpus file (large, no longer needed)
    tmp_txt.unlink(missing_ok=True)
    return out_path


def _validate(tok) -> None:
    """Roundtrip tests across scripts and structured tokens."""
    samples = {
        "english":   "The quick brown fox jumps over the lazy dog.",
        "french":    "En France, le gouvernement a annoncé une nouvelle politique économique.",
        "chinese":   "你好世界，这是一个测试。",
        "russian":   "Привет мир, как дела сегодня?",
        "arabic":    "مرحبا بالعالم، كيف حالك اليوم؟",
        "hindi":     "नमस्ते दुनिया, आप कैसे हैं?",
        "japanese":  "こんにちは世界、お元気ですか？",
        "mixed":     "Hello 世界 Bonjour мир! <|user|> <OPCODE>BIND_ROLE</OPCODE> testing",
    }
    print("\n-- roundtrip validation --")
    all_ok = True
    for label, text in samples.items():
        enc = tok.encode(text)
        ids = enc.ids if hasattr(enc, "ids") else list(enc)
        back = tok.decode(ids, skip_special_tokens=False)
        ok = back == text
        if not ok:
            all_ok = False
            print(f"  FAIL [{label}]")
            print(f"    in:  {text!r}")
            print(f"    out: {back!r}")
            print(f"    ids: {ids[:20]}...")
        else:
            bpt = len(text.encode("utf-8")) / max(len(ids), 1)
            print(f"  OK   [{label:8s}]  {len(ids):3d} tokens  {bpt:.1f} bytes/tok")

    # special token atomicity
    special_test = "<|user|> <OPCODE>BIND_ROLE</OPCODE> <TASK:SCHEMA2RULE>"
    enc = tok.encode(special_test)
    ids = enc.ids if hasattr(enc, "ids") else list(enc)
    print(f"\n  special atomicity: {len(ids)} tokens for {special_test!r}")
    for sp in ["<|user|>", "<OPCODE>", "</OPCODE>", "BIND_ROLE", "<TASK:SCHEMA2RULE>"]:
        sp_enc = tok.encode(sp)
        sp_ids = sp_enc.ids if hasattr(sp_enc, "ids") else list(sp_enc)
        print(f"    {sp:30s} -> {sp_ids}  (len={len(sp_ids)})")

    if all_ok:
        print("\n✓ all roundtrips pass")
    else:
        print("\n✗ some roundtrips FAILED")


def main():
    p = argparse.ArgumentParser(description="Train Cubby multilingual BPE tokenizer")
    p.add_argument("--vocab-size", type=int, default=32768, help="target vocabulary size")
    p.add_argument("--max-gb", type=float, default=10.0, help="max GB of JSONL to use (default 10)")
    p.add_argument("--name", default="cubby_mbpe32k", help="output directory name")
    p.add_argument("--data-dir", default=str(DATA_ROOT), help="directory with *.jsonl training files")
    p.add_argument("--min-frequency", type=int, default=2, help="min merge frequency")
    p.add_argument("--max-lines", type=int, default=None, help="cap on training lines (for quick experiments)")
    args = p.parse_args()
    train_tokenizer(
        vocab_size=args.vocab_size,
        data_dir=Path(args.data_dir),
        max_gb=args.max_gb,
        name=args.name,
        min_frequency=args.min_frequency,
        max_lines=args.max_lines,
    )


if __name__ == "__main__":
    main()
