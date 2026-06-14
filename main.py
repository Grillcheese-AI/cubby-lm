"""Cubby — command-line entry point (the only module in the repo root).

    python main.py info  --version 0.0.0
    python main.py smoke --version 0.0.0
    python main.py parity --version 0.0.0    # grilly trunk vs torch reference
    python main.py train --version 0.0.0 --steps 4000
    python main.py gen   --version 0.0.0 --prompt "Once upon a time, "

Everything else lives under the `cubby/` package. Subcommands dispatch into it;
unbuilt ones raise with a pointer to cubby/ROADMAP.md.
"""
from __future__ import annotations

import argparse

from cubby.config import make_config, VERSIONS, DEFAULT_VERSION


def cmd_info(args: argparse.Namespace) -> None:
    cfg = make_config(args.version)
    print(f"Cubby {args.version}  (head={cfg.head_type}, dtype={cfg.dtype})")
    print(f"  vocab={cfg.total_vocab}  d_model={cfg.d_model}  layers={cfg.n_layers}"
          f"  d_ffn={cfg.d_ffn}  seq_len={cfg.seq_len}")
    on = [k for k in ("enable_input_gate", "enable_attention", "enable_moe",
                      "enable_hebbian_growth", "enable_segment_memory", "enable_mtp",
                      "enable_cubelang_head", "enable_adapter_bank") if getattr(cfg, k)]
    print(f"  components on: {', '.join(c.replace('enable_', '') for c in on) or 'none (substrate)'}")


def _todo(name: str):
    raise SystemExit(
        f"`{name}` is not built yet for this version. See cubby/ROADMAP.md for the "
        f"build order; 0.0.0 (grilly MinGRU trunk) is the current target.")


def cmd_smoke(args):  _todo("smoke")
def cmd_parity(args): _todo("parity")
def cmd_train(args):  _todo("train")
def cmd_gen(args):    _todo("gen")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cubby", description="Sparse Cubby (grilly).")
    sub = p.add_subparsers(dest="command", required=True)

    def add(name, fn, help_):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--version", choices=list(VERSIONS), default=DEFAULT_VERSION)
        sp.set_defaults(func=fn)
        return sp

    add("info", cmd_info, "print the resolved config for a version")
    add("smoke", cmd_smoke, "forward/backward/generate plumbing check")
    add("parity", cmd_parity, "grilly trunk vs torch reference (max_abs_diff)")
    tr = add("train", cmd_train, "train the trunk on grilly")
    tr.add_argument("--steps", type=int, default=4000)
    tr.add_argument("--data", default="tinystory_50k.json")
    gn = add("gen", cmd_gen, "autoregressive generation")
    gn.add_argument("--prompt", default="Once upon a time, ")
    gn.add_argument("--max-new-tokens", type=int, default=200)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
