"""Sparse Cubby configuration.

Single source of truth for the architecture, ported to run on the grilly Vulkan
backend (no torch in the trunk path). Faithful to the working v3.3 run
(`cubemind/model/cubby/colab_v3_3_test.ipynb`) and the architecture spec
(`docs/why_sparse_cubby.md`).

Discipline (from the v4 post-mortem): every component above the substrate is
flag-gated and OFF at v0. A version preset flips exactly the switches its rung
adds; an unbuilt component raises rather than silently no-op'ing. Build order is
forced by the dependency DAG — the MinGRU trunk must generate before anything
rides on it.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass
class SparseCubbyConfig:
    # ── tokenizer / vocab ────────────────────────────────────────────────
    vocab_size: int = 65536           # BBPE-65k v3 (grillcheese_bbpe65k_v3)
    # structured/AST control tokens registered as atomic special tokens on the
    # BBPE so cubelang/agent DSL markers never fragment (see tokenizer/).
    n_special_tokens: int = 0

    # ── trunk dims (v3.3 production shape) ───────────────────────────────
    d_model: int = 1024
    n_layers: int = 18
    d_ffn: int = 4096                 # SwiGLU inner dim
    ffn_type: str = "swiglu"          # "swiglu" | "ternary_swiglu" (BitNet b1.58 QAT)
    seq_len: int = 4096
    rmsnorm_eps: float = 1e-6
    embed_init_std: float = 0.02
    decay_bias_init: float = 1.0      # proj_d bias -> sigmoid ~0.73 retention t=0
    enable_residual_scale: bool = True  # per-residual alpha (helps at L>=18)
    dtype: str = "fp32"               # grilly trunk: fp32 | bf16

    # ── output head ──────────────────────────────────────────────────────
    # "linear" = weight-tied linear (the v4 fix, default for the word vocab).
    # "vsa"    = frozen MAP-bipolar binding head (D=vsa_d), cosine readout —
    #            enables the downstream VSA reasoning substrate; off until vetted.
    head_type: str = "linear"
    tie_embeddings: bool = True
    vsa_d: int = 10240
    vsa_seed: int = 0xC0DEB00C
    vsa_temperature: str = "learned"     # "learned" | "<float>" (v4 froze it)
    vsa_learned_codebook: bool = False   # spec default: frozen codebook buffer

    # ── unlimited context: chunked sliding-window attention (0.0.2) ──────
    enable_attention: bool = False
    attn_heads: int = 16
    attn_window: int = 512               # chunked SDPA, O(L) memory
    attn_every_n: int = 3

    # ── sparse MoE-MinGRU (0.0.3) ───────────────────────────────────────────
    enable_moe: bool = False
    moe_n_experts: int = 4
    moe_top_k: int = 2
    moe_shared_experts: int = 3
    moe_aux_weight: float = 0.01
    moe_bias_rebal_lr: float = 0.05      # DeepSeek-V3 aux-loss-free bias update
    moe_decay_stagger: bool = True
    moe_max_experts: int = 4             # Hebbian growth ceiling (0.0.4)

    # ── Hebbian growth: novelty -> spawn expert (0.0.4) ─────────────────────
    enable_hebbian_growth: bool = False
    hebbian_sig_dim: int = 256           # stable-prefix rows used as mem signature
    hebbian_residual_threshold: float = 0.5

    # ── compressive segment memory (0.0.5) — 128k effective context ──────
    enable_segment_memory: bool = False
    mem_bucket_cap: int = 256

    # ── afferent input gate: SNN perceptron triage (the input layer) ─────
    # The front door. A spiking perceptron scores salience/threat on every input,
    # then dual-routes (LeDoux low-road / high-road):
    #   high stress/threat/urgency -> CNS reflex path (live_brain SNN: amygdala +
    #     neurochemistry; an interrupt that pre-empts the cortical router);
    #   otherwise -> cortical router that classifies {modality,tone,intent,domain}
    #     and dispatches to the matching WorldManager specialist, spawning one
    #     (Hebbian expert-on-demand) when no specialist owns that regime.
    # Sits above the trunk; bypassed for raw-token substrate tests (0.0.0).
    enable_input_gate: bool = False
    gate_snn_hidden: int = 256             # spiking perceptron hidden width
    gate_stress_threshold: float = 0.7     # threat score -> CNS reflex path
    gate_route_axes: tuple = ("modality", "tone", "intent", "domain")
    gate_spawn_on_missing_specialist: bool = True

    # ── post-training attachments (frozen trunk) ─────────────────────────
    # MTP is DECODE-TIME ONLY (self-speculative head attached post-training);
    # never enabled during pretraining — on an untrained trunk it hurts.
    enable_mtp: bool = False             # 0.0.7: multi-token speculative decode
    mtp_k: int = 2
    enable_cubelang_head: bool = False   # 0.0.8: closed 15-op OpcodeStmt grammar
    enable_adapter_bank: bool = False    # 0.0.9: MindForge LoRA-MoE over frozen feats

    # ── observability (cross-cutting, EVERY rung) ────────────────────────
    # Hard requirement: every step, even the smallest, is auditable AND
    # visualizable — real-time per example (which neuron fires lights up) and
    # captured in tests for debugging. The trunk forward writes to `cubby.trace`
    # from line 1. OFF = zero overhead (production); higher levels add per-unit
    # intensities/spikes (visual) and full tensors (parity debugging).
    enable_trace: bool = False
    trace_level: str = "audit"           # off | audit | visual | full

    def __post_init__(self) -> None:
        if self.trace_level not in ("off", "audit", "visual", "full"):
            raise ValueError(f"trace_level must be off|audit|visual|full, got {self.trace_level!r}")
        if self.ffn_type not in ("swiglu", "ternary_swiglu"):
            raise ValueError(f"ffn_type must be 'swiglu'|'ternary_swiglu', got {self.ffn_type!r}")
        if self.head_type not in ("linear", "vsa"):
            raise ValueError(f"head_type must be 'linear'|'vsa', got {self.head_type!r}")
        if self.dtype not in ("fp32", "bf16"):
            raise ValueError(f"dtype must be 'fp32'|'bf16', got {self.dtype!r}")

    @property
    def total_vocab(self) -> int:
        return self.vocab_size + self.n_special_tokens


# ── version registry (semantic versioning) ──────────────────────────────
# Each rung is a semver patch on the ladder; 0.0.0 is the substrate. A preset is
# the smallest delta that adds one validated component, plus the gate that must
# pass before the next. 0.0.0 is a SMALL trunk (the exonerated 5.6M MinGRU
# baseline) so the substrate is proven to generate on grilly before scaling;
# 0.0.1 scales to the v3.3 production shape. A larger milestone bumps the minor
# (0.1.0), a release the major (1.0.0).

_R0 = dict(  # 0.0.0 substrate: grilly MinGRU + tied-linear + SwiGLU
    vocab_size=65536, d_model=256, n_layers=6, d_ffn=512, seq_len=512,
    enable_residual_scale=False,
)
_R1 = dict(_R0, d_model=1024, n_layers=18, d_ffn=4096, seq_len=4096,
           enable_residual_scale=True)                  # 0.0.1 scale to v3.3 shape
_R2 = dict(_R1, enable_attention=True)                  # 0.0.2 +chunked SWA (context)
_R3 = dict(_R2, enable_moe=True)                        # 0.0.3 +sparse MoE-MinGRU
_R4 = dict(_R3, enable_hebbian_growth=True)             # 0.0.4 +Hebbian growth
_R5 = dict(_R4, enable_segment_memory=True)             # 0.0.5 +compressive memory
_R6 = dict(_R5, head_type="vsa")                        # 0.0.6 +VSA binding head
_R7 = dict(_R6, enable_mtp=True)                        # 0.0.7 +MTP (DECODE-TIME ONLY)
_R8 = dict(_R7, enable_cubelang_head=True)              # 0.0.8 +CubeLang/VM bridge
_R9 = dict(_R8, enable_adapter_bank=True)               # 0.0.9 +no-retrain adapters

VERSIONS: dict[str, dict] = {
    "0.0.0": _R0, "0.0.1": _R1, "0.0.2": _R2, "0.0.3": _R3, "0.0.4": _R4,
    "0.0.5": _R5, "0.0.6": _R6, "0.0.7": _R7, "0.0.8": _R8, "0.0.9": _R9,
}

DEFAULT_VERSION = "0.0.0"


def make_config(version: str = DEFAULT_VERSION, **overrides) -> SparseCubbyConfig:
    """Build a config for a semver rung (0.0.0 .. 0.0.9), with optional overrides."""
    if version not in VERSIONS:
        raise ValueError(f"unknown version {version!r}; have {list(VERSIONS)}")
    return SparseCubbyConfig(**{**VERSIONS[version], **overrides})
