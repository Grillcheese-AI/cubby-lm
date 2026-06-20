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
    vocab_size: int = 65536           # base BPE vocab (language region)
    tokenizer_kind: str = "bbpe65k"   # "bbpe65k" | "multilingual_bpe" | "byte"
    tokenizer_path: str = ""          # explicit path override (empty = use default)
    # structured/AST control tokens registered as atomic special tokens so
    # cubelang/agent DSL markers never fragment. In multilingual_bpe these live
    # at the tail of the vocab (IDs >= ast_start_id); the language head covers
    # [0, ast_start_id) and the AST head covers [ast_start_id, total_vocab).
    n_special_tokens: int = 0         # set from tokenizer.n_special_tokens at build time

    # ── output head: dual-head architecture (language + AST) ──────────
    # The single shared trunk projects to two separate output heads, gated by
    # a learned router. Language loss uses the language head; AST loss uses
    # the AST head. At generation time the router picks which head to decode from.
    enable_dual_head: bool = False
    router_d: int = 2                 # language vs AST (extendable to more routes)
    ast_head_d_ffn: int = 0           # 0 = no FFN in AST head (just RMSNorm + linear)
    tie_lang_embeddings: bool = True  # tie language output proj to embed_lang
    tie_ast_embeddings: bool = True   # tie AST output proj to embed_ast

    # ── sampled softmax (importance sampling, training only) ──────────
    # Replace full softmax/CE with sampled importance-sampling CE during training.
    # At inference, full softmax is used (cheap at 30k vocab; the GPU handles it).
    enable_sampled_softmax: bool = False
    n_samples: int = 1024             # number of negative samples per token
    sampler: str = "uniform"          # "uniform" | "log_uniform" | "model"

    # ── old / compat ───────────────────────────────────────────────────
    head_type: str = "linear"         # kept for config compat; "linear" | "vsa"
    tie_embeddings: bool = True       # kept for compat; see tie_lang_embeddings

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

    # ── VSA binding head (0.0.6) ─────────────────────────────────────────
    # "vsa" = frozen MAP-bipolar binding head (D=vsa_d), cosine readout —
    #         enables the downstream VSA reasoning substrate; off until vetted.
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
_R1 = dict(_R0, d_model=1024, n_layers=18, d_ffn=4096, seq_len=512,
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
    # toy configs for quick iteration
    "tiny": dict(_R1, vocab_size=10000, d_model=1024, n_layers=8, d_ffn=3072, seq_len=128),
    "tiny_mbpe": dict(
        _R1, vocab_size=32768, tokenizer_kind="multilingual_bpe",
        enable_dual_head=True, enable_sampled_softmax=True, n_samples=1024,
        d_model=1024, n_layers=8, d_ffn=3072, seq_len=128,
    ),
    # Production-shape mbpe substrate: the validated mbpe32k + dual-head +
    # sampled-softmax stack (PPL 4 in <1000 steps at tiny shape) scaled to the
    # v3.3 trunk (d=1024, L=18) WITH 0.0.2 chunked SWA attention. This is the
    # consolidation gate that must generate coherent prose before 0.0.3 (MoE).
    "mbpe_v33": dict(
        _R2,  # = _R1 + enable_attention=True (chunked SWA every 3rd)
        vocab_size=32768, tokenizer_kind="multilingual_bpe",
        enable_dual_head=True, enable_sampled_softmax=True, n_samples=1024,
        d_model=1024, n_layers=12, d_ffn=4096, seq_len=512,
    ),
    # Head-1 emission: SINGLE-HEAD mbpe (the dual-head opcode path is dead/insecure;
    # the trunk emits .cube SOURCE as text, the compiler gates it -- see
    # docs/TRUNK_VM_EMISSION_CONTRACT.md). Full softmax (honest loss; V=32k is cheap).
    # Attention on (programs have structure). Trained on the verified v4 source corpus.
    "mbpe_emit": dict(
        _R2, vocab_size=32768, tokenizer_kind="multilingual_bpe",
        enable_dual_head=False, enable_sampled_softmax=False,
        d_model=1024, n_layers=8, d_ffn=2048, seq_len=256,
    ),
}

DEFAULT_VERSION = "0.0.0"


def make_config(version: str = DEFAULT_VERSION, **overrides) -> SparseCubbyConfig:
    """Build a config for a semver rung (0.0.0 .. 0.0.9), with optional overrides."""
    if version not in VERSIONS:
        raise ValueError(f"unknown version {version!r}; have {list(VERSIONS)}")
    return SparseCubbyConfig(**{**VERSIONS[version], **overrides})
