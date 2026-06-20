<#
  finetune_identity.ps1 -- Phase 2: identity SFT over a coherent trunk.

  Run this AFTER the Phase-1 general run (run_mbpe_grilly.ps1) has reached good
  coherence. It copies the Phase-1 checkpoint to a new file (so Phase 1 is
  preserved), then fine-tunes on a HIGH identity ratio (50/50 with fineweb to
  keep fluency -- pure identity risks catastrophic forgetting of general
  language), at a lower LR.

  Usage:
    .\finetune_identity.ps1                 # 50/50, lr 2e-4, 2000 steps
    .\finetune_identity.ps1 -IdentityFrac 0.4 -Steps 3000

  After it finishes, check the persona:
    .\gen_grilly.ps1 -Temperature 0 -Ckpt ckpt_mbpe_grilly_id.grl
#>
param(
    [int]$Steps = 2000,
    [int]$Batch = 4,
    [int]$SeqLen = 128,            # longer context helps the chat turn structure
    [int]$MaxTokens = 400000,
    [string]$Lr = "2e-4",         # gentler than pretrain, to adapt not overwrite
    [double]$IdentityFrac = 0.5
)

$ErrorActionPreference = "Stop"
$root = "C:\Users\grill\Documents\GitHub\cubby-lm"
$py   = Join-Path $root ".venv\Scripts\python.exe"
$base = Join-Path $root "ckpt_mbpe_grilly.grl"        # Phase-1 coherent trunk
$ph2  = "ckpt_mbpe_grilly_id.grl"                     # Phase-2 output (preserves Phase 1)

Set-Location $root
if (-not (Test-Path $base)) { throw "no Phase-1 checkpoint at $base -- run run_mbpe_grilly.ps1 first" }

# Seed Phase 2 from the Phase-1 weights (resume continues from this copy).
Copy-Item $base (Join-Path $root $ph2) -Force
Write-Host "[ft] seeded $ph2 from $base" -ForegroundColor Yellow

$fineFrac = [math]::Round(1.0 - $IdentityFrac, 2)
$data = "D:\grillcheese_training_data\v3_3\fineweb_edu.2m.jsonl:$fineFrac,D:\grillcheese_training_data\identity_corpus_grilly.txt:$IdentityFrac"

$args = @(
    "main.py", "train",
    "--version",    "mbpe_v33",
    "--tokenizer",  "mbpe32k",
    "--data",       $data,
    "--ckpt",       $ph2,
    "--steps",      "$Steps",
    "--batch",      "$Batch",
    "--seqlen",     "$SeqLen",
    "--max-tokens", "$MaxTokens",
    "--lr",         $Lr,
    "--warmup",     "30",
    "--clip",       "1.0"
)
Write-Host "[ft] identity=$IdentityFrac fineweb=$fineFrac  lr=$Lr  steps=$Steps" -ForegroundColor Cyan
# resume is on by default in the trainer, so loading $ph2 continues from Phase-1 weights.
& $py @args
