<#
  run_mbpe_grilly.ps1 -- resident training run for the mbpe_v33 stack
  (mbpe32k + dual-head + sampled-softmax + chunked SWA), on the
  fineweb_edu 90% + Grilly identity 10% mix.

  Usage:
    .\run_mbpe_grilly.ps1                 # fresh run (deletes old checkpoint)
    .\run_mbpe_grilly.ps1 -Resume         # continue from ckpt_mbpe_grilly.grl
    .\run_mbpe_grilly.ps1 -Steps 2000 -Batch 4

  Notes:
    - B=4 / S=128 (N=512) is the 12 GB-VRAM sweet spot. B=8 falls into the
      VRAM-bound path (~0.7 it/s); keep batch at 4.
    - Each sample point prints a general continuation AND a Grilly identity probe.
#>
param(
    [switch]$Resume,
    [int]$Steps = 10000,
    [int]$Batch = 8,
    [int]$SeqLen = 64,        # N=512 sweet spot; it/s is ~flat vs S, so 128 >> 64 in tokens/sec
    [int]$MaxTokens = 600000,  # smaller stream => more epochs per step
    [string]$Lr = "6e-4"       # ceiling per the grad-norm diagnosis; safe with clip+warmup
)

$ErrorActionPreference = "Stop"
$root = "C:\Users\grill\Documents\GitHub\cubby-lm"
$py   = Join-Path $root ".venv\Scripts\python.exe"
$ckpt = "ckpt_mbpe_grilly.grl"

$data = "D:\grillcheese_training_data\v3_3\fineweb_edu.2m.jsonl:0.9,D:\grillcheese_training_data\identity_corpus_grilly.txt:0.1"

Set-Location $root

# Fresh start unless -Resume: remove the old checkpoint so weights/optimizer reset.
$ckptPath = Join-Path $root $ckpt
if (-not $Resume -and (Test-Path $ckptPath)) {
    Write-Host "[run] fresh start -- removing $ckpt" -ForegroundColor Yellow
    Remove-Item $ckptPath -Force
}

$trainArgs = @(
    "main.py", "train",
    "--version",   "mbpe_v33",
    "--tokenizer", "mbpe32k",
    "--data",      $data,
    "--ckpt",      $ckpt,
    "--steps",      "$Steps",
    "--batch",      "$Batch",
    "--seqlen",     "$SeqLen",
    "--max-tokens", "$MaxTokens",
    "--lr",         $Lr,
    "--warmup",    "50",
    "--clip",      "1.0"
)

Write-Host "[run] $py $($trainArgs -join ' ')" -ForegroundColor Cyan
& $py @trainArgs
