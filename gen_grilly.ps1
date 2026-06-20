<#
  gen_grilly.ps1 -- free-run a checkpoint to spot-check coherence + the Grilly
  identity. Run when the trainer is IDLE (one Vulkan context / 12 GB at a time).

  Usage:
    .\gen_grilly.ps1                       # Grilly chat identity probe (default)
    .\gen_grilly.ps1 -Prompt "In 1969, "   # general prose continuation
    .\gen_grilly.ps1 -Temperature 0        # greedy/argmax
#>
param(
    [string]$Prompt = "<|system|>`nYou are Grilly, a helpful assistant.`n<|user|>`nWho are you?`n<|assistant|>`n",
    [double]$Temperature = 0.7,
    [int]$MaxNewTokens = 120,
    [string]$Ckpt = "ckpt_mbpe_grilly.grl"
)

$ErrorActionPreference = "Stop"
$root = "C:\Users\grill\Documents\GitHub\cubby-lm"
$py   = Join-Path $root ".venv\Scripts\python.exe"
Set-Location $root

$genArgs = @(
    "main.py", "gen",
    "--version",        "mbpe_v33",
    "--tokenizer",      "mbpe32k",
    "--ckpt",           $Ckpt,
    "--prompt",         $Prompt,
    "--temperature",    "$Temperature",
    "--max-new-tokens", "$MaxNewTokens"
)

& $py @genArgs
