# sa3_tflite bootstrap for Windows - Stable Audio 3 inference on CPU (LiteRT/TFLite).
#
# The PowerShell twin of bootstrap.sh. Hosted at:
#   https://raw.githubusercontent.com/Stability-AI/stable-audio-3/main/optimized/tflite/bootstrap.ps1
#
# Usage (default demo):
#   irm https://raw.githubusercontent.com/Stability-AI/stable-audio-3/main/optimized/tflite/bootstrap.ps1 | iex
#
# Custom args (iex cannot forward arguments - download first):
#   irm https://raw.githubusercontent.com/Stability-AI/stable-audio-3/main/optimized/tflite/bootstrap.ps1 -OutFile bootstrap.ps1
#   .\bootstrap.ps1 --prompt "Death Metal" --dit medium --decoder same-l
#
# What it does:
#   1. Fetches the project (git clone if git is installed, else a zip pull).
#   2. Runs install.bat (python -m venv + pip install; needs Python 3.10-3.13
#      on PATH - ai-edge-litert ships win_amd64 wheels for those versions).
#   3. Runs sa3.bat with your args (default: 30 s demo prompt + --play).
#      Weights auto-download from HuggingFace on first use.

$ErrorActionPreference = "Stop"

# Older Windows PowerShell 5.1 hosts may default to TLS < 1.2, which GitHub
# rejects. Harmless no-op on PowerShell 7+ / current Win10+.
try { [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12 } catch {}

$RepoOwner = "Stability-AI"
$RepoName  = "stable-audio-3"
$Branch    = "main"
$SubDir    = "optimized\tflite"
$LocalDir  = "sa3_tflite"

function Step($msg) { Write-Host "`n-> $msg" -ForegroundColor Cyan }
function Ok($msg)   { Write-Host "  OK $msg" -ForegroundColor Green }

# --- 1. fetch the project -------------------------------------------------
if (Get-Command git -ErrorAction SilentlyContinue) {
    $WorkDir = Join-Path $RepoName $SubDir
    if (Test-Path (Join-Path $RepoName ".git")) {
        Step "Reusing existing .\$RepoName (git pull --ff-only)"
        git -C $RepoName pull --ff-only
    } elseif (Test-Path $RepoName) {
        throw ".\$RepoName exists but isn't a git repo - remove or rename it."
    } else {
        Step "git clone https://github.com/$RepoOwner/$RepoName -> .\$RepoName"
        git clone --depth=1 "https://github.com/$RepoOwner/$RepoName" $RepoName
    }
    if (-not (Test-Path $WorkDir)) { throw "Expected '$SubDir' inside the repo but didn't find it." }
} else {
    $WorkDir = $LocalDir
    if (Test-Path (Join-Path $LocalDir "install.bat")) {
        Step "Reusing existing $LocalDir\ (delete it to re-download)"
    } elseif (Test-Path $LocalDir) {
        throw ".\$LocalDir exists but doesn't look like a sa3_tflite checkout - remove or rename it."
    } else {
        Step "git not installed - downloading $RepoOwner/$RepoName ($Branch) zip -> .\$LocalDir"
        $zip     = Join-Path $env:TEMP "sa3_repo.zip"
        $extract = Join-Path $env:TEMP "sa3_extract"
        Invoke-WebRequest "https://github.com/$RepoOwner/$RepoName/archive/refs/heads/$Branch.zip" -OutFile $zip
        if (Test-Path $extract) { Remove-Item $extract -Recurse -Force }
        Expand-Archive $zip -DestinationPath $extract -Force
        Move-Item (Join-Path $extract "$RepoName-$Branch\$SubDir") $LocalDir
        Copy-Item `
            (Join-Path $extract "$RepoName-$Branch\stable_audio_3\audio_output.py") `
            (Join-Path $LocalDir "models\defs\audio_output.py")
        Remove-Item $zip -Force
        Remove-Item $extract -Recurse -Force
        Ok "extracted to .\$LocalDir"
    }
}

Set-Location $WorkDir
Ok "ready at $(Get-Location)"

# --- 2. install -----------------------------------------------------------
Step "Running install.bat"
cmd /c install.bat
if ($LASTEXITCODE -ne 0) { throw "install.bat failed (exit $LASTEXITCODE)." }

# --- 3. inference ---------------------------------------------------------
if ($args.Count -gt 0) {
    Step "Running sa3.bat $($args -join ' ')"
    cmd /c sa3.bat @args
} else {
    Step 'Running demo: sa3.bat --prompt "Impending tribal, epic orchestral buildup" --dit sm-music --decoder same-s --seconds 30 --play'
    Write-Host '   (for custom args, download bootstrap.ps1 first and run: .\bootstrap.ps1 --prompt "..." ...)' -ForegroundColor DarkGray
    cmd /c sa3.bat --prompt "Impending tribal, epic orchestral buildup" --dit sm-music --decoder same-s --seconds 30 --play
}

Write-Host ""
Write-Host "You are set up in $(Get-Location)" -ForegroundColor Green
Write-Host "  run .\sa3.bat --help for options" -ForegroundColor DarkGray
