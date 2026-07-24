# sa3.ps1 - PowerShell twin of the sa3 wrapper (and of sa3.bat).
# PowerShell users get clean Ctrl-C handling (no cmd.exe
# "Terminate batch job (Y/N)?" prompt, which is a .bat quirk).
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Py = Join-Path $ScriptDir ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    $Py = Join-Path $ScriptDir ".venv/bin/python"   # POSIX layout (pwsh on mac/linux)
}
if (-not (Test-Path $Py)) { $Py = "python" }
& $Py (Join-Path $ScriptDir "scripts/sa3_tflite.py") @args
exit $LASTEXITCODE
