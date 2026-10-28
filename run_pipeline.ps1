<#
.SYNOPSIS  Windows launcher for run_pipeline.py (project virtual environment, one run at a time).
.EXAMPLE   .\run_pipeline.ps1 preflight --config config\fixture_tueg.json
.EXAMPLE   .\run_pipeline.ps1 run --config config\fixture_tueg.json --confirm
#>
$ErrorActionPreference = 'Stop'
$Repo = $PSScriptRoot
$Python = Join-Path $Repo '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Missing virtual environment. Create it with: py -3.12 -m venv .venv; .\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt"
}
$Mutex = [System.Threading.Mutex]::new($false, 'Global\FSMS_PIPELINE')
$Held = $false
try {
    try { $Held = $Mutex.WaitOne(0) } catch [System.Threading.AbandonedMutexException] { $Held = $true }
    if (-not $Held) { throw 'Another FS_MS pipeline process is already running.' }
    Set-Location -LiteralPath $Repo
    # Native stderr must pass through (PowerShell 5.1 turns it into a terminating error under 'Stop');
    # the Python exit code is the contract: 0 ok, 1 failed check, 2 refused/failed.
    $ErrorActionPreference = 'Continue'
    & $Python -B (Join-Path $Repo 'run_pipeline.py') @args
    $Code = $LASTEXITCODE
    exit $Code
} finally {
    if ($Held) { $Mutex.ReleaseMutex() }
    $Mutex.Dispose()
}
