$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$schedulerScript = Join-Path $repoRoot 'scripts\run_scheduler.py'
$breakevenScript = Join-Path $repoRoot 'scripts\run_breakeven_monitor.py'

if (-not (Test-Path $schedulerScript)) {
    throw "run_scheduler.py not found: $schedulerScript"
}
if (-not (Test-Path $breakevenScript)) {
    throw "run_breakeven_monitor.py not found: $breakevenScript"
}

$pythonCandidates = @()
if ($env:GP_MATE_PYTHON) {
    $pythonCandidates += $env:GP_MATE_PYTHON
}
$pythonCandidates += (Join-Path $repoRoot '.venv\Scripts\python.exe')
$pythonCandidates += 'C:\Users\user\openHands-test\LLM_FxTrading\.venv\Scripts\python.exe'

$pythonExe = $null
foreach ($candidate in $pythonCandidates) {
    if ($candidate -and (Test-Path $candidate)) {
        $pythonExe = $candidate
        break
    }
}

if (-not $pythonExe) {
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        $pythonExe = $pythonCmd.Source
    }
}

if (-not $pythonExe) {
    throw 'python executable not found. Set GP_MATE_PYTHON or install python in PATH.'
}

function Test-PythonScriptRunning {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ScriptPath
    )

    $scriptName = [IO.Path]::GetFileName($ScriptPath)
    $escaped = [Regex]::Escape($scriptName)
    $proc = Get-CimInstance Win32_Process |
        Where-Object {
            $_.Name -match '^python(\.exe)?$' -and
            $_.CommandLine -match $escaped
        } |
        Select-Object -First 1

    return $null -ne $proc
}

function Start-PythonScriptIfNeeded {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ScriptPath,
        [Parameter(Mandatory = $true)]
        [string]$Label
    )

    if (Test-PythonScriptRunning -ScriptPath $ScriptPath) {
        Write-Host "$Label is already running."
        return
    }

    Start-Process -FilePath $pythonExe -ArgumentList $ScriptPath -WorkingDirectory $repoRoot -WindowStyle Minimized | Out-Null
    Write-Host "$Label started."
}

Start-PythonScriptIfNeeded -ScriptPath $schedulerScript -Label 'Main scheduler'
Start-PythonScriptIfNeeded -ScriptPath $breakevenScript -Label 'Breakeven monitor'

Write-Host 'GP-MATE schedulers launch completed.'
