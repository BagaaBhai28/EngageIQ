param(
    [switch]$SetupOnly,
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

function Run-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed (exit $LASTEXITCODE): $Executable $($Arguments -join ' ')" }
}

function Find-Python313 {
    $candidates = @(
        (Join-Path $PSScriptRoot '.venv\Scripts\python.exe'),
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python313\python.exe'),
        'C:\Python313\python.exe'
    )
    foreach ($command in @('py', 'python', 'python3')) {
        $found = Get-Command $command -ErrorAction SilentlyContinue
        if ($found) {
            try {
                $argsForPython = @('-c', 'import sys; print(sys.executable)')
                if ($command -eq 'py') { $argsForPython = @('-3.13') + $argsForPython }
                $resolved = & $found.Source @argsForPython 2>$null
                if ($LASTEXITCODE -eq 0) { $candidates += [string]$resolved }
            } catch { }
        }
    }
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            try {
                & $candidate -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 13) and sys.maxsize > 2**32 else 1)' 2>$null
                if ($LASTEXITCODE -eq 0) { return $candidate }
            } catch { }
        }
    }
    return $null
}

try {
    Write-Host "`nEngageIQ | one-command local setup" -ForegroundColor Cyan
    Write-Host '[1/5] Checking Python 3.13 (64-bit)...'
    $taskPython = Find-Python313
    if (-not $taskPython) {
        if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
            throw 'Python 3.13 is missing and winget is unavailable. Install Python 3.13 (64-bit) from python.org, then rerun this script. Windows App Installer provides winget.'
        }
        Write-Host 'Installing Python for the current Windows user...'
        Run-Checked -Executable 'winget' -Arguments @('install', '--id', 'Python.Python.3.13', '--exact', '--source', 'winget', '--scope', 'user', '--architecture', 'x64', '--silent', '--accept-package-agreements', '--accept-source-agreements', '--disable-interactivity')
        $taskPython = Find-Python313
        if (-not $taskPython) { throw 'Python installation finished but Python 3.13 could not be found. Reopen the terminal and rerun this script.' }
    }

    Write-Host '[2/5] Preparing the project virtual environment...'
    $venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $venvPython)) {
        Run-Checked -Executable $taskPython -Arguments @('-m', 'venv', (Join-Path $PSScriptRoot '.venv'))
    }
    Run-Checked -Executable $venvPython -Arguments @('-c', 'import sys; assert sys.version_info[:2] == (3,13), "Existing .venv must use Python 3.13. Rename it and rerun setup."')

    Write-Host '[3/5] Installing/verifying all pinned dependencies...'
    # Rechecking pip on every run also repairs partially installed environments.
    Run-Checked -Executable $venvPython -Arguments @('-m', 'ensurepip', '--upgrade')
    Run-Checked -Executable $venvPython -Arguments @('-m', 'pip', 'install', '--disable-pip-version-check', '-r', (Join-Path $PSScriptRoot 'requirements-lock.txt'))
    Run-Checked -Executable $venvPython -Arguments @('-m', 'pip', 'check')

    Write-Host '[4/5] Verifying datasets, trained models and local storage...'
    Run-Checked -Executable $venvPython -Arguments @('scripts\preflight.py')
    if ($SetupOnly) {
        Write-Host "`nSetup complete. Run this script without -SetupOnly to start the site." -ForegroundColor Green
        exit 0
    }

    Write-Host '[5/5] Starting EngageIQ...'
    $serverArgs = @('scripts\run_server.py')
    if ($NoBrowser) { $serverArgs += '--no-browser' }
    Run-Checked -Executable $venvPython -Arguments $serverArgs
} catch {
    Write-Host "`nSETUP STOPPED: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host 'Correct the reported issue and rerun. Existing data is preserved.'
    exit 1
}
