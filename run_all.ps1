param(
    [switch]$SkipInstall,
    [switch]$DryRun,
    [switch]$RestartBackend
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

Set-Location $PSScriptRoot

function Get-PythonExe {
    $venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        return $venvPython
    }

    $pyCmd = Get-Command py -ErrorAction SilentlyContinue
    if (-not $pyCmd) {
        throw "Python launcher 'py' not found. Install Python 3.12+ and try again."
    }

    Write-Host "Creating virtualenv at .venv ..." -ForegroundColor Cyan
    & $pyCmd.Source -3 -m venv ".venv"
    return $venvPython
}

function Get-NpmCommand {
    $npmCmd = Get-Command npm -ErrorAction SilentlyContinue
    if ($npmCmd) {
        return $npmCmd.Source
    }

    $programFilesNpm = Join-Path ${env:ProgramFiles} "nodejs\npm.cmd"
    if (Test-Path $programFilesNpm) {
        return $programFilesNpm
    }

    throw "npm not found. Install Node.js LTS and reopen PowerShell."
}

function Start-Window([string]$Title, [string]$Command) {
    if ($DryRun) {
        Write-Host "[dry-run] $Title :: $Command" -ForegroundColor Yellow
        return
    }
    Start-Process powershell -ArgumentList @(
        "-NoExit",
        "-ExecutionPolicy", "Bypass",
        "-Command", "`$Host.UI.RawUI.WindowTitle = '$Title'; $Command"
    ) | Out-Null
}

function Get-ListeningProcessId([int]$Port) {
    $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($null -eq $conn) {
        return $null
    }
    return [int]$conn.OwningProcess
}

$python = Get-PythonExe
$npm = Get-NpmCommand
$nodeDir = Split-Path $npm -Parent

if (-not $SkipInstall) {
    Write-Host "Installing backend dependencies ..." -ForegroundColor Cyan
    & $python -m pip install --upgrade pip
    & $python -m pip install -r "requirements.txt"

    Write-Host "Installing frontend dependencies ..." -ForegroundColor Cyan
    & $npm --prefix "frontend" install
}

$daphne = Join-Path $PSScriptRoot ".venv\Scripts\daphne.exe"
if (-not (Test-Path $daphne)) {
    throw "Daphne not found in .venv. Run without -SkipInstall once to install dependencies."
}

$backendPortPid = Get-ListeningProcessId -Port 8000
if ($backendPortPid) {
    if ($RestartBackend) {
        if ($DryRun) {
            Write-Host "Port 8000 already in use by PID $backendPortPid; would stop it (dry-run)." -ForegroundColor Yellow
        } else {
            Write-Host "Port 8000 already in use by PID $backendPortPid; stopping it ..." -ForegroundColor Yellow
            Stop-Process -Id $backendPortPid -Force
            Start-Sleep -Milliseconds 300
        }
    } else {
        Write-Host "Port 8000 already in use by PID $backendPortPid. Backend launch will be skipped." -ForegroundColor Yellow
        Write-Host "Use -RestartBackend to stop that process automatically." -ForegroundColor Yellow
    }
}

$backendCommand = "Set-Location '$PSScriptRoot\backend'; & '$daphne' -b 0.0.0.0 -p 8000 privateedge.asgi:application"
$frontendCommand = "`$env:Path = '$nodeDir;' + `$env:Path; Set-Location '$PSScriptRoot\frontend'; & '$npm' run dev"

if (-not $backendPortPid -or $RestartBackend) {
    Start-Window -Title "PrivateEdge Backend" -Command $backendCommand
}
Start-Window -Title "PrivateEdge Frontend" -Command $frontendCommand

Write-Host ""
Write-Host "PrivateEdge launch started." -ForegroundColor Green
Write-Host "Backend:  http://127.0.0.1:8000/api/health/"
Write-Host "Frontend: http://localhost:5173/"
Write-Host ""
Write-Host "Usage:"
Write-Host "  .\run_all.ps1           # install deps + start both"
Write-Host "  .\run_all.ps1 -SkipInstall"
Write-Host "  .\run_all.ps1 -DryRun"
Write-Host "  .\run_all.ps1 -RestartBackend"
