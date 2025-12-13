# Launcher script for export_svr_animations.py (PowerShell)
# Note: This project uses WSL environment - run via WSL instead

param(
    [switch]$ForceWSL = $true
)

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "SVR Animation Generator - Launcher" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $ProjectRoot ".venv_wsl"

# Check if .venv_wsl exists
if (-not (Test-Path $VenvPath)) {
    Write-Host "ERROR: Virtual environment not found at: $VenvPath" -ForegroundColor Red
    Write-Host ""
    Write-Host "This project uses WSL environment. Create it via:" -ForegroundColor Yellow
    Write-Host "  wsl bash" -ForegroundColor Gray
    Write-Host "  cd /mnt/c/Users/Matt/Desktop/CS506/CS506_Project" -ForegroundColor Gray
    Write-Host "  python3 -m venv .venv_wsl" -ForegroundColor Gray
    Write-Host "  source .venv_wsl/bin/activate" -ForegroundColor Gray
    Write-Host "  pip install -r Dependencies/requirements.txt" -ForegroundColor Gray
    exit 1
}

# Check if WSL is available
try {
    wsl --version | Out-Null
    $WSLAvailable = $true
} catch {
    $WSLAvailable = $false
}

# Run via WSL (recommended)
if ($WSLAvailable -and $ForceWSL) {
    Write-Host "Running via WSL (recommended)..." -ForegroundColor Green
    Write-Host ""
    
    # Convert Windows path to WSL path
    $WSLPath = $ProjectRoot -replace '\\','/' -replace 'C:','/mnt/c'
    
    $WSLCommand = "cd '$WSLPath' && source .venv_wsl/bin/activate && python Build/export_svr_animations.py"
    
    wsl bash -c $WSLCommand
    
} else {
    Write-Host "WARNING: .venv_wsl is a Linux environment" -ForegroundColor Yellow
    Write-Host "Use WSL to run this script properly." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Alternative: Use the bash launcher via WSL:" -ForegroundColor Cyan
    Write-Host "  wsl bash Build/run_export_animations.sh" -ForegroundColor Gray
    exit 1
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Animation generation complete!" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
