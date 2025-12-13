# Activate Virtual Environment - PowerShell Script
# Usage: .\Dependencies\activate_env.ps1

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "CS506 Project - Virtual Environment Setup" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Get project root (parent of Dependencies folder)
$ProjectRoot = Split-Path -Parent $PSScriptRoot

# Check for .venv_wsl (primary environment)
$VenvPath = Join-Path $ProjectRoot ".venv_wsl"

if (Test-Path $VenvPath) {
    Write-Host "✓ Found virtual environment: .venv_wsl" -ForegroundColor Green
    
    # Check if running in WSL
    if ($env:WSL_DISTRO_NAME) {
        Write-Host "Running in WSL - use bash activation script instead:" -ForegroundColor Yellow
        Write-Host "  bash Dependencies/activate_env.sh" -ForegroundColor Yellow
        exit 1
    }
    
    # Activate for PowerShell
    $ActivateScript = Join-Path $VenvPath "Scripts\Activate.ps1"
    
    if (Test-Path $ActivateScript) {
        Write-Host "Activating environment..." -ForegroundColor Cyan
        & $ActivateScript
        Write-Host ""
        Write-Host "✓ Virtual environment activated!" -ForegroundColor Green
        Write-Host "Python location: $(Get-Command python).Path" -ForegroundColor Gray
        Write-Host ""
        Write-Host "To deactivate, run: deactivate" -ForegroundColor Yellow
    } else {
        Write-Host "✗ Activation script not found at: $ActivateScript" -ForegroundColor Red
        Write-Host "Run setup first: python -m venv .venv_wsl" -ForegroundColor Yellow
        exit 1
    }
} else {
    Write-Host "✗ Virtual environment not found at: $VenvPath" -ForegroundColor Red
    Write-Host ""
    Write-Host "Create the environment first:" -ForegroundColor Yellow
    Write-Host "  1. Open WSL terminal" -ForegroundColor Gray
    Write-Host "  2. cd /mnt/c/Users/Matt/Desktop/CS506/CS506_Project" -ForegroundColor Gray
    Write-Host "  3. python3 -m venv .venv_wsl" -ForegroundColor Gray
    Write-Host "  4. source .venv_wsl/bin/activate" -ForegroundColor Gray
    Write-Host "  5. pip install -r Dependencies/requirements.txt" -ForegroundColor Gray
    exit 1
}
