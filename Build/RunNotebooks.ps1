# Run all 7 notebooks sequentially in WSL
# This script will not be interrupted by other terminal commands

$ErrorActionPreference = "Continue"

Write-Host "Starting notebook execution in WSL..." -ForegroundColor Cyan
Write-Host "This will take 30-60 minutes. Do not close this window." -ForegroundColor Yellow
Write-Host ""

# Detect project root (parent of Build directory)
$BuildFolder = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $BuildFolder
Set-Location $ProjectRoot

# Convert Windows path to WSL path
$WslPath = $ProjectRoot -replace '\\', '/' -replace '^([A-Z]):', '/mnt/$1' -replace '([A-Z])', {$_.Value.ToLower()}

# Run the makefile in WSL (auto-activates venv and checks deps)
wsl bash -c "cd '$WslPath' && make -f Build/Makefile.wsl run"

Write-Host ""
Write-Host "Notebook execution complete!" -ForegroundColor Green

# Extract and log results
Write-Host "Extracting model results..." -ForegroundColor Cyan
wsl bash -c "cd '$WslPath' && source .venv_wsl/bin/activate && python3 Build/extract_results.py"

Write-Host ""
Write-Host "Results logged to model_results.log" -ForegroundColor Green
Write-Host "Check the notebooks and log file for detailed results." -ForegroundColor Cyan
