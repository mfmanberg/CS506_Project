# Run all 7 notebooks sequentially in WSL
# This script will not be interrupted by other terminal commands

$ErrorActionPreference = "Continue"

Write-Host "Starting notebook execution in WSL..." -ForegroundColor Cyan
Write-Host "This will take 30-60 minutes. Do not close this window." -ForegroundColor Yellow
Write-Host ""

# Detect project root (directory containing this script)
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

# Convert Windows path to WSL path
$WslPath = $ProjectRoot -replace '\\', '/' -replace '^([A-Z]):', '/mnt/$1' -replace '([A-Z])', {$_.Value.ToLower()}

# Run the makefile in WSL
wsl bash -c "cd '$WslPath' && source .venv_wsl/bin/activate && make -f Makefile.wsl run"

Write-Host ""
Write-Host "Notebook execution complete!" -ForegroundColor Green

# Extract and log results
Write-Host "Extracting model results..." -ForegroundColor Cyan
wsl bash -c "cd '$WslPath' && source .venv_wsl/bin/activate && python3 extract_results.py"

Write-Host ""
Write-Host "Results logged to model_results.log" -ForegroundColor Green
Write-Host "Check the notebooks and log file for detailed results." -ForegroundColor Cyan
