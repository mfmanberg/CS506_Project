# GitHub Workflows Status Monitor
# PowerShell script to check and report workflow status

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "GitHub Workflows Status Monitor" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check if gh CLI is available
try {
    $ghVersion = gh --version 2>&1 | Select-Object -First 1
    Write-Host "✓ GitHub CLI: $ghVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ GitHub CLI (gh) is not installed" -ForegroundColor Red
    Write-Host "Install from: https://cli.github.com/"
    exit 1
}

Write-Host ""

# Define workflows to check
$workflows = @("Linear Regression Test", "SVM Daily Test", "XGBoost Testing")

# Check for in-progress workflows
Write-Host "Checking workflow status..." -ForegroundColor Yellow
$allRuns = gh run list --limit 10 --json status,conclusion,workflowName,databaseId,createdAt | ConvertFrom-Json
$inProgress = $allRuns | Where-Object { $_.status -eq "in_progress" }

if ($inProgress.Count -gt 0) {
    Write-Host "⏳ Found $($inProgress.Count) workflow(s) currently running:" -ForegroundColor Yellow
    foreach ($run in $inProgress) {
        Write-Host "   - $($run.workflowName) (ID: $($run.databaseId))" -ForegroundColor Yellow
    }
    Write-Host ""
    Write-Host "Waiting for completion (checking every 30s, max 5 minutes)..." -ForegroundColor Yellow
    Write-Host ""
    
    $maxChecks = 10
    $checkCount = 0
    
    while ($inProgress.Count -gt 0 -and $checkCount -lt $maxChecks) {
        Start-Sleep -Seconds 30
        $checkCount++
        
        $allRuns = gh run list --limit 10 --json status,conclusion,workflowName,databaseId | ConvertFrom-Json
        $inProgress = $allRuns | Where-Object { $_.status -eq "in_progress" }
        
        if ($inProgress.Count -gt 0) {
            Write-Host "[Check $checkCount/$maxChecks] Still running: $($inProgress.Count) workflow(s)..." -ForegroundColor Yellow
        } else {
            Write-Host "[Check $checkCount/$maxChecks] All workflows completed!" -ForegroundColor Green
        }
    }
    
    if ($inProgress.Count -gt 0) {
        Write-Host ""
        Write-Host "⚠️  Timeout: Workflows still running after 5 minutes" -ForegroundColor Yellow
        Write-Host "Reporting current status..." -ForegroundColor Yellow
    }
    Write-Host ""
}

# Get latest status for each workflow
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Latest Workflow Status" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

$allSuccess = $true
$results = @()

foreach ($workflow in $workflows) {
    $latest = gh run list --workflow $workflow --limit 1 --json status,conclusion,databaseId,createdAt | ConvertFrom-Json | Select-Object -First 1
    
    if ($latest) {
        $result = [PSCustomObject]@{
            Workflow = $workflow
            Status = $latest.status
            Conclusion = $latest.conclusion
            RunID = $latest.databaseId
        }
        
        if ($latest.status -eq "completed") {
            if ($latest.conclusion -eq "success") {
                Write-Host "✅ $workflow" -ForegroundColor Green
                Write-Host "   Status: SUCCESS" -ForegroundColor Green
                Write-Host "   Run ID: $($latest.databaseId)" -ForegroundColor Gray
            } else {
                Write-Host "❌ $workflow" -ForegroundColor Red
                Write-Host "   Status: $($latest.conclusion.ToUpper())" -ForegroundColor Red
                Write-Host "   Run ID: $($latest.databaseId)" -ForegroundColor Gray
                $allSuccess = $false
                
                # Get failure logs
                Write-Host "   Fetching error logs..." -ForegroundColor Yellow
                $errorLogs = gh run view $($latest.databaseId) --log-failed 2>&1 | Select-Object -First 15
                foreach ($line in $errorLogs) {
                    Write-Host "     $line" -ForegroundColor DarkRed
                }
            }
        } else {
            Write-Host "⏳ $workflow" -ForegroundColor Yellow
            Write-Host "   Status: $($latest.status.ToUpper())" -ForegroundColor Yellow
            Write-Host "   Run ID: $($latest.databaseId)" -ForegroundColor Gray
            $allSuccess = $false
        }
        
        $results += $result
        Write-Host ""
    } else {
        Write-Host "⚠️  ${workflow}: No runs found" -ForegroundColor Yellow
        Write-Host ""
        $allSuccess = $false
    }
}

# Summary
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Summary" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

$completedSuccess = $results | Where-Object { $_.Status -eq "completed" -and $_.Conclusion -eq "success" }
$completedFailure = $results | Where-Object { $_.Status -eq "completed" -and $_.Conclusion -ne "success" -and $_.Conclusion -ne "" }
$inProgressCount = $results | Where-Object { $_.Status -eq "in_progress" }

Write-Host "Total Workflows: $($workflows.Count)" -ForegroundColor White
Write-Host "Successful: $($completedSuccess.Count)" -ForegroundColor Green
Write-Host "Failed: $($completedFailure.Count)" -ForegroundColor Red
Write-Host "In Progress: $($inProgressCount.Count)" -ForegroundColor Yellow
Write-Host ""

if ($allSuccess) {
    Write-Host "✅ ALL WORKFLOWS SUCCESSFUL!" -ForegroundColor Green
    Write-Host ""
    Write-Host "All 3 workflows passed their latest run:" -ForegroundColor Green
    Write-Host "  - Linear Regression Test ✅" -ForegroundColor Green
    Write-Host "  - SVM Daily Test ✅" -ForegroundColor Green
    Write-Host "  - XGBoost Testing ✅" -ForegroundColor Green
    Write-Host ""
    Write-Host "Code is reproducible and ready for production! 🚀" -ForegroundColor Green
    exit 0
} else {
    Write-Host "❌ SOME WORKFLOWS FAILED OR INCOMPLETE" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please review the failures above." -ForegroundColor Red
    Write-Host "View details at: https://github.com/mfmanberg/CS506_Project/actions" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Troubleshooting:" -ForegroundColor Yellow
    Write-Host "  1. Run: bash .github/workflows/test_reproducibility.sh" -ForegroundColor Gray
    Write-Host "  2. Check PYTHONPATH in workflow files" -ForegroundColor Gray
    Write-Host "  3. Verify --cwd flag in papermill commands" -ForegroundColor Gray
    Write-Host "  4. See: .github/workflows/TROUBLESHOOTING.md" -ForegroundColor Gray
    exit 1
}
