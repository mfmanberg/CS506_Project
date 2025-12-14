#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Trigger GitHub Actions workflows using GitHub CLI
    
.DESCRIPTION
    Reproducible script to trigger all GitHub Actions workflows for the CS506 project.
    Automatically detects repository location and triggers workflows.
    
.PARAMETER WorkflowName
    Optional: Name of specific workflow to trigger. If not specified, all workflows are triggered.
    
.PARAMETER ListOnly
    If specified, only list available workflows without triggering them.
    
.EXAMPLE
    .\trigger_workflows.ps1
    Triggers all workflows
    
.EXAMPLE
    .\trigger_workflows.ps1 -ListOnly
    Lists all available workflows
    
.EXAMPLE
    .\trigger_workflows.ps1 -WorkflowName "Linear Regression Test"
    Triggers only the Linear Regression Test workflow
#>

param(
    [Parameter(Mandatory=$false)]
    [string]$WorkflowName,
    
    [Parameter(Mandatory=$false)]
    [switch]$ListOnly
)

# Colors for output
function Write-Success { param($msg) Write-Host "✅ $msg" -ForegroundColor Green }
function Write-Error { param($msg) Write-Host "❌ $msg" -ForegroundColor Red }
function Write-Info { param($msg) Write-Host "ℹ️  $msg" -ForegroundColor Cyan }
function Write-Warning { param($msg) Write-Host "⚠️  $msg" -ForegroundColor Yellow }

# Check prerequisites
Write-Info "Checking prerequisites..."

# Check if GitHub CLI is installed
try {
    $ghVersion = gh --version 2>&1 | Select-Object -First 1
    Write-Success "GitHub CLI installed: $ghVersion"
} catch {
    Write-Error "GitHub CLI (gh) not found. Install from: https://cli.github.com/"
    Write-Info "Windows: winget install GitHub.cli"
    exit 1
}

# Check if authenticated
try {
    $authStatus = gh auth status 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Not authenticated with GitHub CLI"
        Write-Info "Run: gh auth login"
        exit 1
    }
    Write-Success "Authenticated with GitHub CLI"
} catch {
    Write-Error "Failed to check authentication status"
    exit 1
}

# Auto-detect repository root (go up from script directory until we find .git)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$currentDir = $scriptDir
$repoRoot = $null

while ($currentDir) {
    if (Test-Path (Join-Path $currentDir ".git")) {
        $repoRoot = $currentDir
        break
    }
    $parent = Split-Path -Parent $currentDir
    if ($parent -eq $currentDir) { break }  # Reached root
    $currentDir = $parent
}

if (-not $repoRoot) {
    Write-Error "Could not find Git repository root"
    exit 1
}

Write-Success "Repository root: $repoRoot"

# Change to repository root
Push-Location $repoRoot

try {
    # Get repository information
    Write-Info "Getting repository information..."
    $repoInfo = gh repo view --json nameWithOwner,defaultBranchRef | ConvertFrom-Json
    $repoName = $repoInfo.nameWithOwner
    $defaultBranch = $repoInfo.defaultBranchRef.name
    
    Write-Success "Repository: $repoName"
    Write-Success "Default branch: $defaultBranch"
    
    # List workflows
    Write-Info "Fetching available workflows..."
    $workflows = gh workflow list --json name,id,path,state | ConvertFrom-Json
    
    if ($workflows.Count -eq 0) {
        Write-Warning "No workflows found in repository"
        exit 0
    }
    
    Write-Success "Found $($workflows.Count) workflow(s):"
    $workflows | ForEach-Object {
        $status = if ($_.state -eq 'active') { '✅ active' } else { '⚠️  ' + $_.state }
        Write-Host "  - $($_.name) ($status)"
        Write-Host "    Path: $($_.path)" -ForegroundColor Gray
    }
    
    if ($ListOnly) {
        Write-Info "List-only mode. Exiting."
        exit 0
    }
    
    # Filter workflows if specific name provided
    $workflowsToTrigger = if ($WorkflowName) {
        $filtered = $workflows | Where-Object { $_.name -like "*$WorkflowName*" }
        if ($filtered.Count -eq 0) {
            Write-Error "No workflow found matching: $WorkflowName"
            exit 1
        }
        $filtered
    } else {
        $workflows
    }
    
    Write-Host "`n" + ("="*70)
    Write-Host "TRIGGERING WORKFLOWS"
    Write-Host ("="*70)
    
    $triggered = @()
    foreach ($workflow in $workflowsToTrigger) {
        if ($workflow.state -ne 'active') {
            Write-Warning "Skipping inactive workflow: $($workflow.name)"
            continue
        }
        
        Write-Info "Triggering: $($workflow.name)"
        try {
            # Trigger the workflow using workflow_dispatch
            gh workflow run $workflow.id --ref $defaultBranch 2>&1 | Out-Null
            
            if ($LASTEXITCODE -eq 0) {
                Write-Success "Triggered: $($workflow.name)"
                $triggered += $workflow
                Start-Sleep -Seconds 2  # Brief delay between triggers
            } else {
                Write-Error "Failed to trigger: $($workflow.name)"
            }
        } catch {
            Write-Error "Error triggering $($workflow.name): $_"
        }
    }
    
    if ($triggered.Count -eq 0) {
        Write-Warning "No workflows were triggered"
        exit 0
    }
    
    Write-Host "`n" + ("="*70)
    Write-Host "MONITORING WORKFLOW RUNS"
    Write-Host ("="*70)
    
    Write-Info "Waiting for workflow runs to appear (5 seconds)..."
    Start-Sleep -Seconds 5
    
    # Get recent runs
    Write-Info "Fetching recent workflow runs..."
    $runs = gh run list --limit 10 --json databaseId,name,status,conclusion,createdAt,htmlUrl,workflowName | ConvertFrom-Json
    
    if ($runs.Count -eq 0) {
        Write-Warning "No recent runs found. Workflows may still be queuing."
        Write-Info "Check status at: https://github.com/$repoName/actions"
        exit 0
    }
    
    # Display recent runs
    Write-Success "Recent workflow runs:"
    $runs | Select-Object -First $triggered.Count | ForEach-Object {
        $statusIcon = switch ($_.status) {
            'completed' { 
                switch ($_.conclusion) {
                    'success' { '✅' }
                    'failure' { '❌' }
                    'cancelled' { '⚠️' }
                    default { '❓' }
                }
            }
            'in_progress' { '🔄' }
            'queued' { '⏳' }
            default { '❓' }
        }
        
        $statusText = if ($_.status -eq 'completed') { $_.conclusion } else { $_.status }
        Write-Host "  $statusIcon $($_.workflowName) - $statusText"
        Write-Host "    URL: $($_.htmlUrl)" -ForegroundColor Gray
    }
    
    Write-Host "`n" + ("="*70)
    Write-Host "SUMMARY"
    Write-Host ("="*70)
    Write-Success "Triggered $($triggered.Count) workflow(s)"
    Write-Info "Monitor progress at: https://github.com/$repoName/actions"
    
    # Provide commands for monitoring
    Write-Host "`nUseful commands:"
    Write-Host "  gh run list                    # List recent runs" -ForegroundColor Gray
    Write-Host "  gh run list --limit 20         # List more runs" -ForegroundColor Gray
    Write-Host "  gh run watch                   # Watch latest run" -ForegroundColor Gray
    Write-Host "  gh run view <run_id>           # View specific run details" -ForegroundColor Gray
    
} finally {
    Pop-Location
}
