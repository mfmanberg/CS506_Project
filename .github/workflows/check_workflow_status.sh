#!/bin/bash
# Monitor GitHub Workflows and Report Status
# This script checks workflow status using GitHub CLI and reports success/failure

echo "=========================================="
echo "GitHub Workflows Status Monitor"
echo "=========================================="
echo ""

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI (gh) is not installed"
    echo "Install from: https://cli.github.com/"
    exit 1
fi

# Check if authenticated
if ! gh auth status &> /dev/null; then
    echo "❌ Not authenticated with GitHub CLI"
    echo "Run: gh auth login"
    exit 1
fi

echo "✓ GitHub CLI authenticated"
echo ""

# Get list of workflows
echo "Available Workflows:"
echo "===================="
gh workflow list | while read -r line; do
    echo "  $line"
done
echo ""

# Check for in-progress workflows
echo "Checking for in-progress workflows..."
IN_PROGRESS=$(gh run list --limit 10 --json status,workflowName --jq '[.[] | select(.status == "in_progress")] | length')

if [ "$IN_PROGRESS" -gt 0 ]; then
    echo "⏳ Found $IN_PROGRESS workflow(s) currently running"
    echo ""
    echo "In-Progress Workflows:"
    gh run list --limit 10 --json status,workflowName,databaseId,createdAt --jq '.[] | select(.status == "in_progress") | "  - " + .workflowName + " (ID: " + (.databaseId | tostring) + ")"'
    echo ""
    
    echo "Waiting for workflows to complete..."
    echo "(Checking every 30 seconds, max 10 minutes)"
    echo ""
    
    # Wait for completion (max 10 minutes = 20 checks * 30 seconds)
    MAX_CHECKS=20
    CHECK_COUNT=0
    
    while [ "$IN_PROGRESS" -gt 0 ] && [ "$CHECK_COUNT" -lt "$MAX_CHECKS" ]; do
        sleep 30
        CHECK_COUNT=$((CHECK_COUNT + 1))
        IN_PROGRESS=$(gh run list --limit 10 --json status --jq '[.[] | select(.status == "in_progress")] | length')
        
        if [ "$IN_PROGRESS" -gt 0 ]; then
            echo "[Check $CHECK_COUNT/$MAX_CHECKS] Still running: $IN_PROGRESS workflow(s)..."
        fi
    done
    
    if [ "$IN_PROGRESS" -gt 0 ]; then
        echo ""
        echo "⚠️  Timeout: Workflows still running after 10 minutes"
        echo "Check status at: https://github.com/$(gh repo view --json nameWithOwner -q .nameWithOwner)/actions"
    fi
    echo ""
fi

# Get recent workflow run results
echo "=========================================="
echo "Recent Workflow Results (Last 5 Runs)"
echo "=========================================="
echo ""

gh run list --limit 15 --json conclusion,status,workflowName,databaseId,createdAt,headBranch \
    --jq 'group_by(.workflowName) | .[] | .[0] | select(.status == "completed")' | \
    jq -r '"Workflow: " + .workflowName + "\n  Status: " + .status + "\n  Result: " + .conclusion + "\n  Branch: " + .headBranch + "\n  Run ID: " + (.databaseId | tostring) + "\n  Time: " + .createdAt + "\n"'

# Summary
echo "=========================================="
echo "Workflow Summary"
echo "=========================================="

TOTAL_WORKFLOWS=$(gh workflow list --json name | jq '. | length')
RECENT_SUCCESS=$(gh run list --limit 20 --json conclusion,status --jq '[.[] | select(.status == "completed" and .conclusion == "success")] | length')
RECENT_FAILURE=$(gh run list --limit 20 --json conclusion,status --jq '[.[] | select(.status == "completed" and .conclusion == "failure")] | length')

echo ""
echo "Total Workflows: $TOTAL_WORKFLOWS"
echo "Recent Successes (last 20): $RECENT_SUCCESS"
echo "Recent Failures (last 20): $RECENT_FAILURE"
echo ""

# Check latest run of each workflow
echo "Latest Run Status Per Workflow:"
echo "================================"

WORKFLOWS=("Linear Regression Test" "SVM Daily Test" "XGBoost Testing")
ALL_SUCCESS=true

for workflow in "${WORKFLOWS[@]}"; do
    LATEST=$(gh run list --workflow "$workflow" --limit 1 --json conclusion,status,databaseId --jq '.[0]')
    
    if [ -n "$LATEST" ]; then
        STATUS=$(echo "$LATEST" | jq -r '.status')
        CONCLUSION=$(echo "$LATEST" | jq -r '.conclusion')
        RUN_ID=$(echo "$LATEST" | jq -r '.databaseId')
        
        if [ "$STATUS" == "completed" ]; then
            if [ "$CONCLUSION" == "success" ]; then
                echo "  ✅ $workflow: SUCCESS (Run #$RUN_ID)"
            else
                echo "  ❌ $workflow: $CONCLUSION (Run #$RUN_ID)"
                ALL_SUCCESS=false
                
                # Get failure details
                echo "     Viewing logs..."
                gh run view "$RUN_ID" --log-failed | head -20
            fi
        else
            echo "  ⏳ $workflow: $STATUS"
            ALL_SUCCESS=false
        fi
    else
        echo "  ⚠️  $workflow: No runs found"
        ALL_SUCCESS=false
    fi
done

echo ""
echo "=========================================="

if [ "$ALL_SUCCESS" = true ]; then
    echo "✅ ALL WORKFLOWS SUCCESSFUL"
    echo ""
    echo "All 3 workflows passed their latest run:"
    echo "  - Linear Regression Test"
    echo "  - SVM Daily Test"
    echo "  - XGBoost Testing"
    echo ""
    echo "Code is reproducible and ready for production! 🚀"
    exit 0
else
    echo "❌ SOME WORKFLOWS FAILED OR INCOMPLETE"
    echo ""
    echo "Please review the failures above and fix issues."
    echo "View details at: https://github.com/$(gh repo view --json nameWithOwner -q .nameWithOwner)/actions"
    echo ""
    echo "Common fixes:"
    echo "  1. Check PYTHONPATH is set in workflows"
    echo "  2. Verify --cwd flag in papermill commands"
    echo "  3. Ensure Git LFS is configured"
    echo "  4. Run: bash .github/workflows/test_reproducibility.sh"
    exit 1
fi
