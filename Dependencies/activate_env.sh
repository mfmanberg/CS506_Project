#!/bin/bash
# Activate Virtual Environment - Bash Script (WSL/Linux)
# Usage: source Dependencies/activate_env.sh
#    or: . Dependencies/activate_env.sh

echo "============================================"
echo "CS506 Project - Virtual Environment Setup"
echo "============================================"
echo ""

# Get project root (parent of Dependencies folder)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Check for .venv_wsl (primary environment)
VENV_PATH="$PROJECT_ROOT/.venv_wsl"

if [ -d "$VENV_PATH" ]; then
    echo "✓ Found virtual environment: .venv_wsl"
    
    # Check if activate script exists
    ACTIVATE_SCRIPT="$VENV_PATH/bin/activate"
    
    if [ -f "$ACTIVATE_SCRIPT" ]; then
        echo "Activating environment..."
        source "$ACTIVATE_SCRIPT"
        echo ""
        echo "✓ Virtual environment activated!"
        echo "Python location: $(which python)"
        echo "Python version: $(python --version)"
        echo ""
        echo "To deactivate, run: deactivate"
    else
        echo "✗ Activation script not found at: $ACTIVATE_SCRIPT"
        echo "Run setup first: python3 -m venv .venv_wsl"
        return 1
    fi
else
    echo "✗ Virtual environment not found at: $VENV_PATH"
    echo ""
    echo "Create the environment first:"
    echo "  1. cd $PROJECT_ROOT"
    echo "  2. python3 -m venv .venv_wsl"
    echo "  3. source .venv_wsl/bin/activate"
    echo "  4. pip install -r Dependencies/requirements.txt"
    return 1
fi
