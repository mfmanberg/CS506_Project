#!/bin/bash
# Launcher script for export_svr_animations.py
# Automatically activates environment and runs the script

set -e  # Exit on error

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "============================================"
echo "SVR Animation Generator - Launcher"
echo "============================================"
echo ""

# Check if virtual environment exists
VENV_PATH="$PROJECT_ROOT/.venv_wsl"

if [ ! -d "$VENV_PATH" ]; then
    echo "ERROR: Virtual environment not found at: $VENV_PATH"
    echo ""
    echo "Create the environment first:"
    echo "  cd $PROJECT_ROOT"
    echo "  python3 -m venv .venv_wsl"
    echo "  source .venv_wsl/bin/activate"
    echo "  pip install -r Dependencies/requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# Check Python version
PYTHON_VERSION=$(python --version)
echo "Using: $PYTHON_VERSION"
echo ""

# Run the script
cd "$PROJECT_ROOT"
python Build/export_svr_animations.py

echo ""
echo "============================================"
echo "Animation generation complete!"
echo "============================================"
