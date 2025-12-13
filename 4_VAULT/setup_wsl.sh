#!/bin/bash
# CS506 Project - WSL Setup Script
# Sets up virtual environment and installs all dependencies

set -e  # Exit on error

echo "============================================================"
echo "CS506 Project - WSL Environment Setup"
echo "============================================================"
echo ""

# Navigate to project directory
PROJECT_DIR="/mnt/c/Users/Matt/Desktop/CS506/CS506_Project"
cd "$PROJECT_DIR"

echo "Working directory: $PWD"
echo ""

# Check Python version
echo "[1/5] Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed!"
    echo "Install with: sudo apt update && sudo apt install python3 python3-venv python3-pip"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "Found: $PYTHON_VERSION"
echo ""

# Create virtual environment
echo "[2/5] Creating virtual environment..."
VENV_DIR=".venv_wsl"

if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
else
    python3 -m venv "$VENV_DIR"
    echo "Created virtual environment: $VENV_DIR"
fi
echo ""

# Activate virtual environment
echo "[3/5] Activating virtual environment..."
source "$VENV_DIR/bin/activate"
echo "Virtual environment activated"
echo ""

# Upgrade pip
echo "[4/5] Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo ""

# Install dependencies
echo "[5/5] Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Install additional notebook execution tools
echo ""
echo "Installing notebook execution tools..."
pip install papermill nbconvert nbclient

echo ""
echo "============================================================"
echo "Setup Complete!"
echo "============================================================"
echo ""
echo "Virtual environment: $VENV_DIR"
echo "Python: $(python --version)"
echo "Pip packages installed: $(pip list | wc -l)"
echo ""
echo "To activate this environment manually:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To run notebooks:"
echo "  make run"
echo "============================================================"
