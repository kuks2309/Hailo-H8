#!/bin/bash
# Hailo-Compiler-UI Installation Script
# This script sets up the Python environment with correct dependency versions

set -e

echo "=== Hailo-Compiler-UI Installation ==="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "[INFO] Python version: $PYTHON_VERSION"

# Create virtual environment if not exists
VENV_PATH="${1:-hailo_env}"
if [ ! -d "$VENV_PATH" ]; then
    echo "[INFO] Creating virtual environment: $VENV_PATH"
    python3 -m venv "$VENV_PATH"
else
    echo "[INFO] Using existing virtual environment: $VENV_PATH"
fi

# Activate virtual environment
source "$VENV_PATH/bin/activate"

echo "[INFO] Installing dependencies..."

# Install base dependencies first
pip install --upgrade pip

# CRITICAL: Install onnx with pinned version FIRST
# hailo_sdk_client requires onnx<=1.16.x (uses onnx.mapping module removed in 1.17+)
echo "[INFO] Installing onnx==1.16.0 (pinned for hailo_sdk_client compatibility)"
pip install onnx==1.16.0

# Install protobuf with pinned version
# hailo_sdk_client requires protobuf 3.x
echo "[INFO] Installing protobuf==3.20.3 (pinned for hailo_sdk_client compatibility)"
pip install protobuf==3.20.3

# Install remaining dependencies
# Using --no-deps for packages that might upgrade onnx
echo "[INFO] Installing remaining dependencies..."
pip install -r requirements.txt

# Verify critical versions
echo ""
echo "=== Verifying Installation ==="
python3 -c "import onnx; print(f'onnx: {onnx.__version__}')"
python3 -c "import google.protobuf; print(f'protobuf: {google.protobuf.__version__}')"

# Check hailo_sdk_client
if python3 -c "import hailo_sdk_client" 2>/dev/null; then
    echo "hailo_sdk_client: OK"
else
    echo "hailo_sdk_client: Not installed (install separately from Hailo)"
fi

echo ""
echo "=== Installation Complete ==="
echo "To activate the environment: source $VENV_PATH/bin/activate"
echo "To run the application: python main.py"
