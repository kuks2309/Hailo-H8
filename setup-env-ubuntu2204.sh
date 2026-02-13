#!/bin/bash
# =============================================================================
# Hailo-H8 Environment Setup - Ubuntu 22.04 (Native)
# HailoRT-Ui + Hailo-Compiler-UI 통합 설치
# Usage: bash setup-env-ubuntu2204.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/hailo_env"
INSTALL_SW="$SCRIPT_DIR/install_sw"
HAILORT_REQ="$SCRIPT_DIR/HailoRT-Ui/requirements.txt"
COMPILER_REQ="$SCRIPT_DIR/Hailo-Compiler-UI/requirements.txt"

echo "================================================"
echo "  Hailo-H8 Environment Setup"
echo "  Ubuntu 22.04 (Native)"
echo "  HailoRT-Ui + Hailo-Compiler-UI"
echo "================================================"
echo ""

# -----------------------------------------------------------------------------
# STEP 1/7: OS Detection
# -----------------------------------------------------------------------------
echo "[STEP 1/7] OS detection..."

if grep -qi microsoft /proc/version 2>/dev/null; then
    echo "[ERROR] WSL detected. Use setup-env-wsl2.sh instead."
    echo "  Run: bash setup-env-wsl2.sh"
    exit 1
fi

if ! grep -q "22.04" /etc/lsb-release 2>/dev/null; then
    echo "[WARN] This script is designed for Ubuntu 22.04."
    read -p "  Continue anyway? (y/N): " PROCEED
    [ "$PROCEED" != "y" ] && [ "$PROCEED" != "Y" ] && exit 0
fi
echo "[OK] Ubuntu detected"

# -----------------------------------------------------------------------------
# STEP 2/7: System Packages
# -----------------------------------------------------------------------------
echo ""
echo "[STEP 2/7] Installing system packages..."
sudo apt update
sudo apt install -y \
    python3.10 python3.10-venv python3.10-dev \
    python3-tk \
    graphviz graphviz-dev
echo "[OK] System packages installed"

# -----------------------------------------------------------------------------
# STEP 3/7: Python Detection & Virtual Environment
# -----------------------------------------------------------------------------
echo ""
echo "[STEP 3/7] Setting up Python virtual environment..."

PYTHON_CMD=""
for cmd in python3.10 python3 python; do
    if command -v "$cmd" &>/dev/null; then
        PY_VER=$("$cmd" --version 2>&1 | grep -oP '\d+\.\d+')
        PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
        PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
        if [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -ge 10 ]; then
            PYTHON_CMD="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "[ERROR] Python 3.10+ required."
    echo "  Install: sudo apt install python3.10 python3.10-venv"
    exit 1
fi
echo "[OK] Python: $($PYTHON_CMD --version)"

if [ -d "$VENV_DIR" ]; then
    echo "[INFO] Virtual environment already exists: $VENV_DIR"
    read -p "  Recreate? (y/N): " RECREATE
    if [ "$RECREATE" = "y" ] || [ "$RECREATE" = "Y" ]; then
        rm -rf "$VENV_DIR"
        echo "[INFO] Removed old environment"
    else
        echo "[INFO] Using existing environment"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    $PYTHON_CMD -m venv "$VENV_DIR"
    echo "[OK] Created: $VENV_DIR"
fi

# -----------------------------------------------------------------------------
# STEP 4/7: pip Upgrade
# -----------------------------------------------------------------------------
echo ""
echo "[STEP 4/7] Upgrading pip..."
source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel --quiet
echo "[OK] pip upgraded"

# -----------------------------------------------------------------------------
# STEP 5/7: Install Dependencies (pinned versions first)
# -----------------------------------------------------------------------------
echo ""
echo "[STEP 5/7] Installing Python dependencies..."

# Pin onnx and protobuf FIRST (Hailo SDK compatibility)
echo "  Installing pinned versions (onnx==1.16.0, protobuf==3.20.3)..."
pip install onnx==1.16.0 --quiet
pip install protobuf==3.20.3 --quiet

# HailoRT-Ui dependencies
if [ -f "$HAILORT_REQ" ]; then
    echo "  Installing HailoRT-Ui dependencies..."
    pip install -r "$HAILORT_REQ"
else
    echo "[WARN] HailoRT-Ui/requirements.txt not found: $HAILORT_REQ"
fi

# Hailo-Compiler-UI dependencies
if [ -f "$COMPILER_REQ" ]; then
    echo "  Installing Hailo-Compiler-UI dependencies..."
    pip install -r "$COMPILER_REQ"
else
    echo "[WARN] Hailo-Compiler-UI/requirements.txt not found: $COMPILER_REQ"
fi

# Re-pin after requirements install (ultralytics/yolov5 may override)
echo "  Re-verifying pinned versions..."
pip install onnx==1.16.0 protobuf==3.20.3 --quiet
echo "[OK] Dependencies installed"

# -----------------------------------------------------------------------------
# STEP 6/7: Hailo SDK Installation
# -----------------------------------------------------------------------------
echo ""
echo "[STEP 6/7] Installing Hailo SDK..."

if [ -d "$INSTALL_SW" ]; then
    ARCH=$(uname -m)
    echo "[INFO] Architecture: $ARCH"

    # HailoRT Python binding
    HAILORT_WHL=$(ls "$INSTALL_SW"/hailort-*-"$ARCH".whl 2>/dev/null | head -1)
    if [ -z "$HAILORT_WHL" ]; then
        HAILORT_WHL=$(ls "$INSTALL_SW"/hailort-*cp310*.whl 2>/dev/null | grep "$ARCH" | head -1)
    fi
    if [ -n "$HAILORT_WHL" ]; then
        echo "  Installing HailoRT: $(basename "$HAILORT_WHL")"
        pip install "$HAILORT_WHL" --quiet || echo "  [WARN] HailoRT install failed"
    else
        echo "  [SKIP] HailoRT wheel not found for $ARCH"
    fi

    # Hailo Dataflow Compiler
    DFC_WHL=$(ls "$INSTALL_SW"/hailo_dataflow_compiler-*.whl 2>/dev/null | head -1)
    if [ -n "$DFC_WHL" ]; then
        echo "  Installing Dataflow Compiler: $(basename "$DFC_WHL")"
        pip install "$DFC_WHL" --quiet || echo "  [WARN] DFC install failed"
    else
        echo "  [SKIP] Dataflow Compiler wheel not found"
    fi

    # Hailo Model Zoo
    HMZ_WHL=$(ls "$INSTALL_SW"/hailo_model_zoo-*.whl 2>/dev/null | head -1)
    if [ -n "$HMZ_WHL" ]; then
        echo "  Installing Model Zoo: $(basename "$HMZ_WHL")"
        pip install "$HMZ_WHL" --quiet || echo "  [WARN] Model Zoo install failed"
    else
        echo "  [SKIP] Model Zoo wheel not found"
    fi

    # PCIe driver (interactive)
    HAILORT_DEB=$(ls "$INSTALL_SW"/hailort_*.deb 2>/dev/null | head -1)
    PCIE_DEB=$(ls "$INSTALL_SW"/hailort-pcie-driver_*.deb 2>/dev/null | head -1)
    if [ -n "$HAILORT_DEB" ] || [ -n "$PCIE_DEB" ]; then
        echo ""
        read -p "  Install HailoRT system packages and PCIe driver? (y/N): " INSTALL_DEB
        if [ "$INSTALL_DEB" = "y" ] || [ "$INSTALL_DEB" = "Y" ]; then
            [ -n "$HAILORT_DEB" ] && sudo dpkg -i "$HAILORT_DEB"
            [ -n "$PCIE_DEB" ] && sudo dpkg -i "$PCIE_DEB"
            sudo modprobe hailo_pci || echo "  [WARN] Could not load hailo_pci module"
            echo "[OK] PCIe driver installed"
        fi
    fi
else
    echo "  [SKIP] install_sw/ directory not found"
    echo "    Download Hailo SDK from: https://hailo.ai/developer-zone/"
    echo "    Place files in: $INSTALL_SW/"
fi

# -----------------------------------------------------------------------------
# STEP 7/7: Verification
# -----------------------------------------------------------------------------
echo ""
echo "[STEP 7/7] Verifying installation..."
echo ""

# Critical version check
echo "=== Version Check ==="
python -c "import onnx; print(f'  onnx: {onnx.__version__}')" 2>/dev/null || echo "  onnx: NOT INSTALLED"
python -c "import google.protobuf; print(f'  protobuf: {google.protobuf.__version__}')" 2>/dev/null || echo "  protobuf: NOT INSTALLED"
echo ""

# HailoRT-Ui environment check
echo "=== HailoRT-Ui Environment ==="
cd "$SCRIPT_DIR/HailoRT-Ui"
python -c "
import sys
sys.path.insert(0, 'src')
from utils.environment import check_environment
result = check_environment()
for key, info in result['packages'].items():
    status = '[OK]' if info['installed'] else '[MISSING]'
    version = info['version'] or 'N/A'
    req = '(required)' if info['required'] else '(optional)'
    print(f'  {status} {key:20s} {version:15s} {req}')
" 2>/dev/null || echo "  [WARN] Could not run HailoRT-Ui environment check"

echo ""

# Hailo-Compiler-UI environment check
echo "=== Hailo-Compiler-UI Environment ==="
cd "$SCRIPT_DIR/Hailo-Compiler-UI"
python -c "
import sys
sys.path.insert(0, '.')
from src.core.environment import check_environment
result = check_environment()
for key, info in result['packages'].items():
    status = '[OK]' if info.installed else '[MISSING]'
    version = info.version or 'N/A'
    req = '(required)' if info.required else '(optional)'
    print(f'  {status} {info.name:20s} {version:15s} {req}')
" 2>/dev/null || echo "  [WARN] Could not run Hailo-Compiler-UI environment check"

cd "$SCRIPT_DIR"

echo ""
echo "================================================"
echo "  Setup Complete!"
echo "================================================"
echo ""
echo "  Activate:  source $VENV_DIR/bin/activate"
echo ""
echo "  Run HailoRT-Ui:"
echo "    python $SCRIPT_DIR/HailoRT-Ui/main.py"
echo ""
echo "  Run Hailo-Compiler-UI:"
echo "    python $SCRIPT_DIR/Hailo-Compiler-UI/main.py"
echo ""
