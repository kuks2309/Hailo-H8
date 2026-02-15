#!/bin/bash
# =============================================================================
# Hailo-H8 Environment Setup - WSL2
# Hailo-Compiler-UI 전용 (모델 변환: PT -> ONNX -> HEF)
# Usage: bash setup-env-wsl2.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/hailo_env"
INSTALL_SW="$SCRIPT_DIR/install_sw"
COMPILER_REQ="$SCRIPT_DIR/Hailo-Compiler-UI/requirements.txt"

echo "================================================"
echo "  Hailo-H8 Environment Setup"
echo "  WSL2 (Hailo-Compiler-UI)"
echo "================================================"
echo ""

# -----------------------------------------------------------------------------
# STEP 1/7: WSL Detection
# -----------------------------------------------------------------------------
echo "[STEP 1/7] WSL detection..."

if ! grep -qi microsoft /proc/version 2>/dev/null; then
    echo "[WARN] WSL not detected. This script is intended for WSL2."
    echo "  For native Ubuntu 22.04, use: bash setup-env-ubuntu2204.sh"
    read -p "  Continue anyway? (y/N): " PROCEED
    [ "$PROCEED" != "y" ] && [ "$PROCEED" != "Y" ] && exit 0
fi
echo "[OK] WSL2 detected"

# -----------------------------------------------------------------------------
# STEP 2/7: System Packages
# -----------------------------------------------------------------------------
echo ""
echo "[STEP 2/7] Installing system packages..."
sudo apt update
sudo apt install -y \
    python3.10 python3.10-venv python3.10-dev \
    python3-tk \
    graphviz graphviz-dev \
    libxcb-xinerama0 libxkbcommon-x11-0
echo "[OK] System packages installed"

# -----------------------------------------------------------------------------
# STEP 3/7: Memory Check
# -----------------------------------------------------------------------------
echo ""
echo "[STEP 3/7] Checking WSL2 memory..."

MEM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
MEM_GB=$((MEM_KB / 1024 / 1024))
echo "[INFO] Current memory: ${MEM_GB}GB"

if [ "$MEM_GB" -lt 12 ]; then
    echo "[WARN] Hailo SDK compilation requires at least 12GB RAM."
    echo ""
    echo "  Create/edit %USERPROFILE%\\.wslconfig on Windows:"
    echo "    [wsl2]"
    echo "    memory=12GB"
    echo "    processors=4"
    echo "    swap=4GB"
    echo ""
    echo "  Then run in PowerShell: wsl --shutdown"
    echo ""
    read -p "  Continue with current memory? (y/N): " PROCEED
    [ "$PROCEED" != "y" ] && [ "$PROCEED" != "Y" ] && exit 0
else
    echo "[OK] Memory sufficient (${MEM_GB}GB >= 12GB)"
fi

# -----------------------------------------------------------------------------
# STEP 4/7: Python Detection & Virtual Environment
# -----------------------------------------------------------------------------
echo ""
echo "[STEP 4/7] Setting up Python virtual environment..."

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
# STEP 5/7: pip Upgrade & Pinned Dependencies
# -----------------------------------------------------------------------------
echo ""
echo "[STEP 5/7] Installing pinned dependencies (order matters)..."
source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel --quiet

# CRITICAL: Pin onnx and protobuf BEFORE other packages
# hailo_sdk_client requires onnx<=1.16.x (onnx.mapping removed in 1.17+)
# hailo_sdk_client requires protobuf 3.x (not 4.x)
echo "  Installing onnx==1.16.0 (hailo_sdk_client compatibility)..."
pip install onnx==1.16.0 --quiet
echo "  Installing protobuf==3.20.3 (hailo_sdk_client compatibility)..."
pip install protobuf==3.20.3 --quiet
echo "[OK] Pinned versions installed"

# -----------------------------------------------------------------------------
# STEP 6/7: Install Compiler-UI Dependencies & Hailo SDK
# -----------------------------------------------------------------------------
echo ""
echo "[STEP 6/7] Installing Hailo-Compiler-UI dependencies..."

if [ -f "$COMPILER_REQ" ]; then
    pip install -r "$COMPILER_REQ"
else
    echo "[ERROR] requirements.txt not found: $COMPILER_REQ"
    deactivate
    exit 1
fi

# Re-pin after requirements install (ultralytics/yolov5 may override)
echo "  Re-verifying pinned versions..."
pip install onnx==1.16.0 protobuf==3.20.3 --quiet
echo "[OK] Dependencies installed"

# Hailo SDK wheels
echo ""
echo "  Installing Hailo SDK..."

GDRIVE_FOLDER_URL="https://drive.google.com/drive/folders/1pAv-qojczbuIskbWu0W_0sac6LhJRF66"

# Count available Hailo SDK wheels
_count_hailo_wheels() {
    local count=0
    ls "$INSTALL_SW"/hailort-*.whl &>/dev/null && ((count++)) || true
    ls "$INSTALL_SW"/hailo_dataflow_compiler-*.whl &>/dev/null && ((count++)) || true
    ls "$INSTALL_SW"/hailo_model_zoo-*.whl &>/dev/null && ((count++)) || true
    echo "$count"
}

# Download from Google Drive if install_sw is missing or incomplete
if [ ! -d "$INSTALL_SW" ] || [ "$(_count_hailo_wheels)" -lt 3 ]; then
    echo "  [INFO] Hailo SDK packages not found (or incomplete) in install_sw/"
    echo ""
    echo "  Required files:"
    echo "    - hailort-<version>.whl"
    echo "    - hailo_dataflow_compiler-<version>.whl"
    echo "    - hailo_model_zoo-<version>.whl"
    echo ""
    echo "  Options:"
    echo "    1) Auto-download from Google Drive (gdown)"
    echo "    2) Skip (download manually later)"
    echo ""
    read -p "  Select (1/2): " DL_CHOICE

    if [ "$DL_CHOICE" = "1" ]; then
        echo ""
        echo "  Installing gdown for Google Drive download..."
        pip install gdown --quiet

        mkdir -p "$INSTALL_SW"
        echo "  Downloading Hailo SDK from Google Drive..."
        echo "  URL: $GDRIVE_FOLDER_URL"
        echo ""
        gdown --folder "$GDRIVE_FOLDER_URL" -O "$INSTALL_SW" --remaining-ok || {
            echo ""
            echo "  [ERROR] Auto-download failed. Please download manually:"
            echo "    URL: ${GDRIVE_FOLDER_URL}?usp=sharing"
            echo "    Place files in: $INSTALL_SW/"
        }

        # gdown preserves Google Drive folder structure (e.g. ubuntu/install_sw/)
        # Move .whl and .deb files from subdirectories to install_sw root
        find "$INSTALL_SW" -mindepth 2 -name "*.whl" -exec mv {} "$INSTALL_SW"/ \;
        find "$INSTALL_SW" -mindepth 2 -name "*.deb" -exec mv {} "$INSTALL_SW"/ \;

        # Clean up: Zone.Identifier files, Windows installers, empty dirs
        find "$INSTALL_SW" -name "*Zone.Identifier" -delete 2>/dev/null || true
        find "$INSTALL_SW" -name "*.msi" -delete 2>/dev/null || true
        find "$INSTALL_SW" -mindepth 1 -type d -empty -delete 2>/dev/null || true

        echo ""
        echo "  [INFO] Downloaded $(_count_hailo_wheels)/3 wheel files"
    else
        echo "  [SKIP] Manual download required"
        echo "    URL: ${GDRIVE_FOLDER_URL}?usp=sharing"
        echo "    Place files in: $INSTALL_SW/"
    fi
fi

# Install Hailo SDK wheels
if [ -d "$INSTALL_SW" ]; then
    # Dataflow Compiler (primary for WSL2 - provides hailo_sdk_client)
    DFC_WHL=$(ls "$INSTALL_SW"/hailo_dataflow_compiler-*.whl 2>/dev/null | head -1)
    if [ -n "$DFC_WHL" ]; then
        echo "    Installing Dataflow Compiler: $(basename "$DFC_WHL")"
        pip install "$DFC_WHL" --quiet || echo "    [WARN] DFC install failed"
    else
        echo "    [SKIP] Dataflow Compiler wheel not found"
    fi

    # HailoRT Python binding (optional for WSL2)
    HAILORT_WHL=$(ls "$INSTALL_SW"/hailort-*cp310*.whl 2>/dev/null | head -1)
    if [ -n "$HAILORT_WHL" ]; then
        echo "    Installing HailoRT: $(basename "$HAILORT_WHL")"
        pip install "$HAILORT_WHL" --quiet || echo "    [WARN] HailoRT install failed"
    else
        echo "    [SKIP] HailoRT wheel not found"
    fi

    # Model Zoo (optional)
    HMZ_WHL=$(ls "$INSTALL_SW"/hailo_model_zoo-*.whl 2>/dev/null | head -1)
    if [ -n "$HMZ_WHL" ]; then
        echo "    Installing Model Zoo: $(basename "$HMZ_WHL")"
        pip install "$HMZ_WHL" --quiet || echo "    [WARN] Model Zoo install failed"
    else
        echo "    [SKIP] Model Zoo wheel not found"
    fi
else
    echo "  [SKIP] install_sw/ directory not found"
    echo "    Download from: ${GDRIVE_FOLDER_URL}?usp=sharing"
    echo "    Or from: https://hailo.ai/developer-zone/"
    echo "    Place .whl files in: $INSTALL_SW/"
fi

# -----------------------------------------------------------------------------
# STEP 7/7: Verification
# -----------------------------------------------------------------------------
echo ""
echo "[STEP 7/7] Verifying installation..."
echo ""

# Critical version check
echo "=== Critical Version Check ==="
ONNX_VER=$(python -c "import onnx; print(onnx.__version__)" 2>/dev/null)
PROTO_VER=$(python -c "import google.protobuf; print(google.protobuf.__version__)" 2>/dev/null)

if [ "$ONNX_VER" = "1.16.0" ]; then
    echo "  [OK] onnx: $ONNX_VER"
else
    echo "  [ERROR] onnx: ${ONNX_VER:-NOT INSTALLED} (expected 1.16.0)"
    echo "    Fix: pip install onnx==1.16.0"
fi

if [ "$PROTO_VER" = "3.20.3" ]; then
    echo "  [OK] protobuf: $PROTO_VER"
else
    echo "  [ERROR] protobuf: ${PROTO_VER:-NOT INSTALLED} (expected 3.20.3)"
    echo "    Fix: pip install protobuf==3.20.3"
fi

# hailo_sdk_client check
python -c "import hailo_sdk_client; print('  [OK] hailo_sdk_client')" 2>/dev/null \
    || echo "  [SKIP] hailo_sdk_client: Not installed (install from install_sw/)"

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
if not result['all_required_ok']:
    print()
    print('  [ERROR] Some required packages are missing!')
    sys.exit(1)
" 2>/dev/null || echo "  [WARN] Could not run environment check"

cd "$SCRIPT_DIR"

echo ""
echo "================================================"
echo "  Setup Complete!"
echo "================================================"
echo ""
echo "  Activate:  source $VENV_DIR/bin/activate"
echo ""
echo "  Run Hailo-Compiler-UI:"
echo "    python $SCRIPT_DIR/Hailo-Compiler-UI/main.py"
echo ""
