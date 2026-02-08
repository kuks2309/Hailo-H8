#!/bin/bash
# Hailo-H8 Development Environment Setup Script
# Usage: bash setup_env.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/hailo_env"
INSTALL_SW="$SCRIPT_DIR/install_sw"
REQ_FILE="$SCRIPT_DIR/HailoRT-Ui/requirements.txt"

echo "================================================"
echo "  Hailo-H8 Environment Setup"
echo "================================================"

# 1. Check Python version
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
    echo "[ERROR] Python 3.10+ required. Install: sudo apt install python3.10 python3.10-venv"
    exit 1
fi
echo "[OK] Python: $($PYTHON_CMD --version)"

# 2. Create virtual environment
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
    echo "[STEP 1/4] Creating virtual environment..."
    $PYTHON_CMD -m venv "$VENV_DIR"
    echo "[OK] Created: $VENV_DIR"
fi

# 3. Activate and upgrade pip
source "$VENV_DIR/bin/activate"
echo "[STEP 2/4] Upgrading pip..."
pip install --upgrade pip setuptools wheel --quiet

# 4. Install requirements
echo "[STEP 3/4] Installing dependencies..."
if [ -f "$REQ_FILE" ]; then
    pip install -r "$REQ_FILE"
else
    echo "[ERROR] requirements.txt not found: $REQ_FILE"
    deactivate
    exit 1
fi

# 5. Install Hailo SDK (local wheel files)
echo "[STEP 4/4] Installing Hailo SDK..."
if [ -d "$INSTALL_SW" ]; then
    # Detect architecture
    ARCH=$(uname -m)
    echo "[INFO] Architecture: $ARCH"

    # HailoRT
    HAILORT_WHL=$(ls "$INSTALL_SW"/hailort-*-"$ARCH".whl 2>/dev/null | head -1)
    if [ -z "$HAILORT_WHL" ]; then
        # Try cp310 specific
        HAILORT_WHL=$(ls "$INSTALL_SW"/hailort-*cp310*.whl 2>/dev/null | grep "$ARCH" | head -1)
    fi
    if [ -n "$HAILORT_WHL" ]; then
        echo "  Installing HailoRT: $(basename "$HAILORT_WHL")"
        pip install "$HAILORT_WHL" --quiet || echo "  [WARN] HailoRT install failed (may need HailoRT driver)"
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
else
    echo "  [SKIP] install_sw/ directory not found (Hailo SDK not installed)"
fi

echo ""
echo "================================================"
echo "  Setup Complete!"
echo "================================================"
echo ""
echo "  Activate:  source $VENV_DIR/bin/activate"
echo "  Run App:   python $SCRIPT_DIR/HailoRT-Ui/main.py"
echo ""
echo "  Quick start:"
echo "    source hailo_env/bin/activate && python HailoRT-Ui/main.py"
echo ""
