@echo off
REM Hailo-Compiler-UI Installation Script for Windows
REM This script sets up the Python environment with correct dependency versions

echo === Hailo-Compiler-UI Installation ===
echo.

REM Check Python
python --version
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.8+
    exit /b 1
)

REM Create virtual environment if not exists
set VENV_PATH=hailo_env
if not exist "%VENV_PATH%" (
    echo [INFO] Creating virtual environment: %VENV_PATH%
    python -m venv %VENV_PATH%
) else (
    echo [INFO] Using existing virtual environment: %VENV_PATH%
)

REM Activate virtual environment
call %VENV_PATH%\Scripts\activate.bat

echo [INFO] Installing dependencies...

REM Upgrade pip
pip install --upgrade pip

REM CRITICAL: Install onnx with pinned version FIRST
REM hailo_sdk_client requires onnx<=1.16.x (uses onnx.mapping module removed in 1.17+)
echo [INFO] Installing onnx==1.16.0 (pinned for hailo_sdk_client compatibility)
pip install onnx==1.16.0

REM Install protobuf with pinned version
REM hailo_sdk_client requires protobuf 3.x
echo [INFO] Installing protobuf==3.20.3 (pinned for hailo_sdk_client compatibility)
pip install protobuf==3.20.3

REM Install remaining dependencies
echo [INFO] Installing remaining dependencies...
pip install -r requirements.txt

echo.
echo === Verifying Installation ===
python -c "import onnx; print(f'onnx: {onnx.__version__}')"
python -c "import google.protobuf; print(f'protobuf: {google.protobuf.__version__}')"

REM Check hailo_sdk_client
python -c "import hailo_sdk_client" 2>nul
if errorlevel 1 (
    echo hailo_sdk_client: Not installed (install separately from Hailo)
) else (
    echo hailo_sdk_client: OK
)

echo.
echo === Installation Complete ===
echo To activate the environment: %VENV_PATH%\Scripts\activate.bat
echo To run the application: python main.py
