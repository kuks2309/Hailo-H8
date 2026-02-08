#!/bin/bash
# Run Hailo-H8 Control Panel

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Activate virtual environment if it exists
if [ -d "../hailo_env" ]; then
    source ../hailo_env/bin/activate
fi

# Run the application
python main.py "$@"
