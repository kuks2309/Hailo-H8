#!/usr/bin/env python3
"""
Hailo-Compiler-UI
Main entry point for the model compilation application.
"""

import sys
import os

# CRITICAL: Disable GPU subprocess check BEFORE any imports
# Key fix: HAILO_DISABLE_MO_SUB_PROCESS=1 skips the problematic subprocess entirely
# See: hailo_model_optimization/acceleras/utils/tf_utils.py line 65
os.environ['HAILO_DISABLE_MO_SUB_PROCESS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from src.core.environment import check_environment, get_missing_packages_message
from src.ui.main_window import MainWindow


def check_required_packages() -> bool:
    """
    Check if required packages are installed.

    Returns:
        True if all required packages are available
    """
    env_result = check_environment()

    if not env_result['all_required_ok']:
        # Create a minimal app to show error dialog
        app = QApplication(sys.argv)

        message = get_missing_packages_message(env_result)
        QMessageBox.critical(
            None,
            "Missing Required Packages",
            f"Cannot start Hailo Compiler.\n\n{message}\n\n"
            "Please install the required packages and try again."
        )
        return False

    return True


def main():
    """Main entry point."""
    # Get base path
    base_path = os.path.dirname(os.path.abspath(__file__))

    # Check required packages first
    if not check_required_packages():
        sys.exit(1)

    # Enable High DPI scaling
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)

    # Set application info
    app.setApplicationName("Hailo Compiler")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("Hailo-Compiler-UI")

    # Set default font (larger for better visibility)
    font = QFont("Segoe UI", 12)
    app.setFont(font)

    # Create and show main window
    window = MainWindow(base_path)
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
