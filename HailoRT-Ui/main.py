#!/usr/bin/env python3
"""
Hailo-H8 Control Panel
Main entry point for the Qt5 application.
"""

import sys
import os

# CRITICAL: Disable GPU subprocess check BEFORE any Hailo imports
# Key fix: HAILO_DISABLE_MO_SUB_PROCESS=1 skips the problematic subprocess entirely
# See: hailo_model_optimization/acceleras/utils/tf_utils.py line 65
os.environ['HAILO_DISABLE_MO_SUB_PROCESS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import glob
import shutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from __version__ import __version__
from app import HailoApp


def collect_logs():
    """Move scattered *.log files to logs/ folder at startup."""
    app_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(app_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    # Scan app root and src/ for SDK-generated log files
    scan_dirs = [app_dir, os.path.join(app_dir, 'src')]

    moved = 0
    for scan_dir in scan_dirs:
        if not os.path.isdir(scan_dir):
            continue
        for log_file in glob.glob(os.path.join(scan_dir, '*.log')):
            filename = os.path.basename(log_file)
            dst = os.path.join(logs_dir, filename)
            try:
                shutil.move(log_file, dst)
                moved += 1
            except OSError:
                pass

    if moved:
        print(f"[Startup] {moved} log file(s) moved to logs/")


def main():
    """Main entry point."""
    # Collect scattered log files before app starts
    collect_logs()

    # Enable High DPI scaling
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)

    # Set application info
    app.setApplicationName("Hailo-H8 Control Panel")
    app.setApplicationVersion(__version__)
    app.setOrganizationName("HailoRT-UI")

    # Set default font
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    # Create and show main window
    window = HailoApp()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
