#!/usr/bin/env python3
"""
Hailo-H8 Control Panel
Main entry point for the Qt5 application.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from __version__ import __version__
from app import HailoApp


def main():
    """Main entry point."""
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
