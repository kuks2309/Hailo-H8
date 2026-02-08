"""
Main Application Window
"""

import os
from PyQt5.QtWidgets import QMainWindow, QMessageBox
from PyQt5.QtCore import QFile, QIODevice
from PyQt5 import uic

from __version__ import __version__
from views.device_tab import DeviceTabController
from views.convert_tab import ConvertTabController
from views.inference_tab import InferenceTabController
from views.monitor_tab import MonitorTabController


class HailoApp(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()

        # Get base path
        self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Load UI
        ui_path = os.path.join(self.base_path, 'ui', 'main_window.ui')
        uic.loadUi(ui_path, self)

        # Initialize tab controllers
        self._init_tabs()

        # Connect menu actions
        self._connect_actions()

        # Set initial status
        self.statusbar.showMessage("Ready")

    def _init_tabs(self):
        """Initialize tab controllers."""
        # Device tab
        self.device_controller = DeviceTabController(
            self.tab_device, self.base_path
        )

        # Convert tab
        self.convert_controller = ConvertTabController(
            self.tab_convert, self.base_path
        )

        # Inference tab
        self.inference_controller = InferenceTabController(
            self.tab_inference, self.base_path
        )

        # Monitor tab
        self.monitor_controller = MonitorTabController(
            self.tab_monitor, self.base_path
        )

    def _connect_actions(self):
        """Connect menu actions to handlers."""
        # File menu
        self.actionExit.triggered.connect(self.close)

        # Device menu
        self.actionConnect.triggered.connect(self.device_controller.connect_device)
        self.actionDisconnect.triggered.connect(self.device_controller.disconnect_device)
        self.actionRefresh.triggered.connect(self.device_controller.refresh_status)

        # Model menu
        self.actionLoadModel.triggered.connect(self.inference_controller.load_model)
        self.actionUnloadModel.triggered.connect(self.inference_controller.unload_model)

        # Tools menu
        self.actionSettings.triggered.connect(self._show_settings)

        # Help menu
        self.actionAbout.triggered.connect(self._show_about)

    def _show_settings(self):
        """Show settings dialog."""
        from views.settings_dialog import SettingsDialog
        dialog = SettingsDialog(self, self.base_path)
        if dialog.exec_():
            self.statusbar.showMessage("Settings saved", 3000)

    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Hailo-H8 Control Panel",
            "<h3>Hailo-H8 Control Panel</h3>"
            f"<p>Version {__version__}</p>"
            "<p>A Qt5 application for managing Hailo-8 AI accelerator.</p>"
            "<p>Features:</p>"
            "<ul>"
            "<li>Device monitoring</li>"
            "<li>Model conversion (PT → ONNX → HEF)</li>"
            "<li>Real-time inference</li>"
            "<li>Performance monitoring</li>"
            "</ul>"
        )

    def closeEvent(self, event):
        """Handle window close event."""
        # Stop any running inference
        if hasattr(self, 'inference_controller'):
            self.inference_controller.stop_inference()

        # Stop monitor updates
        if hasattr(self, 'monitor_controller'):
            self.monitor_controller.stop_monitoring()

        event.accept()
