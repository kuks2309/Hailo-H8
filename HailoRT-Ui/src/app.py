"""
Main Application Window
"""

import os
from PyQt5.QtWidgets import QMainWindow, QMessageBox
from PyQt5.QtCore import QFile, QIODevice
from PyQt5 import uic

from __version__ import __version__
from views.project_tab import ProjectTabController
from views.device_tab import DeviceTabController
from views.convert_tab import ConvertTabController
from views.inference_tab import InferenceTabController
from views.monitor_tab import MonitorTabController
from utils.styles import get_theme
from utils.environment import check_environment, get_missing_packages_message
from utils.logger import setup_logger

logger = setup_logger(__name__)


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

        # Apply theme
        self.setStyleSheet(get_theme())

        # Check environment at startup
        self._check_environment()

    def _init_tabs(self):
        """Initialize tab controllers."""
        # Project tab (first - provides paths to other tabs)
        self.project_controller = ProjectTabController(
            self.tab_project, self.base_path
        )

        # Device tab
        self.device_controller = DeviceTabController(
            self.tab_device, self.base_path
        )

        # Convert tab
        self.convert_controller = ConvertTabController(
            self.tab_convert, self.base_path
        )

        # Inference tab (shares device_controller for device access)
        self.inference_controller = InferenceTabController(
            self.tab_inference, self.base_path,
            device_controller=self.device_controller
        )

        # Monitor tab (shares device_controller for real device stats)
        self.monitor_controller = MonitorTabController(
            self.tab_monitor, self.base_path,
            device_controller=self.device_controller
        )

        # Wire Project tab -> other tabs
        self.project_controller.project_changed.connect(
            self._on_project_changed
        )

    def _on_project_changed(self, project_info: dict):
        """Handle project folder change - propagate to other tabs."""
        logger.info(f"Project changed: {project_info.get('summary', '')}")

        # Apply to Convert tab
        self.convert_controller.set_project_info(project_info)

        # Apply to Inference tab (HEF path)
        if hasattr(self.inference_controller, 'set_project_info'):
            self.inference_controller.set_project_info(project_info)

        self.statusbar.showMessage(
            f"Project: {os.path.basename(project_info.get('base', ''))} - "
            f"{project_info.get('summary', '')}",
            5000
        )

    def _connect_actions(self):
        """Connect menu actions to handlers."""
        # File menu
        self.actionOpenFolder.triggered.connect(self._open_folder)
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

    def _check_environment(self):
        """Check environment and warn about missing packages."""
        env_result = check_environment()

        missing_required = env_result['missing_required']
        missing_optional = env_result['missing_optional']

        if missing_required:
            message = get_missing_packages_message(env_result)
            logger.warning(f"Missing required packages: {missing_required}")
            QMessageBox.warning(
                self,
                "Missing Required Packages",
                f"Some required packages are not installed:\n\n{message}\n\n"
                "Some features may not work correctly."
            )
        elif missing_optional:
            logger.info(f"Missing optional packages: {missing_optional}")
            self.statusbar.showMessage(
                f"Ready (some optional packages missing: {', '.join(missing_optional)})",
                5000
            )
        else:
            logger.info("All packages available")

    def _open_folder(self):
        """Open folder via Project tab and switch to it."""
        self.tabWidget.setCurrentWidget(self.tab_project)
        self.project_controller._browse_folder()

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
            "<hr>"
            "<p><b>Author:</b> Prof. Kuk Won Ko with Claude Code</p>"
            "<p><b>License:</b> Free for personal and non-commercial use.<br>"
            "Commercial and enterprise use requires a paid license.</p>"
        )

    def closeEvent(self, event):
        """Handle window close event."""
        # Stop conversion worker if running
        if hasattr(self, 'convert_controller') and self.convert_controller.worker:
            worker = self.convert_controller.worker
            if worker.isRunning():
                reply = QMessageBox.question(
                    self, "Confirm Exit",
                    "A conversion is in progress. Exit anyway?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No
                )
                if reply == QMessageBox.No:
                    event.ignore()
                    return
                worker.cancel()
                worker.wait(10000)
                if worker.isRunning():
                    logger.warning("ConvertWorker did not stop gracefully, forcing terminate")
                    worker.terminate()
                    worker.wait(3000)

        # Stop any running inference
        if hasattr(self, 'inference_controller'):
            self.inference_controller.stop_inference()

        # Stop monitor updates
        if hasattr(self, 'monitor_controller'):
            self.monitor_controller.stop_monitoring()

        event.accept()
