"""
Main Window for Hailo-Compiler-UI.
Loads UI from .ui file and connects signals.
"""

import os
from datetime import datetime
from PyQt5.QtWidgets import (
    QMainWindow, QFileDialog, QMessageBox, QApplication
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QTextCharFormat, QColor, QTextCursor
from PyQt5 import uic

from .styles import get_theme, get_color
from .converter_panel import detect_yolo_project_from_pt
from ..core.environment import check_environment, can_compile_hef
from ..workers.convert_worker import ConvertWorker


class MainWindow(QMainWindow):
    """
    Main application window.
    Loads UI from .ui file.
    """

    def __init__(self, base_path: str = ""):
        super().__init__()
        self.base_path = base_path
        self.data_path = os.path.join(base_path, 'data') if base_path else ""
        self.worker = None
        self._detected_project = {}
        self.use_modelzoo_mode = False

        self._load_ui()
        self._setup_connections()
        self._apply_theme()
        self._check_environment()

    def _load_ui(self):
        """Load UI from .ui file."""
        ui_path = os.path.join(os.path.dirname(__file__), '..', '..', 'ui', 'main_window.ui')
        uic.loadUi(ui_path, self)

        # Set opset combo default to 17
        self.opsetCombo.setCurrentText('17')

    def _apply_theme(self):
        """Apply theme stylesheet."""
        self.setStyleSheet(get_theme())

        # Apply accent style to action buttons
        self.convertBtn.setProperty('accent', True)
        self.compileBtn.setProperty('accent', True)

        # Apply browse style to browse buttons
        for btn in [self.browsePtBtn, self.browseOnnxOutputBtn, self.browseOnnxBtn,
                    self.browseCalibBtn, self.browseHefBtn, self.browseProjectBtn]:
            btn.setProperty('browse', True)

        # Set arrow color
        self.arrowLabel.setStyleSheet(f"color: {get_color('accent')};")

        # Set header style
        self.headerLabel.setProperty('heading', True)

        # Refresh styles
        self.style().unpolish(self)
        self.style().polish(self)

    def _setup_connections(self):
        """Setup signal connections."""
        # Environment
        self.refreshBtn.clicked.connect(self._refresh_environment)

        # Project
        self.browseProjectBtn.clicked.connect(self._browse_project)

        # PT to ONNX
        self.browsePtBtn.clicked.connect(self._browse_pt)
        self.browseOnnxOutputBtn.clicked.connect(self._browse_onnx_output)
        self.ptPath.textChanged.connect(self._auto_fill_onnx)
        self.convertBtn.clicked.connect(self._on_pt_to_onnx)

        # ONNX to HEF
        self.browseOnnxBtn.clicked.connect(self._browse_onnx)
        self.browseCalibBtn.clicked.connect(self._browse_calib)
        self.browseHefBtn.clicked.connect(self._browse_hef)
        self.onnxPath.textChanged.connect(self._auto_fill_hef)
        self.onnxPath.textChanged.connect(self._check_onnx_compatibility)
        self.useModelZooBtn.clicked.connect(self._toggle_model_zoo_mode)
        self.compileBtn.clicked.connect(self._on_onnx_to_hef)

        # Log
        self.exportLogBtn.clicked.connect(self._export_log)
        self.clearLogBtn.clicked.connect(self._clear_log)

    def _check_environment(self):
        """Check environment and update UI."""
        self._refresh_environment()

        # Disable HEF compilation if SDK not available
        if not can_compile_hef():
            self.compileBtn.setEnabled(False)
            self.compileBtn.setToolTip("hailo_sdk_client not installed")
            self._log_warning("hailo_sdk_client not installed - HEF compilation disabled")

    def _refresh_environment(self):
        """Refresh environment status."""
        env_result = check_environment()

        labels = {
            'torch': self.torchLabel,
            'numpy': self.numpyLabel,
            'PIL': self.pilLabel,
            'ultralytics': self.ultralyticsLabel,
            'hailo_sdk_client': self.hailoLabel,
            'onnx': self.onnxLabel,
        }

        for pkg, label in labels.items():
            if pkg in env_result['packages']:
                info = env_result['packages'][pkg]
                if info.installed:
                    label.setText(f"[✓] {info.name}")
                    label.setProperty('status', 'success')
                    label.setToolTip(f"Version: {info.version}")
                else:
                    label.setText(f"[✗] {info.name}")
                    if info.required:
                        label.setProperty('status', 'error')
                    else:
                        label.setProperty('status', 'warning')
                    label.setToolTip(f"Install: {info.install_cmd}")

                label.style().unpolish(label)
                label.style().polish(label)

        if not env_result['all_required_ok']:
            missing = ", ".join(env_result['missing_required'])
            QMessageBox.warning(
                self,
                "Missing Required Packages",
                f"The following required packages are missing:\n\n{missing}\n\n"
                "Some features may not work correctly."
            )

    def _browse_project(self):
        """Browse for project folder."""
        path = QFileDialog.getExistingDirectory(
            self, "Select Project Folder",
            os.path.expanduser("~")
        )
        if path:
            self.projectPath.setText(path)
            self.base_path = path
            self.data_path = path
            self._log_info(f"Project folder set: {path}")
            self.statusBar().showMessage(f"Project: {os.path.basename(path)}")

            # Try to find calibration folder
            for folder in ['train/images', 'valid/images', 'test/images']:
                calib_path = os.path.join(path, folder)
                if os.path.isdir(calib_path):
                    self.calibPath.setText(calib_path)
                    break

    def _browse_pt(self):
        """Browse for PT file."""
        start_dir = os.path.join(self.data_path, 'models', 'pt') if self.data_path else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select PyTorch Model", start_dir,
            "PyTorch Models (*.pt *.pth);;All Files (*)"
        )
        if path:
            self.ptPath.setText(path)

    def _browse_onnx_output(self):
        """Browse for ONNX output path."""
        start_dir = os.path.join(self.data_path, 'models', 'onnx') if self.data_path else ""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save ONNX Model", start_dir,
            "ONNX Models (*.onnx)"
        )
        if path:
            if not path.endswith('.onnx'):
                path += '.onnx'
            self.onnxOutputPath.setText(path)

    def _auto_fill_onnx(self, pt_path: str):
        """Auto-fill ONNX path based on PT path."""
        if pt_path and (pt_path.endswith('.pt') or pt_path.endswith('.pth')):
            # Detect YOLO project structure
            self._detected_project = detect_yolo_project_from_pt(pt_path)

            base_name = os.path.basename(pt_path)
            base_name = base_name.replace('.pth', '.onnx').replace('.pt', '.onnx')

            # Determine onnx output directory
            if self._detected_project.get('project_root'):
                onnx_dir = os.path.join(self._detected_project['project_root'], 'models', 'onnx')
                os.makedirs(onnx_dir, exist_ok=True)

                # Apply project settings to HEF panel
                if self._detected_project.get('train_images'):
                    self.calibPath.setText(self._detected_project['train_images'])
                if self._detected_project.get('nc'):
                    self.numClasses.setValue(self._detected_project['nc'])

                self._log_info(f"Detected YOLO project: {self._detected_project['project_root']}")
            elif self.data_path:
                onnx_dir = os.path.join(self.data_path, 'models', 'onnx')
                os.makedirs(onnx_dir, exist_ok=True)
            else:
                onnx_dir = os.path.dirname(pt_path)

            self.onnxOutputPath.setText(os.path.join(onnx_dir, base_name))

        # Auto-set opset and model type based on deep PT analysis
        # (works regardless of project detection - uses actual model content)
        yolo_version = self._detected_project.get('yolo_version')
        model_type = self._detected_project.get('model_type', 'detect')
        recommended_opset = self._detected_project.get('recommended_opset', 17)

        if yolo_version:
            # Set recommended opset
            self.opsetCombo.setCurrentText(str(recommended_opset))

            # Set model type in combo
            idx = self.modelTypeCombo.findText(model_type)
            if idx >= 0:
                self.modelTypeCombo.setCurrentIndex(idx)

            self._log_info(f"YOLO{yolo_version} {model_type} detected → opset={recommended_opset}")

    def _browse_onnx(self):
        """Browse for ONNX file."""
        start_dir = os.path.join(self.data_path, 'models', 'onnx') if self.data_path else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select ONNX Model", start_dir,
            "ONNX Models (*.onnx);;All Files (*)"
        )
        if path:
            self.onnxPath.setText(path)

    def _browse_calib(self):
        """Browse for calibration directory."""
        start_dir = self.data_path if self.data_path else ""
        path = QFileDialog.getExistingDirectory(
            self, "Select Calibration Images Directory", start_dir
        )
        if path:
            self.calibPath.setText(path)

    def _browse_hef(self):
        """Browse for HEF output path."""
        start_dir = os.path.join(self.data_path, 'models', 'hef') if self.data_path else ""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save HEF Model", start_dir,
            "Hailo Models (*.hef)"
        )
        if path:
            if not path.endswith('.hef'):
                path += '.hef'
            self.hefPath.setText(path)

    def _auto_fill_hef(self, onnx_path: str):
        """Auto-fill HEF path based on ONNX path."""
        if onnx_path and onnx_path.endswith('.onnx'):
            base_name = os.path.basename(onnx_path).replace('.onnx', '.hef')
            onnx_dir = os.path.dirname(onnx_path)

            if onnx_dir.endswith('/onnx') or onnx_dir.endswith('\\onnx'):
                models_dir = os.path.dirname(onnx_dir)
                hef_dir = os.path.join(models_dir, 'hef')
            elif os.path.basename(onnx_dir) == 'models':
                hef_dir = os.path.join(onnx_dir, 'hef')
            elif self.data_path:
                hef_dir = os.path.join(self.data_path, 'models', 'hef')
            else:
                hef_dir = onnx_dir

            os.makedirs(hef_dir, exist_ok=True)
            self.hefPath.setText(os.path.join(hef_dir, base_name))

    def _check_onnx_compatibility(self, onnx_path: str):
        """Check ONNX compatibility."""
        if not onnx_path or not os.path.exists(onnx_path):
            self.compatStatus.setText("")
            self.useModelZooBtn.setVisible(False)
            self.compileBtn.setEnabled(False)
            return

        try:
            from ..core.converter import validate_onnx_hailo_compatibility
            compat = validate_onnx_hailo_compatibility(onnx_path)

            if compat['compatible']:
                style = compat['naming_style']
                version = compat.get('yolo_version', 'unknown')
                heads = len(compat.get('detection_heads', []))
                self.compatStatus.setText(f"✓ {style} ({version}) - {heads} detection heads")
                self.compatStatus.setStyleSheet("color: #4CAF50;")
                self.useModelZooBtn.setVisible(False)
                self.compileBtn.setEnabled(True)
            else:
                error_msg = compat['errors'][0] if compat['errors'] else "Unknown error"
                self.compatStatus.setText(f"✗ INCOMPATIBLE: {error_msg}")
                self.compatStatus.setStyleSheet("color: #F44336;")
                self.useModelZooBtn.setVisible(True)
                self.compileBtn.setEnabled(False)
        except Exception as e:
            self.compatStatus.setText(f"⚠ Error: {str(e)}")
            self.compatStatus.setStyleSheet("color: #FF9800;")
            self.useModelZooBtn.setVisible(False)
            self.compileBtn.setEnabled(True)

    def _toggle_model_zoo_mode(self):
        """Toggle Model Zoo mode."""
        if self.use_modelzoo_mode:
            # Switch back to custom mode
            self.onnxPath.setPlaceholderText("Select ONNX model file...")
            self.onnxPath.setEnabled(True)
            self.browseOnnxBtn.setEnabled(True)
            self.use_modelzoo_mode = False
            self.compatStatus.setText("")
            self.useModelZooBtn.setText("Use Model Zoo Standard")
            self.useModelZooBtn.setVisible(False)
            self.compileBtn.setEnabled(False)
        else:
            # Switch to Model Zoo mode
            self.onnxPath.setText("")
            self.onnxPath.setPlaceholderText("Using Hailo Model Zoo standard model...")
            self.onnxPath.setEnabled(False)
            self.browseOnnxBtn.setEnabled(False)
            self.use_modelzoo_mode = True
            self.compatStatus.setText("⚠ Model Zoo mode: Using Hailo standard model")
            self.compatStatus.setStyleSheet("color: #FF9800; font-weight: bold;")
            self.useModelZooBtn.setText("← Use Custom ONNX Instead")
            self.compileBtn.setEnabled(True)

    def _on_pt_to_onnx(self):
        """Handle PT to ONNX conversion."""
        pt_path = self.ptPath.text()
        onnx_path = self.onnxOutputPath.text()

        if not pt_path:
            QMessageBox.warning(self, "Warning", "Please select a PT model file.")
            return

        if not os.path.exists(pt_path):
            QMessageBox.warning(self, "Warning", f"PT file not found:\n{pt_path}")
            return

        if not onnx_path:
            QMessageBox.warning(self, "Warning", "Please specify output ONNX path.")
            return

        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

        self._set_converting(True)
        self._clear_log()
        self._log_info("Starting PT to ONNX conversion...")

        params = {
            'pt_path': pt_path,
            'onnx_path': onnx_path,
            'input_size': (self.heightSpin.value(), self.widthSpin.value()),
            'batch_size': 1,
            'opset': int(self.opsetCombo.currentText()),
            'export_method': self.exportMethodCombo.currentText()
        }

        self.worker = ConvertWorker('pt_to_onnx', params)
        self.worker.progress.connect(self._set_progress)
        self.worker.log.connect(self._log_info)
        self.worker.error.connect(self._on_error)
        self.worker.finished.connect(lambda ok, msg: self._on_conversion_done(ok, msg, 'pt_to_onnx'))
        self.worker.start()

    def _on_onnx_to_hef(self):
        """Handle ONNX to HEF compilation."""
        onnx_path = "" if self.use_modelzoo_mode else self.onnxPath.text()
        calib_dir = self.calibPath.text()
        hef_path = self.hefPath.text()

        if not self.use_modelzoo_mode:
            if not onnx_path:
                QMessageBox.warning(self, "Warning", "Please select an ONNX model file.")
                return
            if not os.path.exists(onnx_path):
                QMessageBox.warning(self, "Warning", f"ONNX file not found:\n{onnx_path}")
                return

        if not calib_dir:
            QMessageBox.warning(self, "Warning", "Please select calibration images directory.")
            return

        if not os.path.exists(calib_dir):
            QMessageBox.warning(self, "Warning", f"Calibration directory not found:\n{calib_dir}")
            return

        if not hef_path:
            QMessageBox.warning(self, "Warning", "Please specify output HEF path.")
            return

        os.makedirs(os.path.dirname(hef_path), exist_ok=True)

        self._set_converting(True)
        self._clear_log()
        self._log_info("Starting ONNX to HEF compilation...")

        params = {
            'onnx_path': onnx_path,
            'hef_path': hef_path,
            'calib_dir': calib_dir,
            'target': self.targetCombo.currentText(),
            'calib_size': (self.calibHeight.value(), self.calibWidth.value()),
            'model_type': self.modelTypeCombo.currentText(),
            'model_name': self.modelNameCombo.currentText(),
            'num_classes': self.numClasses.value()
        }

        self.worker = ConvertWorker('onnx_to_hef', params)
        self.worker.progress.connect(self._set_progress)
        self.worker.log.connect(self._log_info)
        self.worker.error.connect(self._on_error)
        self.worker.finished.connect(lambda ok, msg: self._on_conversion_done(ok, msg, 'onnx_to_hef'))
        self.worker.start()

    def _on_error(self, error_type: str, message: str):
        """Handle conversion error."""
        self._log_error(f"[{error_type}] {message}")
        self.progressBar.setProperty('error', True)
        self.progressBar.style().unpolish(self.progressBar)
        self.progressBar.style().polish(self.progressBar)

    def _on_conversion_done(self, success: bool, result: str, task_type: str):
        """Handle conversion completion."""
        self._set_converting(False)

        if success:
            self._log_success(f"Completed: {result}")
            self.statusBar().showMessage("Conversion successful")

            if task_type == 'pt_to_onnx':
                self.onnxPath.setText(result)
                QMessageBox.information(
                    self, "Success",
                    f"ONNX conversion completed!\n\nOutput: {result}"
                )
            else:
                QMessageBox.information(
                    self, "Success",
                    f"HEF compilation completed!\n\nOutput: {result}"
                )
        else:
            self.statusBar().showMessage("Conversion failed")
            QMessageBox.critical(
                self, "Error",
                f"Conversion failed:\n\n{result}"
            )

    def _set_converting(self, converting: bool):
        """Enable/disable UI during conversion."""
        widgets = [
            self.ptPath, self.onnxOutputPath, self.browsePtBtn, self.browseOnnxOutputBtn,
            self.widthSpin, self.heightSpin, self.opsetCombo, self.exportMethodCombo,
            self.convertBtn, self.onnxPath, self.calibPath, self.hefPath,
            self.browseOnnxBtn, self.browseCalibBtn, self.browseHefBtn,
            self.targetCombo, self.calibWidth, self.calibHeight,
            self.modelTypeCombo, self.modelNameCombo, self.numClasses,
            self.compileBtn
        ]

        for w in widgets:
            w.setEnabled(not converting)

        if converting:
            self.statusBar().showMessage("Converting...")
        else:
            if not can_compile_hef():
                self.compileBtn.setEnabled(False)

    def _set_progress(self, value: int):
        """Set progress bar value."""
        self.progressBar.setValue(value)
        if 0 < value < 100:
            self.progressBar.setProperty('error', False)
            self.progressBar.style().unpolish(self.progressBar)
            self.progressBar.style().polish(self.progressBar)

    def _log(self, message: str, level: str = 'info'):
        """Add a log message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {message}"

        cursor = self.logText.textCursor()
        cursor.movePosition(QTextCursor.End)

        fmt = QTextCharFormat()
        if level == 'success':
            fmt.setForeground(QColor(get_color('success')))
        elif level == 'warning':
            fmt.setForeground(QColor(get_color('warning')))
        elif level == 'error':
            fmt.setForeground(QColor(get_color('error')))
        else:
            fmt.setForeground(QColor(get_color('text')))

        cursor.insertText(formatted + "\n", fmt)
        self.logText.setTextCursor(cursor)
        self.logText.ensureCursorVisible()

    def _log_info(self, message: str):
        self._log(message, 'info')

    def _log_success(self, message: str):
        self._log(message, 'success')

    def _log_warning(self, message: str):
        self._log(message, 'warning')

    def _log_error(self, message: str):
        self._log(message, 'error')

    def _clear_log(self):
        """Clear log and reset progress."""
        self.logText.clear()
        self.progressBar.setValue(0)
        self.progressBar.setProperty('error', False)
        self.progressBar.style().unpolish(self.progressBar)
        self.progressBar.style().polish(self.progressBar)

    def _export_log(self):
        """Export log content to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"hailo_compile_log_{timestamp}.txt"

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Log",
            default_name,
            "Text Files (*.txt);;All Files (*)"
        )

        if filepath:
            try:
                with open(filepath, 'w') as f:
                    f.write(self.logText.toPlainText())
                self._log_success(f"Log exported to: {filepath}")
            except Exception as e:
                QMessageBox.warning(self, "Export Failed", f"Could not export log: {e}")

    def closeEvent(self, event):
        """Handle window close."""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self, "Confirm Exit",
                "A conversion is in progress. Are you sure you want to exit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                event.ignore()
                return

            self.worker.terminate()
            self.worker.wait()

        event.accept()
