"""
Settings Dialog Controller
"""

import os
import yaml
from PyQt5.QtWidgets import QDialog, QFileDialog, QMessageBox
from PyQt5 import uic
from utils.logger import setup_logger

logger = setup_logger(__name__)


class SettingsDialog(QDialog):
    """Settings dialog."""

    def __init__(self, parent, base_path: str):
        super().__init__(parent)
        self.base_path = base_path
        self.config_path = os.path.join(base_path, 'config.yaml')

        # Load UI
        ui_path = os.path.join(base_path, 'ui', 'dialogs', 'settings.ui')
        uic.loadUi(ui_path, self)

        # Connect browse buttons
        self.btnBrowseModels.clicked.connect(lambda: self._browse_dir(self.editModelsDir))
        self.btnBrowseCalib.clicked.connect(lambda: self._browse_dir(self.editCalibDir))
        self.btnBrowseOutput.clicked.connect(lambda: self._browse_dir(self.editOutputDir))

        # Load current settings
        self._load_settings()

        # Connect save button
        self.btnSave.clicked.connect(self._save_settings)

    def _validate_path(self, path: str, field_name: str) -> tuple:
        """
        Validate a path for security and correctness.
        Returns: (is_valid, error_message)
        """
        if not path or not path.strip():
            return True, ""  # Empty is OK (optional)

        path = path.strip()

        # Check for path traversal
        if '..' in path:
            return False, f"{field_name}: Path traversal (..) not allowed"

        # Check for suspicious characters
        suspicious = ['|', '&', ';', '$', '`', '>', '<', '\n', '\r']
        for char in suspicious:
            if char in path:
                return False, f"{field_name}: Invalid character '{char}' in path"

        # Normalize and check if path is within reasonable bounds
        try:
            normalized = os.path.normpath(path)
            # Check if path tries to escape (on Unix, starts with multiple /)
            if normalized.startswith('//'):
                return False, f"{field_name}: Invalid path format"
        except Exception:
            return False, f"{field_name}: Invalid path"

        return True, ""

    def _validate_all_paths(self) -> tuple:
        """Validate all path inputs in the settings dialog."""
        errors = []

        # Get all path fields
        path_fields = [
            (self.editModelsDir, "Models Directory"),
            (self.editCalibDir, "Calibration Directory"),
            (self.editOutputDir, "Output Directory"),
        ]

        for widget, name in path_fields:
            is_valid, error = self._validate_path(widget.text(), name)
            if not is_valid:
                errors.append(error)

        return len(errors) == 0, errors

    def _browse_dir(self, line_edit):
        """Browse for directory."""
        current = line_edit.text()
        if not os.path.isabs(current):
            current = os.path.join(self.base_path, current)

        path = QFileDialog.getExistingDirectory(self, "Select Directory", current)
        if path:
            # Make relative if inside base path
            if path.startswith(self.base_path):
                path = os.path.relpath(path, self.base_path)
            line_edit.setText(path)

    def _load_settings(self):
        """Load settings from config file."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f) or {}

                # Paths
                paths = config.get('paths', {})
                self.editModelsDir.setText(paths.get('models', 'data/models'))
                self.editCalibDir.setText(paths.get('calibration', 'data/calibration/images'))
                self.editOutputDir.setText(paths.get('output', 'data/output'))

                # Inference settings
                inference = config.get('inference', {})
                self.spinConfThreshold.setValue(inference.get('confidence_threshold', 0.5))
                self.spinIouThreshold.setValue(inference.get('iou_threshold', 0.45))
                self.spinMaxDetections.setValue(inference.get('max_detections', 100))

                # Display settings
                display = config.get('display', {})
                self.chkShowBboxes.setChecked(display.get('show_bboxes', True))
                self.chkShowLabels.setChecked(display.get('show_labels', True))
                self.chkShowConfidence.setChecked(display.get('show_confidence', True))
                self.chkShowFps.setChecked(display.get('show_fps', True))

            except Exception as e:
                logger.error(f"Error loading settings: {e}")

    def _save_settings(self):
        """Save settings to config file."""
        # Validate all paths before saving
        is_valid, errors = self._validate_all_paths()
        if not is_valid:
            QMessageBox.warning(
                self,
                "Invalid Settings",
                "Please fix the following issues:\n\n" + "\n".join(errors)
            )
            return

        # Sanitize paths before saving
        models_path = self.editModelsDir.text().strip()
        calib_path = self.editCalibDir.text().strip()
        output_path = self.editOutputDir.text().strip()

        config = {
            'paths': {
                'models': models_path,
                'calibration': calib_path,
                'output': output_path,
            },
            'inference': {
                'confidence_threshold': self.spinConfThreshold.value(),
                'iou_threshold': self.spinIouThreshold.value(),
                'max_detections': self.spinMaxDetections.value(),
            },
            'display': {
                'show_bboxes': self.chkShowBboxes.isChecked(),
                'show_labels': self.chkShowLabels.isChecked(),
                'show_confidence': self.chkShowConfidence.isChecked(),
                'show_fps': self.chkShowFps.isChecked(),
            }
        }

        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            logger.info("Settings saved successfully")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save settings: {e}")
            logger.error(f"Failed to save settings: {e}")
