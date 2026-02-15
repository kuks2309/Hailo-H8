"""
Convert Tab Controller
Manages model conversion from PT to ONNX to HEF.
"""

from __future__ import annotations

import os
from typing import Optional, Dict, Any, List
from datetime import datetime
from PyQt5.QtWidgets import QWidget, QFileDialog, QMessageBox
from PyQt5 import uic

from utils.logger import setup_logger
from utils.onnx_utils import validate_onnx_hailo_compatibility, ONNX_AVAILABLE
from utils.model_detection import detect_yolo_from_pt
from workers.convert_worker import ConvertWorker

logger = setup_logger(__name__)


class ConvertTabController:
    """Controller for model conversion tab."""

    def __init__(self, tab_widget: QWidget, base_path: str) -> None:
        self.tab: QWidget = tab_widget
        self.base_path: str = base_path
        self.worker: Optional[ConvertWorker] = None
        self.model_zoo_mode: bool = False
        self._project_info: Optional[Dict] = None

        # Load UI into tab
        ui_path = os.path.join(base_path, 'ui', 'tabs', 'convert_tab.ui')
        uic.loadUi(ui_path, self.tab)

        # Set default paths
        self._set_default_paths()

        # Connect signals
        self._connect_signals()

    # --- Public API (called from app.py via project_changed signal) ---

    def set_project_info(self, info: Dict) -> None:
        """
        Apply project info from Project tab.

        Auto-fills calibration, ONNX output, HEF output, and PT file paths.
        """
        self._project_info = info
        if not info.get('valid'):
            return

        # Auto-fill calibration path
        if info.get('calib_found'):
            self.tab.editCalibPath.setText(info['calib_dir'])

        # Auto-fill ONNX output path
        onnx_dir = info.get('onnx_dir', '')
        if onnx_dir:
            os.makedirs(onnx_dir, exist_ok=True)
            current_onnx = self.tab.editOnnxOutPath.text()
            if not current_onnx or 'HailoRT-Ui/data/models/onnx' in current_onnx:
                self.tab.editOnnxOutPath.setText(
                    os.path.join(onnx_dir, 'model.onnx')
                )

        # Auto-fill HEF output path
        hef_dir = info.get('hef_dir', '')
        if hef_dir:
            os.makedirs(hef_dir, exist_ok=True)
            current_hef = self.tab.editHefOutPath.text()
            if not current_hef or 'HailoRT-Ui/data/models/hef' in current_hef:
                self.tab.editHefOutPath.setText(
                    os.path.join(hef_dir, 'model.hef')
                )

        # Auto-select PT file if found
        pt_files = info.get('pt_files', [])
        if pt_files and not self.tab.editPtPath.text():
            best_pt = next(
                (f for f in pt_files if os.path.basename(f) == 'best.pt'),
                pt_files[0]
            )
            self.tab.editPtPath.setText(best_pt)
            self._log(f"[Auto] PT model: {os.path.basename(best_pt)}")

        # Auto-select ONNX file if found (for HEF compilation)
        onnx_files = info.get('onnx_files', [])
        if onnx_files and not self.tab.editOnnxPath.text():
            best_onnx = next(
                (f for f in onnx_files if 'best' in os.path.basename(f).lower()),
                onnx_files[0]
            )
            self.tab.editOnnxPath.setText(best_onnx)
            self._log(f"[Auto] ONNX model: {os.path.basename(best_onnx)}")

        # Log
        if info.get('num_classes'):
            self._log(f"[Auto] Detected classes: {info['num_classes']}")

        self._log(f"[{self._timestamp()}] Project paths applied: {info.get('summary', '')}")

    # --- Private methods ---

    def _set_default_paths(self) -> None:
        """Set default directory paths."""
        data_path = os.path.join(self.base_path, 'data')

        self.tab.editOnnxOutPath.setText(
            os.path.join(data_path, 'models', 'onnx', 'model.onnx')
        )
        self.tab.editCalibPath.setText(
            os.path.join(data_path, 'calibration', 'images')
        )
        self.tab.editHefOutPath.setText(
            os.path.join(data_path, 'models', 'hef', 'model.hef')
        )

    def _connect_signals(self) -> None:
        """Connect UI signals."""
        # Browse buttons
        self.tab.btnBrowsePt.clicked.connect(self._browse_pt_file)
        self.tab.btnBrowseOnnxOut.clicked.connect(self._browse_onnx_output)
        self.tab.btnBrowseOnnx.clicked.connect(self._browse_onnx_file)
        self.tab.btnBrowseCalib.clicked.connect(self._browse_calib_dir)
        self.tab.btnBrowseHefOut.clicked.connect(self._browse_hef_output)

        # Convert buttons
        self.tab.btnConvertOnnx.clicked.connect(self._start_pt_to_onnx)
        self.tab.btnCompileHef.clicked.connect(self._start_onnx_to_hef)

        # Clear log
        self.tab.btnClearLog.clicked.connect(self._clear_log)

        # Auto-fill ONNX path after PT selection
        self.tab.editPtPath.textChanged.connect(self._auto_fill_onnx_path)
        # Auto-fill HEF path after ONNX selection
        self.tab.editOnnxPath.textChanged.connect(self._auto_fill_hef_path)
        # Check ONNX compatibility on file selection
        self.tab.editOnnxPath.textChanged.connect(self._check_onnx_compatibility)
        # Model Zoo toggle (if button exists in UI)
        if hasattr(self.tab, 'btnModelZoo'):
            self.tab.btnModelZoo.clicked.connect(self._toggle_model_zoo)

    def _get_start_dir(self, subdir: str, project_key: str = None) -> str:
        """Get file dialog start directory, preferring project paths."""
        if project_key and self._project_info and self._project_info.get(project_key):
            return self._project_info[project_key]
        if self._project_info and self._project_info.get('base'):
            return self._project_info['base']
        return os.path.join(self.base_path, 'data', subdir)

    def _browse_pt_file(self) -> None:
        """Browse for PyTorch model file."""
        start_dir = self._get_start_dir('models/pt')
        path, _ = QFileDialog.getOpenFileName(
            self.tab,
            "Select PyTorch Model",
            start_dir,
            "PyTorch Models (*.pt *.pth);;All Files (*)"
        )
        if path:
            logger.debug(f"Selected PT file: {path}")
            self.tab.editPtPath.setText(path)

    def _browse_onnx_output(self) -> None:
        """Browse for ONNX output path."""
        start_dir = self._get_start_dir('models/onnx', 'onnx_dir')
        path, _ = QFileDialog.getSaveFileName(
            self.tab,
            "Save ONNX Model",
            start_dir,
            "ONNX Models (*.onnx)"
        )
        if path:
            if not path.endswith('.onnx'):
                path += '.onnx'
            self.tab.editOnnxOutPath.setText(path)

    def _browse_onnx_file(self) -> None:
        """Browse for ONNX model file."""
        start_dir = self._get_start_dir('models/onnx', 'onnx_dir')
        path, _ = QFileDialog.getOpenFileName(
            self.tab,
            "Select ONNX Model",
            start_dir,
            "ONNX Models (*.onnx);;All Files (*)"
        )
        if path:
            logger.debug(f"Selected ONNX file: {path}")
            self.tab.editOnnxPath.setText(path)

    def _browse_calib_dir(self) -> None:
        """Browse for calibration directory."""
        start_dir = self._get_start_dir('calibration/images', 'calib_dir')
        path = QFileDialog.getExistingDirectory(
            self.tab,
            "Select Calibration Images Directory",
            start_dir
        )
        if path:
            self.tab.editCalibPath.setText(path)

    def _browse_hef_output(self) -> None:
        """Browse for HEF output path."""
        start_dir = self._get_start_dir('models/hef', 'hef_dir')
        path, _ = QFileDialog.getSaveFileName(
            self.tab,
            "Save HEF Model",
            start_dir,
            "Hailo Models (*.hef)"
        )
        if path:
            if not path.endswith('.hef'):
                path += '.hef'
            self.tab.editHefOutPath.setText(path)

    def _auto_fill_onnx_path(self, pt_path: str) -> None:
        """Auto-fill paths and settings based on PT file selection."""
        if not pt_path:
            self._reset_detection_ui()
            return

        # Auto-fill ONNX output path
        if pt_path.endswith(('.pt', '.pth')):
            base_name = os.path.basename(pt_path)
            name_no_ext = os.path.splitext(base_name)[0]

            if self._project_info and self._project_info.get('onnx_dir'):
                onnx_dir = self._project_info['onnx_dir']
            else:
                onnx_dir = os.path.join(self.base_path, 'data', 'models', 'onnx')

            os.makedirs(onnx_dir, exist_ok=True)
            self.tab.editOnnxOutPath.setText(
                os.path.join(onnx_dir, f"{name_no_ext}.onnx")
            )

        # Deep PT analysis - detect YOLO version, task, opset
        if os.path.exists(pt_path):
            self._run_model_detection(pt_path)

        # Try to detect YOLO project structure
        project_info = self._detect_yolo_project(pt_path)
        if project_info:
            logger.info(f"Detected YOLO project: {project_info}")

            # Auto-fill calibration path
            if project_info.get('calib_dir'):
                self.tab.editCalibPath.setText(project_info['calib_dir'])
                self._log(f"[Auto] Calibration path: {project_info['calib_dir']}")

            # Log detected info
            if project_info.get('num_classes'):
                self._log(f"[Auto] Detected classes: {project_info['num_classes']}")

    def _auto_fill_hef_path(self, onnx_path: str) -> None:
        """Auto-fill HEF output path based on ONNX path."""
        if not onnx_path:
            return

        if onnx_path.endswith('.onnx'):
            base_name = os.path.basename(onnx_path).replace('.onnx', '.hef')

            if self._project_info and self._project_info.get('hef_dir'):
                hef_dir = self._project_info['hef_dir']
            else:
                hef_dir = os.path.join(self.base_path, 'data', 'models', 'hef')

            os.makedirs(hef_dir, exist_ok=True)
            self.tab.editHefOutPath.setText(os.path.join(hef_dir, base_name))

    def _run_model_detection(self, pt_path: str) -> None:
        """Run deep PT analysis and update detection UI."""
        try:
            detection = detect_yolo_from_pt(pt_path)

            yolo_version = detection.get('yolo_version')
            task_type = detection.get('task_type', 'detect')
            recommended_opset = detection.get('recommended_opset', 11)
            model_name = detection.get('model_name_prefix', '')

            # Update detection UI labels
            if hasattr(self.tab, 'lblYoloVersion'):
                if yolo_version:
                    version_text = f"YOLO{yolo_version}"
                    self.tab.lblYoloVersion.setText(version_text)
                    self.tab.lblYoloVersion.setStyleSheet(
                        "color: #1565c0; font-weight: bold;"
                    )
                else:
                    self.tab.lblYoloVersion.setText("Unknown")
                    self.tab.lblYoloVersion.setStyleSheet(
                        "color: #999999; font-weight: bold;"
                    )

            if hasattr(self.tab, 'lblTaskType'):
                task_colors = {
                    'detect': '#2e7d32',
                    'segment': '#6a1b9a',
                    'classify': '#e65100'
                }
                self.tab.lblTaskType.setText(task_type)
                self.tab.lblTaskType.setStyleSheet(
                    f"color: {task_colors.get(task_type, '#333333')}; font-weight: bold;"
                )

            if hasattr(self.tab, 'lblRecommendedOpset'):
                self.tab.lblRecommendedOpset.setText(str(recommended_opset))

            # Auto-set opset combobox
            if hasattr(self.tab, 'comboOpset'):
                opset_str = str(recommended_opset)
                idx = self.tab.comboOpset.findText(opset_str)
                if idx >= 0:
                    self.tab.comboOpset.setCurrentIndex(idx)

            # Log detection result
            if yolo_version:
                self._log(
                    f"[{self._timestamp()}] Detected: YOLO{yolo_version} "
                    f"{task_type} ({model_name}) -> opset={recommended_opset}"
                )
            else:
                self._log(
                    f"[{self._timestamp()}] Model type: {task_type} "
                    f"(YOLO version not detected)"
                )

            # Store for later use in HEF compilation
            self._detected_model_info = detection

        except Exception as e:
            logger.error(f"Model detection failed: {e}")
            self._reset_detection_ui()

    def _reset_detection_ui(self) -> None:
        """Reset model detection UI labels to defaults."""
        if hasattr(self.tab, 'lblYoloVersion'):
            self.tab.lblYoloVersion.setText("-")
            self.tab.lblYoloVersion.setStyleSheet("color: #999999; font-weight: bold;")
        if hasattr(self.tab, 'lblTaskType'):
            self.tab.lblTaskType.setText("-")
            self.tab.lblTaskType.setStyleSheet("color: #999999; font-weight: bold;")
        if hasattr(self.tab, 'lblRecommendedOpset'):
            self.tab.lblRecommendedOpset.setText("-")
        self._detected_model_info = None

    def _detect_yolo_project(self, pt_path: str) -> Optional[dict]:
        """
        Detect YOLO project structure from PT file path.

        Looks for:
        - data.yaml (class count, dataset paths)
        - train/images directory (calibration)
        - Project structure patterns

        Args:
            pt_path: Path to PyTorch model file.

        Returns:
            Dict with project info or None if not detected.
        """
        from pathlib import Path

        pt_dir = Path(pt_path).parent

        # Search upward for data.yaml (max 4 levels)
        search_dirs = [pt_dir]
        current = pt_dir
        for _ in range(4):
            current = current.parent
            search_dirs.append(current)

        data_yaml_path = None
        project_root = None
        for search_dir in search_dirs:
            candidate = search_dir / 'data.yaml'
            if candidate.exists():
                data_yaml_path = candidate
                project_root = search_dir
                break

        if not data_yaml_path:
            return None

        result = {'project_root': str(project_root)}

        # Parse data.yaml for class info
        data_config = None
        try:
            import yaml
            with open(data_yaml_path, 'r') as f:
                data_config = yaml.safe_load(f)

            if data_config:
                nc = data_config.get('nc')
                if nc:
                    result['num_classes'] = int(nc)

                names = data_config.get('names')
                if names:
                    result['class_names'] = names

        except Exception:
            pass

        # Find calibration images directory
        calib_candidates = [
            project_root / 'train' / 'images',
            project_root / 'dataset' / 'train' / 'images',
            project_root / 'data' / 'train' / 'images',
            project_root / 'images' / 'train',
        ]

        # Also check paths from data.yaml
        try:
            if data_config and 'train' in data_config:
                train_path = data_config['train']
                if not os.path.isabs(train_path):
                    train_path = str(project_root / train_path)
                calib_candidates.insert(0, Path(train_path))
        except Exception:
            pass

        for calib_dir in calib_candidates:
            if calib_dir.exists() and calib_dir.is_dir():
                # Verify it has images
                image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
                has_images = any(
                    f.suffix.lower() in image_exts
                    for f in calib_dir.iterdir()
                    if f.is_file()
                )
                if has_images:
                    result['calib_dir'] = str(calib_dir)
                    break

        # Use deep detection results if available, fallback to path patterns
        if hasattr(self, '_detected_model_info') and self._detected_model_info:
            result['task_type'] = self._detected_model_info.get('task_type', 'detect')
            result['yolo_version'] = self._detected_model_info.get('yolo_version')
            result['recommended_opset'] = self._detected_model_info.get('recommended_opset')
            result['model_name'] = self._detected_model_info.get('model_name_prefix')
        else:
            # Fallback: detect task type from path
            path_lower = pt_path.lower()
            if '-seg' in path_lower or 'segment' in path_lower:
                result['task_type'] = 'segment'
            elif '-cls' in path_lower or 'classify' in path_lower:
                result['task_type'] = 'classify'
            else:
                result['task_type'] = 'detect'

        return result

    def _check_onnx_compatibility(self, onnx_path: str) -> None:
        """Check ONNX model compatibility when file is selected."""
        if not onnx_path or not os.path.exists(onnx_path):
            return

        if not ONNX_AVAILABLE:
            self._log(f"[{self._timestamp()}] Warning: ONNX package not installed, cannot validate")
            return

        try:
            result = validate_onnx_hailo_compatibility(onnx_path)

            if result['compatible']:
                self._log(
                    f"[{self._timestamp()}] ONNX compatible "
                    f"(naming: {result['naming_style']})"
                )
            else:
                for error in result.get('errors', []):
                    self._log(f"[{self._timestamp()}] ONNX Error: {error}")
                if result.get('recommended_action'):
                    self._log(
                        f"[{self._timestamp()}] Recommendation: "
                        f"{result['recommended_action']}"
                    )

            for warning in result.get('warnings', []):
                self._log(f"[{self._timestamp()}] Warning: {warning}")

        except Exception as e:
            logger.error(f"ONNX compatibility check failed: {e}")
            self._log(f"[{self._timestamp()}] Warning: Could not validate ONNX: {e}")

    def _toggle_model_zoo(self) -> None:
        """Toggle Model Zoo standard mode."""
        self.model_zoo_mode = not self.model_zoo_mode

        if self.model_zoo_mode:
            self._log(f"[{self._timestamp()}] Model Zoo mode enabled - using standard model")
            if hasattr(self.tab, 'editOnnxPath'):
                self.tab.editOnnxPath.setEnabled(False)
                self.tab.editOnnxPath.setPlaceholderText("Using Hailo Model Zoo standard model...")
            if hasattr(self.tab, 'btnBrowseOnnx'):
                self.tab.btnBrowseOnnx.setEnabled(False)
            if hasattr(self.tab, 'btnModelZoo'):
                self.tab.btnModelZoo.setText("Use Custom ONNX")
        else:
            self._log(f"[{self._timestamp()}] Model Zoo mode disabled - using custom ONNX")
            if hasattr(self.tab, 'editOnnxPath'):
                self.tab.editOnnxPath.setEnabled(True)
                self.tab.editOnnxPath.setPlaceholderText("")
            if hasattr(self.tab, 'btnBrowseOnnx'):
                self.tab.btnBrowseOnnx.setEnabled(True)
            if hasattr(self.tab, 'btnModelZoo'):
                self.tab.btnModelZoo.setText("Use Model Zoo")

    def _start_pt_to_onnx(self) -> None:
        """Start PT to ONNX conversion."""
        pt_path = self.tab.editPtPath.text()
        onnx_path = self.tab.editOnnxOutPath.text()

        if not pt_path:
            logger.warning("PT to ONNX conversion aborted: No PT file selected")
            QMessageBox.warning(self.tab, "Warning", "Please select a PT file.")
            return

        if not os.path.exists(pt_path):
            logger.warning(f"PT to ONNX conversion aborted: File not found: {pt_path}")
            QMessageBox.warning(self.tab, "Warning", f"PT file not found: {pt_path}")
            return

        # Ensure output directory exists
        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

        params = {
            'pt_path': pt_path,
            'onnx_path': onnx_path,
            'input_size': (self.tab.spinWidth.value(), self.tab.spinHeight.value()),
            'batch_size': self.tab.spinBatch.value(),
            'opset': int(self.tab.comboOpset.currentText())
        }

        logger.info(f"Initiating PT to ONNX conversion: {os.path.basename(pt_path)}")
        self._log(f"[{self._timestamp()}] Starting conversion...")
        self._set_converting(True)

        self.worker = ConvertWorker('pt_to_onnx', params)
        self.worker.progress.connect(self._update_progress)
        self.worker.log.connect(self._log)
        self.worker.finished.connect(self._on_conversion_finished)
        self.worker.error.connect(self._on_conversion_error)
        self.worker.start()

    def _start_onnx_to_hef(self) -> None:
        """Start ONNX to HEF compilation."""
        onnx_path = self.tab.editOnnxPath.text() if not self.model_zoo_mode else ''
        hef_path = self.tab.editHefOutPath.text()
        calib_dir = self.tab.editCalibPath.text()

        if not self.model_zoo_mode:
            if not onnx_path:
                QMessageBox.warning(self.tab, "Warning", "Please select an ONNX file.")
                return

            if not os.path.exists(onnx_path):
                QMessageBox.warning(self.tab, "Warning", f"ONNX file not found: {onnx_path}")
                return

        if not os.path.exists(calib_dir):
            QMessageBox.warning(
                self.tab, "Warning",
                f"Calibration directory not found: {calib_dir}"
            )
            return

        # Ensure output directory exists
        os.makedirs(os.path.dirname(hef_path), exist_ok=True)

        # Build params with detection info
        model_type = 'detect'
        model_name = 'yolov5s'
        calib_size = (self.tab.spinWidth.value(), self.tab.spinHeight.value())

        if hasattr(self, '_detected_model_info') and self._detected_model_info:
            task = self._detected_model_info.get('task_type', 'detect')
            model_type = task
            model_name = self._detected_model_info.get('model_name_prefix', 'yolov5s')

        params = {
            'onnx_path': onnx_path,
            'hef_path': hef_path,
            'calib_dir': calib_dir,
            'target': self.tab.comboTarget.currentText(),
            'model_type': model_type,
            'model_name': model_name,
            'calib_size': calib_size,
        }

        self._log(f"[{self._timestamp()}] Starting HEF compilation...")
        self._set_converting(True)

        self.worker = ConvertWorker('onnx_to_hef', params)
        self.worker.progress.connect(self._update_progress)
        self.worker.log.connect(self._log)
        self.worker.finished.connect(self._on_conversion_finished)
        self.worker.error.connect(self._on_conversion_error)
        self.worker.start()

    def _on_conversion_finished(self, success: bool, result: str) -> None:
        """Handle conversion completion."""
        self._set_converting(False)

        if success:
            self._log(f"[{self._timestamp()}] ✓ Success: {result}")
            QMessageBox.information(
                self.tab, "Success",
                f"Conversion completed successfully!\n\nOutput: {result}"
            )
        else:
            self._log(f"[{self._timestamp()}] ✗ Failed: {result}")
            QMessageBox.critical(
                self.tab, "Error",
                f"Conversion failed:\n\n{result}"
            )

    def _on_conversion_error(self, error_type: str, error_message: str) -> None:
        """Handle conversion error with type information."""
        self._log(f"[{self._timestamp()}] Error [{error_type}]: {error_message}")
        logger.error(f"Conversion error [{error_type}]: {error_message}")

    def _update_progress(self, value: int) -> None:
        """Update progress bar."""
        self.tab.progressConvert.setValue(value)

    def _log(self, message: str) -> None:
        """Add message to log."""
        self.tab.txtLog.append(message)

    def _clear_log(self) -> None:
        """Clear the log."""
        self.tab.txtLog.clear()
        self.tab.progressConvert.setValue(0)

    def _set_converting(self, converting: bool) -> None:
        """Enable/disable UI during conversion."""
        self.tab.btnConvertOnnx.setEnabled(not converting)
        self.tab.btnCompileHef.setEnabled(not converting)

    def _timestamp(self) -> str:
        """Get current timestamp string."""
        return datetime.now().strftime("%H:%M:%S")
