"""
Convert Tab Controller
Manages model conversion from PT to ONNX to HEF.
"""

from __future__ import annotations

import os
from typing import Optional, Dict, Any, List
from datetime import datetime
from PyQt5.QtWidgets import QWidget, QFileDialog, QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5 import uic

from utils.logger import setup_logger

logger = setup_logger(__name__)


class ConvertWorker(QThread):
    """Worker thread for model conversion using ConverterService."""
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, conversion_type: str, params: Dict[str, Any]) -> None:
        super().__init__()
        self.conversion_type: str = conversion_type
        self.params: Dict[str, Any] = params

    def run(self) -> None:
        """Run conversion."""
        try:
            logger.debug(f"ConvertWorker started for conversion type: {self.conversion_type}")
            if self.conversion_type == 'pt_to_onnx':
                self._convert_pt_to_onnx()
            elif self.conversion_type == 'onnx_to_hef':
                self._convert_onnx_to_hef()
        except Exception as e:
            logger.error(f"ConvertWorker failed: {e}", exc_info=True)
            self.finished.emit(False, str(e))

    def _convert_pt_to_onnx(self) -> None:
        """Convert PyTorch model to ONNX using ConverterService."""
        from services.converter_service import ConverterService

        pt_path = self.params['pt_path']
        onnx_path = self.params['onnx_path']
        input_size = self.params['input_size']
        batch_size = self.params['batch_size']
        opset = self.params['opset']

        logger.info(f"Starting PT to ONNX conversion: {pt_path} -> {onnx_path}")
        logger.debug(f"Conversion parameters: input_size={input_size}, batch_size={batch_size}, opset={opset}")

        # Use ConverterService for proper YOLO version detection
        service = ConverterService()
        service.set_callbacks(
            progress_cb=lambda v: self.progress.emit(v),
            log_cb=lambda m: self.log.emit(m)
        )

        try:
            success = service.convert_pt_to_onnx(
                pt_path=pt_path,
                onnx_path=onnx_path,
                input_size=input_size,
                batch_size=batch_size,
                opset_version=opset
            )

            if success:
                logger.info(f"PT to ONNX conversion successful: {onnx_path}")
                self.log.emit(f"✓ Conversion complete: {onnx_path}")
                self.finished.emit(True, onnx_path)
            else:
                logger.error("PT to ONNX conversion failed")
                self.finished.emit(False, "Conversion failed")

        except Exception as e:
            logger.error(f"PT to ONNX conversion error: {e}", exc_info=True)
            self.log.emit(f"Error: {str(e)}")
            self.finished.emit(False, str(e))

    def _convert_onnx_to_hef(self) -> None:
        """Convert ONNX model to HEF using ConverterService."""
        from services.converter_service import ConverterService

        onnx_path = self.params['onnx_path']
        hef_path = self.params['hef_path']
        calib_dir = self.params['calib_dir']
        target = self.params['target']

        logger.info(f"Starting ONNX to HEF compilation: {onnx_path} -> {hef_path}")
        logger.debug(f"Compilation parameters: calib_dir={calib_dir}, target={target}")

        # Use ConverterService
        service = ConverterService()
        service.set_callbacks(
            progress_cb=lambda v: self.progress.emit(v),
            log_cb=lambda m: self.log.emit(m)
        )

        try:
            success = service.compile_onnx_to_hef(
                onnx_path=onnx_path,
                hef_path=hef_path,
                calib_dir=calib_dir,
                target=target
            )

            if success:
                logger.info(f"ONNX to HEF compilation successful: {hef_path}")
                self.log.emit(f"✓ Compilation complete: {hef_path}")
                self.finished.emit(True, hef_path)
            else:
                logger.error("ONNX to HEF compilation failed")
                self.finished.emit(False, "Compilation failed")

        except ImportError as e:
            logger.error(f"Hailo SDK not installed: {e}")
            self.log.emit("Error: Hailo SDK not installed")
            self.log.emit("Please install hailo_dataflow_compiler")
            self.finished.emit(False, str(e))
        except Exception as e:
            logger.error(f"ONNX to HEF compilation error: {e}", exc_info=True)
            self.log.emit(f"Error: {str(e)}")
            self.finished.emit(False, str(e))


class ConvertTabController:
    """Controller for model conversion tab."""

    def __init__(self, tab_widget: QWidget, base_path: str) -> None:
        self.tab: QWidget = tab_widget
        self.base_path: str = base_path
        self.worker: Optional[ConvertWorker] = None

        # Load UI into tab
        ui_path = os.path.join(base_path, 'ui', 'tabs', 'convert_tab.ui')
        uic.loadUi(ui_path, self.tab)

        # Set default paths
        self._set_default_paths()

        # Connect signals
        self._connect_signals()

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

    def _browse_pt_file(self) -> None:
        """Browse for PyTorch model file."""
        path, _ = QFileDialog.getOpenFileName(
            self.tab,
            "Select PyTorch Model",
            os.path.join(self.base_path, 'data', 'models', 'pt'),
            "PyTorch Models (*.pt *.pth);;All Files (*)"
        )
        if path:
            logger.debug(f"Selected PT file: {path}")
            self.tab.editPtPath.setText(path)

    def _browse_onnx_output(self) -> None:
        """Browse for ONNX output path."""
        path, _ = QFileDialog.getSaveFileName(
            self.tab,
            "Save ONNX Model",
            os.path.join(self.base_path, 'data', 'models', 'onnx'),
            "ONNX Models (*.onnx)"
        )
        if path:
            if not path.endswith('.onnx'):
                path += '.onnx'
            self.tab.editOnnxOutPath.setText(path)

    def _browse_onnx_file(self) -> None:
        """Browse for ONNX model file."""
        path, _ = QFileDialog.getOpenFileName(
            self.tab,
            "Select ONNX Model",
            os.path.join(self.base_path, 'data', 'models', 'onnx'),
            "ONNX Models (*.onnx);;All Files (*)"
        )
        if path:
            logger.debug(f"Selected ONNX file: {path}")
            self.tab.editOnnxPath.setText(path)

    def _browse_calib_dir(self) -> None:
        """Browse for calibration directory."""
        path = QFileDialog.getExistingDirectory(
            self.tab,
            "Select Calibration Images Directory",
            os.path.join(self.base_path, 'data', 'calibration', 'images')
        )
        if path:
            self.tab.editCalibPath.setText(path)

    def _browse_hef_output(self) -> None:
        """Browse for HEF output path."""
        path, _ = QFileDialog.getSaveFileName(
            self.tab,
            "Save HEF Model",
            os.path.join(self.base_path, 'data', 'models', 'hef'),
            "Hailo Models (*.hef)"
        )
        if path:
            if not path.endswith('.hef'):
                path += '.hef'
            self.tab.editHefOutPath.setText(path)

    def _auto_fill_onnx_path(self, pt_path: str) -> None:
        """Auto-fill ONNX output path based on PT path."""
        if pt_path and pt_path.endswith('.pt'):
            base_name = os.path.basename(pt_path).replace('.pt', '.onnx')
            onnx_dir = os.path.join(self.base_path, 'data', 'models', 'onnx')
            self.tab.editOnnxOutPath.setText(os.path.join(onnx_dir, base_name))

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
        self.worker.start()

    def _start_onnx_to_hef(self) -> None:
        """Start ONNX to HEF compilation."""
        onnx_path = self.tab.editOnnxPath.text()
        hef_path = self.tab.editHefOutPath.text()
        calib_dir = self.tab.editCalibPath.text()

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

        params = {
            'onnx_path': onnx_path,
            'hef_path': hef_path,
            'calib_dir': calib_dir,
            'target': self.tab.comboTarget.currentText()
        }

        self._log(f"[{self._timestamp()}] Starting HEF compilation...")
        self._set_converting(True)

        self.worker = ConvertWorker('onnx_to_hef', params)
        self.worker.progress.connect(self._update_progress)
        self.worker.log.connect(self._log)
        self.worker.finished.connect(self._on_conversion_finished)
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
