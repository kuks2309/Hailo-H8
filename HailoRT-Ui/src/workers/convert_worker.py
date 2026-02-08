"""
Convert Worker - Background thread for model conversion operations.
"""

from typing import Dict, Any
from PyQt5.QtCore import QThread, pyqtSignal

from utils.logger import setup_logger
from utils.exceptions import (
    ModelLoadError, ExportError, CompilationError, CalibrationError
)

logger = setup_logger(__name__)


class ConvertWorker(QThread):
    """Worker thread for model conversion using ConverterService."""
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    error = pyqtSignal(str, str)  # (error_type, error_message)

    def __init__(self, conversion_type: str, params: Dict[str, Any]) -> None:
        super().__init__()
        self.conversion_type: str = conversion_type
        self.params: Dict[str, Any] = params
        self._cancelled: bool = False

    def cancel(self) -> None:
        """Request cooperative cancellation."""
        logger.info("ConvertWorker cancellation requested")
        self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested."""
        return self._cancelled

    def run(self) -> None:
        """Run conversion."""
        try:
            logger.debug(f"ConvertWorker started for conversion type: {self.conversion_type}")
            if self.conversion_type == 'pt_to_onnx':
                self._convert_pt_to_onnx()
            elif self.conversion_type == 'onnx_to_hef':
                self._convert_onnx_to_hef()
        except ModelLoadError as e:
            self._handle_error('ModelLoadError', str(e))
        except ExportError as e:
            self._handle_error('ExportError', str(e))
        except CalibrationError as e:
            self._handle_error('CalibrationError', str(e))
        except CompilationError as e:
            self._handle_error('CompilationError', str(e))
        except Exception as e:
            logger.error(f"ConvertWorker failed: {e}", exc_info=True)
            self._handle_error(type(e).__name__, str(e))

    def _handle_error(self, error_type: str, error_message: str) -> None:
        """Handle conversion error by emitting signals."""
        logger.error(f"ConvertWorker error [{error_type}]: {error_message}")
        self.error.emit(error_type, error_message)
        self.finished.emit(False, error_message)

    def _check_cancelled(self) -> None:
        """Raise if cancellation was requested."""
        if self._cancelled:
            raise InterruptedError("Conversion cancelled by user")

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

        self._check_cancelled()

        service = ConverterService()
        service.set_callbacks(
            progress_cb=lambda v: self.progress.emit(v),
            log_cb=lambda m: self.log.emit(m),
            cancel_cb=lambda: self._cancelled
        )

        try:
            success = service.convert_pt_to_onnx(
                pt_path=pt_path,
                onnx_path=onnx_path,
                input_size=input_size,
                batch_size=batch_size,
                opset_version=opset
            )

            self._check_cancelled()

            if success:
                logger.info(f"PT to ONNX conversion successful: {onnx_path}")
                self.log.emit(f"✓ Conversion complete: {onnx_path}")
                self.finished.emit(True, onnx_path)
            else:
                logger.error("PT to ONNX conversion failed")
                self.finished.emit(False, "Conversion failed")

        except InterruptedError:
            logger.info("PT to ONNX conversion cancelled by user")
            self.log.emit("Conversion cancelled")
            self.finished.emit(False, "Cancelled by user")
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
        model_type = self.params.get('model_type', 'detect')
        num_classes = self.params.get('num_classes', 80)
        model_name = self.params.get('model_name', 'yolov5s')
        calib_size = self.params.get('calib_size', (640, 640))

        logger.info(f"Starting ONNX to HEF compilation: {onnx_path} -> {hef_path}")
        logger.debug(f"Compilation parameters: calib_dir={calib_dir}, target={target}, model_type={model_type}")

        self._check_cancelled()

        service = ConverterService()
        service.set_callbacks(
            progress_cb=lambda v: self.progress.emit(v),
            log_cb=lambda m: self.log.emit(m),
            cancel_cb=lambda: self._cancelled
        )

        try:
            success = service.compile_onnx_to_hef(
                onnx_path=onnx_path,
                hef_path=hef_path,
                calib_dir=calib_dir,
                target=target,
                model_type=model_type,
                num_classes=num_classes,
                model_name=model_name,
                calib_size=calib_size
            )

            self._check_cancelled()

            if success:
                logger.info(f"ONNX to HEF compilation successful: {hef_path}")
                self.log.emit(f"✓ Compilation complete: {hef_path}")
                self.finished.emit(True, hef_path)
            else:
                logger.error("ONNX to HEF compilation failed")
                self.finished.emit(False, "Compilation failed")

        except InterruptedError:
            logger.info("ONNX to HEF compilation cancelled by user")
            self.log.emit("Compilation cancelled")
            self.finished.emit(False, "Cancelled by user")
        except ImportError as e:
            logger.error(f"Hailo SDK not installed: {e}")
            self.log.emit("Error: Hailo SDK not installed")
            self.log.emit("Please install hailo_dataflow_compiler")
            self.finished.emit(False, str(e))
        except Exception as e:
            logger.error(f"ONNX to HEF compilation error: {e}", exc_info=True)
            self.log.emit(f"Error: {str(e)}")
            self.finished.emit(False, str(e))
