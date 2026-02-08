"""
Background worker for model conversion.
Runs conversion tasks in a separate thread.
"""

from PyQt5.QtCore import QThread, pyqtSignal

from ..core.converter import ModelConverter
from ..core.exceptions import (
    CompilerUIError,
    ModelLoadError,
    ExportError,
    CompilationError,
    CalibrationError
)


class ConvertWorker(QThread):
    """
    Worker thread for model conversion tasks.

    Signals:
        progress(int): Progress percentage (0-100)
        log(str): Log message
        finished(bool, str): Completion status and result/error message
        error(str, str): Error type and message
    """

    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    error = pyqtSignal(str, str)  # error_type, error_message

    def __init__(self, task_type: str, params: dict, parent=None):
        """
        Initialize worker.

        Args:
            task_type: 'pt_to_onnx' or 'onnx_to_hef'
            params: Task parameters dictionary
            parent: Parent QObject
        """
        super().__init__(parent)
        self.task_type = task_type
        self.params = params
        self.converter = ModelConverter()

        # Connect converter callbacks to signals
        self.converter.set_callbacks(
            progress_cb=self._on_progress,
            log_cb=self._on_log
        )

    def _on_progress(self, value: int):
        """Handle progress update from converter."""
        self.progress.emit(value)

    def _on_log(self, message: str):
        """Handle log message from converter."""
        self.log.emit(message)

    def run(self):
        """Run the conversion task."""
        try:
            if self.task_type == 'pt_to_onnx':
                self._run_pt_to_onnx()
            elif self.task_type == 'onnx_to_hef':
                self._run_onnx_to_hef()
            else:
                self.error.emit('InvalidTask', f'Unknown task type: {self.task_type}')
                self.finished.emit(False, f'Unknown task type: {self.task_type}')

        except ModelLoadError as e:
            self._handle_error('ModelLoadError', str(e))
        except ExportError as e:
            self._handle_error('ExportError', str(e))
        except CalibrationError as e:
            self._handle_error('CalibrationError', str(e))
        except CompilationError as e:
            self._handle_error('CompilationError', str(e))
        except CompilerUIError as e:
            self._handle_error('CompilerUIError', str(e))
        except Exception as e:
            self._handle_error('UnexpectedError', str(e))

    def _handle_error(self, error_type: str, message: str):
        """Handle and emit error."""
        self.log.emit(f"Error [{error_type}]: {message}")
        self.error.emit(error_type, message)
        self.finished.emit(False, message)

    def _run_pt_to_onnx(self):
        """Run PT to ONNX conversion."""
        self.log.emit("Starting PT to ONNX conversion...")

        pt_path = self.params['pt_path']
        onnx_path = self.params['onnx_path']
        input_size = self.params.get('input_size', (640, 640))
        batch_size = self.params.get('batch_size', 1)
        opset = self.params.get('opset', 17)

        result_path = self.converter.convert_pt_to_onnx(
            pt_path=pt_path,
            onnx_path=onnx_path,
            input_size=input_size,
            batch_size=batch_size,
            opset_version=opset
        )

        self.log.emit(f"Conversion complete: {result_path}")
        self.finished.emit(True, result_path)

    def _run_onnx_to_hef(self):
        """Run ONNX to HEF compilation."""
        self.log.emit("Starting ONNX to HEF compilation...")

        onnx_path = self.params['onnx_path']
        hef_path = self.params['hef_path']
        calib_dir = self.params['calib_dir']
        target = self.params.get('target', 'hailo8')
        calib_size = self.params.get('calib_size', (640, 640))
        model_type = self.params.get('model_type', 'detect')
        num_classes = self.params.get('num_classes', 80)
        model_name = self.params.get('model_name', 'yolov5s')

        result_path = self.converter.compile_onnx_to_hef(
            onnx_path=onnx_path,
            hef_path=hef_path,
            calib_dir=calib_dir,
            target=target,
            calib_size=calib_size,
            model_type=model_type,
            num_classes=num_classes,
            model_name=model_name
        )

        self.log.emit(f"Compilation complete: {result_path}")
        self.finished.emit(True, result_path)
