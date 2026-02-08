"""
Custom exceptions for HailoRT-Ui.
"""


class HailoRTUIError(Exception):
    """Base exception class for HailoRT-Ui."""
    pass


class HailoEnvironmentError(HailoRTUIError):
    """Raised when environment check fails (missing packages)."""

    def __init__(self, message: str, missing_packages: list = None):
        super().__init__(message)
        self.missing_packages = missing_packages or []


class ConversionError(HailoRTUIError):
    """Base class for conversion-related errors."""
    pass


class ModelLoadError(ConversionError):
    """Raised when model loading fails."""

    def __init__(self, message: str, model_path: str = None):
        super().__init__(message)
        self.model_path = model_path


class ExportError(ConversionError):
    """Raised when ONNX export fails."""

    def __init__(self, message: str, stage: str = None):
        super().__init__(message)
        self.stage = stage


class CompilationError(ConversionError):
    """Raised when HEF compilation fails."""

    def __init__(self, message: str, sdk_error: str = None):
        super().__init__(message)
        self.sdk_error = sdk_error


class CalibrationError(ConversionError):
    """Raised when calibration data has issues."""

    def __init__(self, message: str, calib_dir: str = None, image_count: int = 0):
        super().__init__(message)
        self.calib_dir = calib_dir
        self.image_count = image_count


class DeviceError(HailoRTUIError):
    """Base class for device-related errors."""
    pass


class DeviceConnectionError(DeviceError):
    """Raised when device connection fails."""

    def __init__(self, message: str, device_id: str = None):
        super().__init__(message)
        self.device_id = device_id


class InferenceError(DeviceError):
    """Raised when inference execution fails."""

    def __init__(self, message: str, model_name: str = None):
        super().__init__(message)
        self.model_name = model_name
