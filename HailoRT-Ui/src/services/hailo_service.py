"""
Hailo Service
Wrapper for HailoRT API.
"""

import numpy as np
from typing import Optional, List, Dict, Any
from utils.logger import setup_logger

logger = setup_logger(__name__)


class HailoService:
    """Service for interacting with Hailo device."""

    def __init__(self):
        self.device = None
        self.hef = None
        self.network_group = None
        self.input_vstreams = None
        self.output_vstreams = None
        self._check_hailo_available()

    def _check_hailo_available(self):
        """Check if HailoRT is available."""
        try:
            from hailo_platform import HailoRTException
            self.hailo_available = True
        except ImportError:
            self.hailo_available = False

    def connect(self) -> Optional[Any]:
        """Connect to Hailo device."""
        if not self.hailo_available:
            raise ImportError("HailoRT not installed")

        from hailo_platform import VDevice, HailoRTException

        try:
            self.device = VDevice()
            return self.device
        except HailoRTException as e:
            logger.error(f"Failed to connect to Hailo device: {e}")
            return None

    def disconnect(self):
        """Disconnect from Hailo device."""
        if self.network_group:
            self.network_group = None
        if self.device:
            self.device = None

    def get_device_info(self) -> Dict[str, Any]:
        """Get device information."""
        if not self.device:
            return {}

        try:
            from hailo_platform import Device

            # Get physical device
            physical_device = self.device.get_physical_devices()[0]

            info = {
                'device_name': 'Hailo-8',
                'architecture': physical_device.get_architecture(),
                'serial': physical_device.get_serial_number(),
                'firmware': physical_device.get_fw_version(),
                'driver': self._get_driver_version(),
                'temperature': physical_device.get_chip_temperature(),
                'power': physical_device.get_power_measurement(),
                'utilization': self._get_utilization()
            }
            return info

        except (IndexError, AttributeError) as e:
            logger.error(f"Device access error: {e}")
            return {}
        except ImportError as e:
            logger.error(f"Failed to import hailo_platform: {e}")
            return {}
        except Exception as e:
            # Final fallback for HailoRT-specific exceptions (HailoRTException, etc.)
            logger.error(f"Unexpected error getting device info ({type(e).__name__}): {e}")
            return {}

    def _get_driver_version(self) -> str:
        """Get HailoRT driver version."""
        try:
            import hailo_platform
            return hailo_platform.__version__
        except (AttributeError, ImportError):
            return "Unknown"

    def _get_utilization(self) -> float:
        """Get device utilization."""
        # This would require monitoring the inference pipeline
        # For now, return 0 if not running inference
        return 0.0

    def load_model(self, hef_path: str) -> bool:
        """Load HEF model."""
        if not self.device:
            raise RuntimeError("Device not connected")

        from hailo_platform import HEF

        try:
            self.hef = HEF(hef_path)

            # Configure network group
            configure_params = self.device.create_configure_params(self.hef)
            self.network_group = self.device.configure(self.hef, configure_params)[0]

            # Get input/output info
            self.input_vstream_info = self.network_group.get_input_vstream_infos()
            self.output_vstream_info = self.network_group.get_output_vstream_infos()

            return True

        except FileNotFoundError as e:
            logger.error(f"HEF file not found: {e}")
            return False
        except (IndexError, AttributeError) as e:
            logger.error(f"Failed to configure network group: {e}")
            return False
        except Exception as e:
            # Final fallback for HailoRT-specific exceptions (HailoRTException, etc.)
            logger.error(f"Failed to load model ({type(e).__name__}): {e}")
            return False

    def unload_model(self):
        """Unload current model."""
        self.hef = None
        self.network_group = None

    def infer(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run inference on frame."""
        if not self.network_group:
            raise RuntimeError("No model loaded")

        from hailo_platform import InferVStreams, InputVStreamParams, OutputVStreamParams

        try:
            # Preprocess frame
            input_data = self._preprocess(frame)

            # Setup vstreams
            input_params = InputVStreamParams.make_from_network_group(
                self.network_group, quantized=False
            )
            output_params = OutputVStreamParams.make_from_network_group(
                self.network_group, quantized=False
            )

            # Run inference
            with InferVStreams(self.network_group, input_params, output_params) as pipeline:
                input_dict = {self.input_vstream_info[0].name: input_data}
                output = pipeline.infer(input_dict)

            # Postprocess
            detections = self._postprocess(output, frame.shape)

            return detections

        except (IndexError, KeyError) as e:
            logger.error(f"Input/output stream error: {e}")
            return []
        except ValueError as e:
            logger.error(f"Invalid input data: {e}")
            return []
        except Exception as e:
            # Final fallback for HailoRT-specific exceptions (HailoRTException, etc.)
            logger.error(f"Inference error ({type(e).__name__}): {e}")
            return []

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for inference."""
        import cv2

        # Get expected input shape
        input_shape = self.input_vstream_info[0].shape

        # Resize
        resized = cv2.resize(frame, (input_shape[2], input_shape[1]))

        # Normalize
        normalized = resized.astype(np.float32) / 255.0

        # Add batch dimension
        batched = np.expand_dims(normalized, axis=0)

        return batched

    def _postprocess(self, output: Dict, original_shape: tuple) -> List[Dict[str, Any]]:
        """Postprocess inference output."""
        detections = []

        # This is a simplified postprocess for YOLO-like models
        # Actual implementation depends on the specific model

        output_name = list(output.keys())[0]
        raw_output = output[output_name]

        # Parse detections (simplified)
        # Real implementation would depend on model output format

        return detections

    def get_model_info(self) -> Dict[str, Any]:
        """Get loaded model information."""
        if not self.hef:
            return {}

        try:
            return {
                'input_shape': self.input_vstream_info[0].shape if self.input_vstream_info else None,
                'output_shape': self.output_vstream_info[0].shape if self.output_vstream_info else None,
                'input_names': [v.name for v in self.input_vstream_info] if self.input_vstream_info else [],
                'output_names': [v.name for v in self.output_vstream_info] if self.output_vstream_info else [],
            }
        except (IndexError, AttributeError) as e:
            logger.error(f"Error accessing model stream info: {e}")
            return {}
        except Exception as e:
            # Final fallback for unexpected errors
            logger.error(f"Error getting model info ({type(e).__name__}): {e}")
            return {}
