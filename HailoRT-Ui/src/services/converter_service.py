"""
Converter Service
Model conversion utilities for PT -> ONNX -> HEF pipeline.
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Tuple, Optional, Callable
import numpy as np
from utils.logger import setup_logger
from utils.exceptions import (
    ModelLoadError, ExportError, CompilationError, CalibrationError
)
from utils.onnx_utils import (
    validate_onnx_hailo_compatibility,
    extract_onnx_nodes,
    rename_onnx_nodes_to_hailo_style,
    patch_onnx_for_hailo,
    downgrade_opset_version,
    verify_onnx_for_hailo,
    detect_yolo_base_model,
    generate_hailo_yaml,
    ONNX_AVAILABLE
)
from utils.model_detection import (
    detect_yolo_version_from_pt,
    detect_yolo_task_from_pt,
    get_recommended_opset,
    detect_yolo_from_pt,
)

logger = setup_logger(__name__)


def _configure_torch_for_yolo():
    """
    Configure PyTorch 2.6+ for YOLO model loading.
    PyTorch 2.6 changed weights_only default to True for security.
    This adds YOLO model classes to safe globals for torch.load().
    """
    try:
        import torch

        # Set environment variable for subprocess calls (yolo CLI)
        os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'

        # For PyTorch 2.6+, add safe globals
        if hasattr(torch, 'serialization') and hasattr(torch.serialization, 'add_safe_globals'):
            try:
                # Try to import YOLO model classes and add them as safe
                from ultralytics.nn.tasks import DetectionModel, SegmentationModel, ClassificationModel
                torch.serialization.add_safe_globals([DetectionModel, SegmentationModel, ClassificationModel])
            except ImportError:
                pass

            try:
                # Also try yolov5 models
                from models.yolo import DetectionModel as YOLOv5Detection
                from models.yolo import SegmentationModel as YOLOv5Segment
                torch.serialization.add_safe_globals([YOLOv5Detection, YOLOv5Segment])
            except ImportError:
                pass

    except ImportError:
        pass


# Configure torch on module load
_configure_torch_for_yolo()


class ConverterService:
    """Service for model conversion operations."""

    def __init__(self):
        self.progress_callback: Optional[Callable[[int], None]] = None
        self.log_callback: Optional[Callable[[str], None]] = None
        self.cancel_callback: Optional[Callable[[], bool]] = None

    def set_callbacks(self, progress_cb=None, log_cb=None, cancel_cb=None):
        """Set progress, log, and cancellation callbacks."""
        self.progress_callback = progress_cb
        self.log_callback = log_cb
        self.cancel_callback = cancel_cb

    def _check_cancelled(self):
        """Check if cancellation was requested and raise if so."""
        if self.cancel_callback and self.cancel_callback():
            raise InterruptedError("Conversion cancelled by user")

    def _log(self, message: str):
        """Log message."""
        if self.log_callback:
            self.log_callback(message)
        logger.info(message)

    def _progress(self, value: int):
        """Update progress."""
        if self.progress_callback:
            self.progress_callback(value)

    def validate_onnx(self, onnx_path: str) -> dict:
        """
        Validate ONNX model for Hailo compatibility.

        Args:
            onnx_path: Path to ONNX model file.

        Returns:
            Validation result dict with 'compatible', 'naming_style', 'errors', 'warnings', 'recommended_action'.
        """
        self._log(f"Validating ONNX compatibility: {onnx_path}")
        result = validate_onnx_hailo_compatibility(onnx_path)

        # Log results
        if result['errors']:
            for error in result['errors']:
                self._log(f"Error: {error}")
        if result['warnings']:
            for warning in result['warnings']:
                self._log(f"Warning: {warning}")
        if result['compatible']:
            self._log(f"ONNX validation passed (naming style: {result['naming_style']})")
        else:
            self._log(f"ONNX validation failed. {result.get('recommended_action', '')}")

        return result

    def _patch_torch_load_for_yolo(self):
        """
        Patch torch.load for PyTorch 2.6+ compatibility with YOLO models.
        PyTorch 2.6+ defaults weights_only=True which breaks YOLO loading.
        """
        try:
            import torch

            # Check if already patched
            if hasattr(torch, '_original_load_for_yolo'):
                return

            _original_load = torch.load

            def patched_load(*args, **kwargs):
                if 'weights_only' not in kwargs:
                    kwargs['weights_only'] = False
                return _original_load(*args, **kwargs)

            torch._original_load_for_yolo = _original_load
            torch.load = patched_load
            self._log("Patched torch.load for PyTorch 2.6+ compatibility")

        except Exception as e:
            self._log(f"Warning: Could not patch torch.load: {e}")

    def convert_pt_to_onnx(
        self,
        pt_path: str,
        onnx_path: str,
        input_size: Tuple[int, int] = (640, 640),
        batch_size: int = 1,
        opset_version: int = 11
    ) -> bool:
        """
        Convert PyTorch model to ONNX format.

        Args:
            pt_path: Path to PyTorch model (.pt file)
            onnx_path: Output path for ONNX model
            input_size: Model input size (height, width)
            batch_size: Batch size for export
            opset_version: ONNX opset version

        Returns:
            True if conversion successful
        """
        import torch

        self._log(f"Loading PyTorch model: {pt_path}")
        self._progress(10)
        self._check_cancelled()

        # Ensure output directory exists
        out_dir = os.path.dirname(onnx_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        try:
            # Check if it's a YOLO model (Ultralytics)
            if self._is_yolo_model(pt_path):
                # Detect YOLO task type and version
                task_type = self._detect_yolo_task(pt_path)
                yolo_version = self._detect_yolo_version(pt_path)

                self._log(f"Detected YOLO task type: {task_type}")

                # Auto-upgrade opset for segmentation models
                if task_type == 'segment' and opset_version < 18:
                    original_opset = opset_version
                    opset_version = 18
                    self._log(f"Auto-upgrading opset from {original_opset} to {opset_version} for segmentation model")

                # Get recommended opset if not explicitly set
                if yolo_version:
                    recommended_opset = self._get_recommended_opset(yolo_version, task_type)
                    if opset_version != recommended_opset:
                        self._log(f"Recommended opset for {yolo_version} {task_type}: {recommended_opset} (using {opset_version})")

                self._check_cancelled()
                return self._export_yolo_to_onnx(
                    pt_path, onnx_path, input_size, opset_version
                )

            # Generic PyTorch model export
            self._check_cancelled()
            return self._export_generic_to_onnx(
                pt_path, onnx_path, input_size, batch_size, opset_version
            )

        except FileNotFoundError as e:
            self._log(f"Model file not found: {e}")
            raise
        except PermissionError as e:
            self._log(f"Permission denied: {e}")
            raise
        except Exception as e:
            # Final fallback for unexpected errors
            self._log(f"Conversion failed ({type(e).__name__}): {e}")
            raise

    def _detect_yolo_version(self, pt_path: str) -> Optional[str]:
        """
        Detect YOLO version from checkpoint file.
        Delegates to module-level detect_yolo_version_from_pt().

        Returns:
            'v5', 'v8', 'v9', 'v10', or None
        """
        version = detect_yolo_version_from_pt(pt_path)
        if version:
            self._log(f"Detected YOLO{version} model")
        return version

    def _is_yolo_model(self, pt_path: str) -> bool:
        """Check if model is a YOLO model."""
        version = self._detect_yolo_version(pt_path)
        if version:
            return True

        # Fallback: check file path patterns
        path_lower = pt_path.lower()
        yolo_path_indicators = ['yolo', 'best.pt', 'last.pt', 'weights/']
        if any(indicator in path_lower for indicator in yolo_path_indicators):
            self._log("Detected YOLO model from file path pattern (version unknown)")
            return True

        return False

    def _detect_yolo_task(self, pt_path: str) -> str:
        """
        Detect YOLO task type from PyTorch checkpoint file.
        Delegates to module-level detect_yolo_task_from_pt().

        Returns:
            'segment', 'detect', or 'classify'
        """
        return detect_yolo_task_from_pt(pt_path)

    def _get_recommended_opset(self, yolo_version: str, task_type: str) -> int:
        """
        Get recommended ONNX opset version.
        Delegates to module-level get_recommended_opset().

        Returns:
            Recommended opset version (11 for YOLOv5, 17 for others)
        """
        return get_recommended_opset(yolo_version, task_type)

    def _export_yolo_to_onnx(
        self,
        pt_path: str,
        onnx_path: str,
        input_size: Tuple[int, int],
        opset_version: int
    ) -> bool:
        """Export YOLO model using appropriate method based on version."""
        version = self._detect_yolo_version(pt_path)
        self._log(f"YOLO version detected: {version or 'unknown'}")
        self._progress(30)

        if version == 'v5':
            return self._export_yolov5_to_onnx(pt_path, onnx_path, input_size, opset_version)
        else:
            # v8 or unknown - try ultralytics first
            return self._export_yolov8_to_onnx(pt_path, onnx_path, input_size, opset_version)

    def _export_yolov5_to_onnx(
        self,
        pt_path: str,
        onnx_path: str,
        input_size: Tuple[int, int],
        opset_version: int
    ) -> bool:
        """
        Export YOLOv5 model to ONNX using official YOLOv5 export.py.

        IMPORTANT: For Hailo compatibility, we MUST use YOLOv5 official export.py.
        This produces node names like /model.24/m.0/Conv that Hailo Model Zoo expects.
        DO NOT use direct torch.onnx.export - it produces conv2d_N naming
        which is incompatible with Hailo.
        """
        from utils.onnx_utils import extract_onnx_nodes

        # Minimum opset 11 for Hailo compatibility
        hailo_opset = max(opset_version, 11)
        self._log(f"Using YOLOv5 official export (opset {hailo_opset})...")

        # Method 1: Use our wrapper script (handles PyTorch 2.6+ compatibility)
        try:
            import subprocess

            wrapper_path = Path(__file__).parent / 'yolov5_export_wrapper.py'
            if wrapper_path.exists():
                self._log("Using yolov5_export_wrapper.py for PyTorch 2.6+ compatibility...")
                self._progress(40)

                cmd = [
                    sys.executable, str(wrapper_path),
                    '--weights', pt_path,
                    '--img-size', str(input_size[0]), str(input_size[1]),
                    '--batch-size', '1',
                    '--device', 'cpu',
                    '--include', 'onnx',
                    '--opset', str(hailo_opset),
                    '--simplify'
                ]

                self._log(f"Running: {' '.join(cmd)}")
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=300
                )

                if result.stdout:
                    for line in result.stdout.strip().split('\n'):
                        self._log(f"  {line}")

                if result.returncode == 0:
                    default_onnx = Path(pt_path).with_suffix('.onnx')
                    if default_onnx.exists():
                        if str(default_onnx) != onnx_path:
                            shutil.move(str(default_onnx), onnx_path)
                        self._log(f"Exported to: {onnx_path}")

                        # Verify node naming
                        if ONNX_AVAILABLE:
                            node_info = extract_onnx_nodes(onnx_path)
                            style = node_info.get('naming_style', 'unknown')
                            self._log(f"ONNX node naming style: {style}")

                        self._progress(100)
                        return True
                else:
                    self._log(f"Wrapper export failed: {result.stderr}")
            else:
                self._log("yolov5_export_wrapper.py not found")

        except subprocess.TimeoutExpired:
            self._log("YOLOv5 wrapper export timed out")
        except Exception as e:
            self._log(f"Wrapper export failed ({type(e).__name__}): {e}")

        # Method 2: Try yolov5 package export directly
        try:
            import subprocess
            import site

            self._log("Attempting export via YOLOv5 export.py...")
            self._progress(50)

            search_paths = [
                Path(pt_path).parent.parent,
                Path(pt_path).parent.parent.parent,
                Path.home() / 'yolov5',
                Path('/opt/yolov5'),
            ]

            for site_path in site.getsitepackages() + [site.getusersitepackages()]:
                if site_path:
                    search_paths.append(Path(site_path) / 'yolov5')

            export_script = None
            for yolo_path in search_paths:
                candidate = yolo_path / 'export.py'
                if candidate.exists():
                    export_script = candidate
                    self._log(f"Found export.py at: {candidate}")
                    break

            if export_script:
                cmd = [
                    sys.executable, str(export_script),
                    '--weights', pt_path,
                    '--img-size', str(input_size[0]),
                    '--opset', str(hailo_opset),
                    '--include', 'onnx',
                    '--simplify'
                ]

                self._log(f"Running: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

                if result.returncode == 0:
                    default_onnx = Path(pt_path).with_suffix('.onnx')
                    if default_onnx.exists():
                        if str(default_onnx) != onnx_path:
                            shutil.move(str(default_onnx), onnx_path)
                        self._log(f"Exported to: {onnx_path}")
                        self._progress(100)
                        return True
                else:
                    self._log(f"YOLOv5 export.py stderr: {result.stderr}")
                    self._log(f"YOLOv5 export.py stdout: {result.stdout}")
            else:
                self._log("YOLOv5 export.py not found in any search path")

        except subprocess.TimeoutExpired:
            self._log("YOLOv5 export timed out")
        except (FileNotFoundError, PermissionError) as e:
            self._log(f"File access error: {e}")
        except Exception as e:
            self._log(f"YOLOv5 subprocess export failed ({type(e).__name__}): {e}")

        raise ExportError(
            "Failed to export YOLOv5 model. Please install yolov5 package:\n"
            "  pip install yolov5\n"
            "Or ensure YOLOv5 repository is available with export.py",
            stage="yolov5_export"
        )

    def _export_yolov8_to_onnx(
        self,
        pt_path: str,
        onnx_path: str,
        input_size: Tuple[int, int],
        opset_version: int
    ) -> bool:
        """Export YOLOv8 model to ONNX using ultralytics."""
        self._log("Using YOLOv8 export method...")

        try:
            from ultralytics import YOLO

            model = YOLO(pt_path)
            self._progress(50)

            self._log(f"Exporting to ONNX with opset {opset_version}...")

            export_path = model.export(
                format='onnx',
                imgsz=input_size[0],
                opset=opset_version,
                simplify=True,
                dynamic=False
            )

            self._progress(80)

            if export_path != onnx_path:
                shutil.move(str(export_path), onnx_path)

            self._log(f"Exported to: {onnx_path}")
            self._progress(100)
            return True

        except ImportError:
            raise ExportError(
                "YOLOv8 export requires 'ultralytics' package.\n"
                "Install with: pip install ultralytics",
                stage="yolov8_import"
            )
        except (FileNotFoundError, PermissionError) as e:
            self._log(f"File access error: {e}")
            raise
        except RuntimeError as e:
            self._log(f"YOLOv8 export runtime error: {e}")
            raise
        except Exception as e:
            # Final fallback for YOLOv8 export errors
            self._log(f"YOLOv8 export failed ({type(e).__name__}): {e}")
            raise

    def _export_generic_to_onnx(
        self,
        pt_path: str,
        onnx_path: str,
        input_size: Tuple[int, int],
        batch_size: int,
        opset_version: int
    ) -> bool:
        """Export generic PyTorch model to ONNX."""
        import torch

        self._progress(30)

        # Load model (weights_only=False needed for full model export)
        checkpoint = torch.load(pt_path, map_location='cpu', weights_only=False)

        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                model = checkpoint['model']
            elif 'state_dict' in checkpoint:
                raise ModelLoadError(
                    "Model file contains only state_dict. "
                    "Full model definition is required.",
                    model_path=pt_path
                )
            else:
                model = checkpoint
        else:
            model = checkpoint

        # Handle wrapped models
        if hasattr(model, 'model'):
            model = model.model

        model.eval()
        model.float()

        self._progress(50)
        self._log("Creating dummy input tensor...")

        # Create dummy input
        dummy_input = torch.randn(batch_size, 3, input_size[0], input_size[1])

        self._progress(60)
        self._log(f"Exporting to ONNX (opset={opset_version})...")

        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            opset_version=opset_version,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes={
                'images': {0: 'batch'},
                'output': {0: 'batch'}
            },
            do_constant_folding=True
        )

        self._progress(90)

        # Verify ONNX model
        self._verify_onnx(onnx_path)

        self._log(f"Successfully exported to: {onnx_path}")
        self._progress(100)
        return True

    def _verify_onnx(self, onnx_path: str):
        """Verify ONNX model is valid."""
        try:
            import onnx
            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)
            self._log("ONNX model verification passed")
        except ImportError:
            self._log("Warning: onnx package not installed, skipping verification")
        except FileNotFoundError as e:
            self._log(f"Warning: ONNX file not found: {e}")
        except ValueError as e:
            self._log(f"Warning: Invalid ONNX model: {e}")
        except Exception as e:
            # Final fallback for ONNX verification errors
            self._log(f"Warning: ONNX verification failed ({type(e).__name__}): {e}")

    def compile_onnx_to_hef(
        self,
        onnx_path: str,
        hef_path: str,
        calib_dir: str,
        target: str = "hailo8",
        optimization_level: int = 2,
        model_type: str = "detect",
        num_classes: int = 80,
        model_name: str = "yolov5s",
        calib_size: Tuple[int, int] = (640, 640)
    ) -> bool:
        """
        Compile ONNX model to Hailo HEF format.

        Args:
            onnx_path: Path to ONNX model
            hef_path: Output path for HEF model
            calib_dir: Directory containing calibration images
            target: Target Hailo device (hailo8, hailo8l, hailo15h)
            optimization_level: Optimization level (0-3)
            model_type: Model type (detect, segment, classify)
            num_classes: Number of detection classes
            model_name: Model name for hailomz
            calib_size: Calibration image size

        Returns:
            True if compilation successful
        """
        self._log(f"Starting ONNX to HEF compilation...")
        self._log(f"Target: {target}, Model: {model_name}, Type: {model_type}")
        self._progress(10)
        self._check_cancelled()

        # Ensure output directory exists
        hef_dir = os.path.dirname(hef_path)
        if hef_dir:
            os.makedirs(hef_dir, exist_ok=True)

        # Strategy: try hailomz CLI first, fall back to SDK
        if not onnx_path:
            # Model Zoo standard mode (no custom ONNX)
            return self._compile_modelzoo_standard(
                hef_path, target, model_name, model_type,
                num_classes, calib_dir, calib_size
            )

        # Validate and prepare ONNX for Hailo BEFORE compilation
        end_nodes = []
        # Convert model_type for onnx_utils ('segment' -> 'seg')
        onnx_model_type = 'seg' if model_type == 'segment' else 'detect'
        if ONNX_AVAILABLE:
            validation = self.validate_onnx(onnx_path)

            # Auto-rename if naming is incompatible
            naming_style = validation.get('naming_style', 'unknown')
            if naming_style in ('pytorch_generic', 'ultralytics', 'unknown'):
                self._log(f"ONNX naming '{naming_style}' needs conversion to Hailo format...")
                renamed_path = onnx_path.replace('.onnx', '_hailo.onnx')
                onnx_path, end_nodes = rename_onnx_nodes_to_hailo_style(
                    onnx_path, renamed_path, onnx_model_type
                )
                self._log(f"Renamed ONNX saved to: {onnx_path}")
                if end_nodes:
                    self._log(f"End nodes: {end_nodes}")
            else:
                # Extract end nodes from compatible ONNX
                node_info = extract_onnx_nodes(onnx_path)
                detection_heads = node_info.get('detection_heads', [])
                proto_node = node_info.get('proto_node')

                # For segment models, ensure proto node is first
                if model_type == 'segment' and proto_node:
                    end_nodes = [proto_node] + detection_heads
                    self._log(f"Segment model: proto={proto_node}, heads={detection_heads}")
                else:
                    end_nodes = detection_heads

                if end_nodes:
                    self._log(f"Detected end nodes: {end_nodes}")
        else:
            self._log("Warning: ONNX package not available, skipping validation")

        # Try hailomz CLI first
        self._check_cancelled()
        if shutil.which('hailomz'):
            try:
                return self._compile_with_hailomz(
                    onnx_path, hef_path, calib_dir, target,
                    model_type, num_classes, model_name, calib_size,
                    end_nodes=end_nodes
                )
            except Exception as e:
                self._log(f"hailomz CLI failed: {e}")
                self._log("Falling back to SDK compilation...")
                self._progress(10)
        else:
            self._log("hailomz CLI not found, using SDK compilation")

        # Store calib_size for SDK path to use
        self._current_calib_size = calib_size

        # Fallback: SDK compilation
        return self._compile_with_sdk(
            onnx_path, hef_path, calib_dir, target,
            optimization_level, model_type, model_name,
            end_nodes=end_nodes
        )

    def _compile_with_sdk(
        self,
        onnx_path: str,
        hef_path: str,
        calib_dir: str,
        target: str,
        optimization_level: int = 2,
        model_type: str = "detect",
        model_name: str = "yolov5s",
        end_nodes: list = None
    ) -> bool:
        """Compile ONNX to HEF using Hailo SDK (ClientRunner)."""
        # Defense-in-depth: ensure env vars are set even if main.py wasn't entry point
        os.environ.setdefault('HAILO_DISABLE_MO_SUB_PROCESS', '1')
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
        os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

        try:
            from hailo_sdk_client import ClientRunner

            self._log("Initializing Hailo SDK compiler...")
            runner = ClientRunner(hw_arch=target)

            # ONNX validation already done in compile_onnx_to_hef

            self._progress(20)
            self._log(f"Translating ONNX model: {onnx_path}")

            # Build translate kwargs
            translate_kwargs = {}
            net_name = model_name
            if model_type == 'segment':
                net_name = f"{model_name}_seg"
            translate_kwargs['net_name'] = net_name
            if end_nodes:
                translate_kwargs['end_node_names'] = end_nodes
                self._log(f"Using end nodes for translate: {end_nodes}")

            # Translate ONNX to Hailo format
            hn, npz = runner.translate_onnx_model(onnx_path, **translate_kwargs)

            self._progress(40)
            self._log("Loading calibration data...")

            # Load and prepare calibration data (explicitly use NHWC layout)
            # Use calib_size from compile_onnx_to_hef if available
            sdk_calib_size = getattr(self, '_current_calib_size', (640, 640))
            calib_data = self._load_calibration_data(calib_dir, input_size=sdk_calib_size, layout='NHWC')

            self._progress(50)
            self._log(f"Running optimization with {len(calib_data)} calibration images...")

            # Optimize (quantization)
            runner.optimize(calib_data)

            self._progress(80)
            self._log("Compiling to HEF...")

            # Compile
            hef = runner.compile()

            # Ensure output directory exists
            hef_out_dir = os.path.dirname(hef_path)
            if hef_out_dir:
                os.makedirs(hef_out_dir, exist_ok=True)

            # Save HEF
            with open(hef_path, 'wb') as f:
                f.write(hef)

            self._log(f"Successfully compiled to: {hef_path}")
            self._progress(100)
            return True

        except ImportError:
            raise CompilationError(
                "Hailo SDK (hailo_sdk_client) not installed. "
                "Please install the Hailo Dataflow Compiler.",
                sdk_error="import_error"
            )
        except FileNotFoundError as e:
            self._log(f"File not found: {e}")
            raise CompilationError(str(e), sdk_error="file_not_found")
        except ValueError as e:
            self._log(f"Invalid calibration data: {e}")
            raise CalibrationError(str(e), calib_dir=calib_dir)
        except RuntimeError as e:
            self._log(f"Compilation runtime error: {e}")
            raise CompilationError(str(e), sdk_error="runtime_error")
        except Exception as e:
            self._log(f"SDK compilation failed ({type(e).__name__}): {e}")
            raise CompilationError(str(e), sdk_error=type(e).__name__)

    def _compile_with_hailomz(
        self,
        onnx_path: str,
        hef_path: str,
        calib_dir: str,
        target: str,
        model_type: str,
        num_classes: int,
        model_name: str,
        calib_size: Tuple[int, int],
        end_nodes: list = None
    ) -> bool:
        """Compile ONNX to HEF using hailomz CLI."""
        import subprocess

        self._log("Using hailomz CLI compilation...")
        self._progress(15)

        # Prepare calibration data as numpy file
        calib_npy = self._prepare_calibration_npy(calib_dir, calib_size)

        # Use end_nodes from parent (already extracted in compile_onnx_to_hef)
        if end_nodes is None:
            end_nodes = []

        # Build model identifier for hailomz yaml
        yaml_name = model_name
        if model_type == 'segment':
            yaml_name = f"{model_name}_seg"

        # Build hailomz command (model name is positional arg after 'compile')
        cmd = [
            'hailomz', 'compile', yaml_name,
            '--ckpt', onnx_path,
            '--hw-arch', target,
            '--calib-path', calib_npy,
            '--classes', str(num_classes)
        ]

        if end_nodes:
            cmd.append('--end-node-names')
            cmd.extend(end_nodes)

        self._log(f"Running: {' '.join(cmd)}")
        self._progress(20)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            # Log output
            if result.stdout:
                for line in result.stdout.strip().split('\n')[-20:]:
                    self._log(f"  {line}")

            if result.returncode != 0:
                self._log(f"hailomz failed (exit code {result.returncode})")
                if result.stderr:
                    self._log(f"Error: {result.stderr[:500]}")
                raise CompilationError(f"hailomz compilation failed (exit code {result.returncode})", sdk_error="hailomz_cli")

            self._progress(90)

            # Find and move generated HEF file
            # hailomz outputs to current directory or a default location
            default_hef = Path(f'{model_name}.hef')
            if default_hef.exists():
                shutil.move(str(default_hef), hef_path)
            else:
                # Search in common output locations
                for search_dir in [Path('.'), Path('hailomz_output'), Path(onnx_path).parent]:
                    for hef_file in search_dir.glob('*.hef'):
                        shutil.move(str(hef_file), hef_path)
                        break
                    else:
                        continue
                    break

            if os.path.exists(hef_path):
                self._log(f"Successfully compiled to: {hef_path}")
                self._progress(100)
                return True
            else:
                raise CompilationError("HEF output file not found after hailomz compilation", sdk_error="hailomz_output")

        except subprocess.TimeoutExpired:
            self._log("hailomz compilation timed out (1 hour limit)")
            raise CompilationError("hailomz compilation timed out", sdk_error="timeout")

    def _compile_modelzoo_standard(
        self,
        hef_path: str,
        target: str,
        model_name: str,
        model_type: str,
        num_classes: int,
        calib_dir: str,
        calib_size: Tuple[int, int]
    ) -> bool:
        """Compile using Model Zoo standard model (auto-download)."""
        import subprocess

        self._log(f"Using Model Zoo standard model: {model_name}")
        self._progress(15)

        # Prepare calibration data
        calib_npy = self._prepare_calibration_npy(calib_dir, calib_size)

        cmd = [
            'hailomz', 'compile', model_name,
            '--hw-arch', target,
            '--calib-path', calib_npy,
            '--classes', str(num_classes)
        ]

        self._log(f"Running: {' '.join(cmd)}")
        self._progress(20)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600
            )

            if result.stdout:
                for line in result.stdout.strip().split('\n')[-20:]:
                    self._log(f"  {line}")

            if result.returncode != 0:
                self._log(f"Model Zoo compilation failed (exit code {result.returncode})")
                if result.stderr:
                    self._log(f"Error: {result.stderr[:500]}")
                raise CompilationError("Model Zoo compilation failed", sdk_error="modelzoo_cli")

            self._progress(90)

            # Find generated HEF
            default_hef = Path(f'{model_name}.hef')
            if default_hef.exists():
                shutil.move(str(default_hef), hef_path)

            if os.path.exists(hef_path):
                self._log(f"Successfully compiled to: {hef_path}")
                self._progress(100)
                return True
            else:
                raise CompilationError("HEF output file not found", sdk_error="modelzoo_output")

        except subprocess.TimeoutExpired:
            self._log("Model Zoo compilation timed out")
            raise CompilationError("Model Zoo compilation timed out", sdk_error="timeout")

    def _prepare_calibration_npy(
        self,
        calib_dir: str,
        calib_size: Tuple[int, int] = (640, 640),
        output_dir: str = None
    ) -> str:
        """
        Prepare calibration data as numpy file for hailomz.

        Args:
            calib_dir: Directory containing calibration images
            calib_size: Target image size
            output_dir: Directory for output .npy file (defaults to calib_dir)

        Returns:
            Path to saved numpy file
        """
        calib_data = self._load_calibration_data(
            calib_dir, input_size=calib_size, layout='NHWC'
        )

        # Save to output dir or inside calib_dir itself
        save_dir = output_dir or calib_dir
        npy_path = os.path.join(save_dir, 'calib_data.npy')
        os.makedirs(os.path.dirname(npy_path), exist_ok=True)
        np.save(npy_path, calib_data)
        self._log(f"Prepared calibration data: {npy_path} (shape: {calib_data.shape})")
        return npy_path

    def _load_calibration_data(
        self,
        calib_dir: str,
        input_size: Tuple[int, int] = (640, 640),
        max_images: int = 500,
        layout: str = 'NHWC'
    ) -> np.ndarray:
        """
        Load calibration images from directory.

        Args:
            calib_dir: Directory containing calibration images
            input_size: Target input size (height, width)
            max_images: Maximum number of images to load
            layout: Data layout - 'NHWC' (default) or 'NCHW'

        Returns:
            Calibration data array in specified layout
        """
        from PIL import Image

        calib_dir = Path(calib_dir)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

        images = []
        skipped = 0
        for f in sorted(calib_dir.iterdir()):
            if f.suffix.lower() in image_extensions:
                if len(images) >= max_images:
                    break

                try:
                    img = Image.open(f).convert('RGB')
                    img = img.resize(input_size)
                    img_array = np.array(img).astype(np.float32) / 255.0

                    # Convert to NCHW if requested
                    if layout == 'NCHW':
                        img_array = np.transpose(img_array, (2, 0, 1))

                    images.append(img_array)
                except Exception as e:
                    skipped += 1
                    self._log(f"Warning: skipping {f.name}: {e}")

        if skipped > 0:
            self._log(f"Skipped {skipped} corrupted/unreadable images")

        if not images:
            raise CalibrationError(f"No calibration images found in {calib_dir}", calib_dir=str(calib_dir), image_count=0)

        result = np.stack(images, axis=0)
        self._log(f"Loaded {len(images)} calibration images ({layout} format, shape: {result.shape})")
        return result

    def prepare_calibration_dataset(
        self,
        input_dir: str,
        output_path: str,
        input_size: Tuple[int, int] = (640, 640),
        max_images: int = 500
    ) -> str:
        """
        Prepare calibration dataset as numpy file.

        Args:
            input_dir: Directory containing calibration images
            output_path: Output path for .npy file
            input_size: Target input size
            max_images: Maximum number of images to include

        Returns:
            Path to saved calibration file
        """
        self._log(f"Preparing calibration dataset from: {input_dir}")

        calib_data = self._load_calibration_data(input_dir, input_size, max_images)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        np.save(output_path, calib_data)
        self._log(f"Saved calibration data to: {output_path}")
        self._log(f"Shape: {calib_data.shape}")

        return output_path
