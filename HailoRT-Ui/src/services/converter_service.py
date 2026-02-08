"""
Converter Service
Model conversion utilities for PT -> ONNX -> HEF pipeline.
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Callable
import numpy as np
from utils.logger import setup_logger

logger = setup_logger(__name__)


class ConverterService:
    """Service for model conversion operations."""

    def __init__(self):
        self.progress_callback: Optional[Callable[[int], None]] = None
        self.log_callback: Optional[Callable[[str], None]] = None

    def set_callbacks(self, progress_cb=None, log_cb=None):
        """Set progress and log callbacks."""
        self.progress_callback = progress_cb
        self.log_callback = log_cb

    def _log(self, message: str):
        """Log message."""
        if self.log_callback:
            self.log_callback(message)
        logger.info(message)

    def _progress(self, value: int):
        """Update progress."""
        if self.progress_callback:
            self.progress_callback(value)

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

        # Ensure output directory exists
        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

        try:
            # Check if it's a YOLO model (Ultralytics)
            if self._is_yolo_model(pt_path):
                return self._export_yolo_to_onnx(
                    pt_path, onnx_path, input_size, opset_version
                )

            # Generic PyTorch model export
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

        Returns:
            'v5', 'v8', or None if not a YOLO model
        """
        import zipfile

        # Method 1: Peek into checkpoint's pickle data for module references
        try:
            with zipfile.ZipFile(pt_path, 'r') as zf:
                # PyTorch saves checkpoints as zip files
                for pkl_name in ['archive/data.pkl', 'data.pkl']:
                    if pkl_name in zf.namelist():
                        with zf.open(pkl_name) as f:
                            # Read more bytes to find module references
                            content = f.read(4096).decode('latin-1', errors='ignore')

                            # YOLOv8: uses ultralytics.nn.tasks module
                            if 'ultralytics.nn' in content or 'ultralytics.engine' in content:
                                self._log("Detected YOLOv8 model from module reference")
                                return 'v8'

                            # YOLOv5: uses models.yolo or models.common module
                            if 'models.yolo' in content or 'models.common' in content:
                                self._log("Detected YOLOv5 model from module reference")
                                return 'v5'
                        break
        except (zipfile.BadZipFile, KeyError, Exception):
            pass

        # Method 2: Try loading and inspect checkpoint structure
        try:
            import torch
            checkpoint = torch.load(pt_path, map_location='cpu', weights_only=False)

            if isinstance(checkpoint, dict):
                # YOLOv8 specific: has 'train_args' with version info
                if 'train_args' in checkpoint:
                    train_args = checkpoint.get('train_args', {})
                    if isinstance(train_args, dict):
                        # YOLOv8 stores model info in train_args
                        self._log("Detected YOLOv8 model from train_args")
                        return 'v8'

                # Check model class module path
                if 'model' in checkpoint:
                    model = checkpoint['model']
                    module_path = type(model).__module__
                    class_name = type(model).__name__

                    if 'ultralytics' in module_path:
                        self._log(f"Detected YOLOv8 model: {module_path}.{class_name}")
                        return 'v8'
                    elif 'models' in module_path:
                        self._log(f"Detected YOLOv5 model: {module_path}.{class_name}")
                        return 'v5'
                    elif 'yolo' in class_name.lower() or 'detect' in class_name.lower():
                        # Fallback: assume v8 if ultralytics loadable
                        self._log(f"Detected YOLO model (assuming v8): {class_name}")
                        return 'v8'

                # YOLOv5 specific keys pattern
                yolo_v5_keys = {'model', 'ema', 'updates', 'optimizer', 'epoch'}
                if yolo_v5_keys.issubset(set(checkpoint.keys())) and 'train_args' not in checkpoint:
                    self._log("Detected YOLOv5 model from checkpoint keys")
                    return 'v5'

            return None

        except ModuleNotFoundError as e:
            error_msg = str(e)
            # YOLOv5: "No module named 'models'"
            if 'models' in error_msg and 'ultralytics' not in error_msg:
                self._log(f"Detected YOLOv5 model (missing module: {e})")
                return 'v5'
            # YOLOv8: "No module named 'ultralytics'"
            if 'ultralytics' in error_msg:
                self._log(f"Detected YOLOv8 model (missing module: {e})")
                return 'v8'
            return None
        except (RuntimeError, KeyError, TypeError):
            return None

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
        """Export YOLOv5 model to ONNX."""
        self._log("Using YOLOv5 export method...")

        # Method 1: Direct torch export with sys.path manipulation
        try:
            import torch
            import sys

            self._log("Attempting direct YOLOv5 export with torch...")
            self._progress(40)

            # Find yolov5 package path and add to sys.path for model loading
            yolov5_path = None
            try:
                import yolov5
                yolov5_path = Path(yolov5.__file__).parent
                self._log(f"Found yolov5 package at: {yolov5_path}")
            except ImportError:
                pass

            # Add yolov5 path to sys.path temporarily
            original_path = sys.path.copy()
            if yolov5_path:
                sys.path.insert(0, str(yolov5_path.parent))
                sys.path.insert(0, str(yolov5_path))

            try:
                # Load model with weights_only=False
                self._log("Loading model with weights_only=False...")
                checkpoint = torch.load(pt_path, map_location='cpu', weights_only=False)

                if isinstance(checkpoint, dict) and 'model' in checkpoint:
                    model = checkpoint['model'].float()
                else:
                    model = checkpoint.float()

                model.eval()

                # Handle model fusion for export
                if hasattr(model, 'fuse'):
                    model.fuse()

                self._progress(60)
                self._log("Creating dummy input and exporting to ONNX...")

                # Create dummy input
                dummy_input = torch.zeros(1, 3, input_size[0], input_size[1])

                # Export to ONNX
                torch.onnx.export(
                    model,
                    dummy_input,
                    onnx_path,
                    opset_version=opset_version,
                    input_names=['images'],
                    output_names=['output0', 'output1'] if hasattr(model, 'model') and len(model.model) > 1 else ['output'],
                    dynamic_axes={'images': {0: 'batch'}, 'output0': {0: 'batch'}} if hasattr(model, 'model') else None,
                    do_constant_folding=True,
                )

                self._progress(90)

                # Verify and simplify
                if Path(onnx_path).exists():
                    try:
                        import onnx
                        from onnxsim import simplify
                        self._log("Simplifying ONNX model...")
                        model_onnx = onnx.load(onnx_path)
                        model_simp, check = simplify(model_onnx)
                        if check:
                            onnx.save(model_simp, onnx_path)
                            self._log("ONNX model simplified successfully")
                    except ImportError:
                        self._log("onnxsim not installed, skipping simplification")
                    except (ValueError, RuntimeError) as e:
                        self._log(f"ONNX simplification failed: {e}")
                    except Exception as e:
                        # Final fallback for ONNX-specific errors
                        self._log(f"ONNX simplification failed ({type(e).__name__}): {e}")

                    self._log(f"Exported to: {onnx_path}")
                    self._progress(100)
                    return True

            finally:
                sys.path = original_path

        except FileNotFoundError as e:
            self._log(f"Model file not found: {e}")
        except (RuntimeError, ValueError) as e:
            self._log(f"Model export error: {e}")
        except Exception as e:
            # Final fallback for torch export errors
            self._log(f"Direct torch export failed ({type(e).__name__}): {e}")

        # Method 2: Try YOLOv5 export.py via subprocess
        try:
            import subprocess
            import sys
            import site

            self._log("Attempting export via YOLOv5 export.py...")
            self._progress(50)

            # Find yolov5 export.py in various locations
            search_paths = [
                Path(pt_path).parent.parent,  # weights/../..
                Path(pt_path).parent.parent.parent,  # deeper nesting
                Path.home() / 'yolov5',
                Path('/opt/yolov5'),
            ]

            # Add site-packages locations
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
                    '--opset', str(opset_version),
                    '--include', 'onnx',
                    '--simplify'
                ]

                self._log(f"Running: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

                if result.returncode == 0:
                    default_onnx = Path(pt_path).with_suffix('.onnx')
                    if default_onnx.exists():
                        if str(default_onnx) != onnx_path:
                            import shutil
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
            # Final fallback for subprocess errors
            self._log(f"YOLOv5 subprocess export failed ({type(e).__name__}): {e}")

        raise RuntimeError(
            "Failed to export YOLOv5 model. Please install yolov5 package:\n"
            "  pip install yolov5\n"
            "Or ensure YOLOv5 repository is available with export.py"
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
                import shutil
                shutil.move(str(export_path), onnx_path)

            self._log(f"Exported to: {onnx_path}")
            self._progress(100)
            return True

        except ImportError:
            raise ImportError(
                "YOLOv8 export requires 'ultralytics' package.\n"
                "Install with: pip install ultralytics"
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
                raise ValueError(
                    "Model file contains only state_dict. "
                    "Full model definition is required."
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
        optimization_level: int = 2
    ) -> bool:
        """
        Compile ONNX model to Hailo HEF format.

        Args:
            onnx_path: Path to ONNX model
            hef_path: Output path for HEF model
            calib_dir: Directory containing calibration images
            target: Target Hailo device (hailo8, hailo8l, hailo15h)
            optimization_level: Optimization level (0-3)

        Returns:
            True if compilation successful
        """
        self._log(f"Starting ONNX to HEF compilation...")
        self._log(f"Target: {target}")
        self._progress(10)

        try:
            from hailo_sdk_client import ClientRunner

            self._log("Initializing Hailo compiler...")
            runner = ClientRunner(hw_arch=target)

            self._progress(20)
            self._log(f"Translating ONNX model: {onnx_path}")

            # Translate ONNX to Hailo format
            hn, npz = runner.translate_onnx_model(onnx_path)

            self._progress(40)
            self._log("Loading calibration data...")

            # Load and prepare calibration data
            calib_data = self._load_calibration_data(calib_dir)

            self._progress(50)
            self._log(f"Running optimization with {len(calib_data)} calibration images...")

            # Optimize (quantization)
            runner.optimize(calib_data)

            self._progress(80)
            self._log("Compiling to HEF...")

            # Compile
            hef = runner.compile()

            # Ensure output directory exists
            os.makedirs(os.path.dirname(hef_path), exist_ok=True)

            # Save HEF
            with open(hef_path, 'wb') as f:
                f.write(hef)

            self._log(f"Successfully compiled to: {hef_path}")
            self._progress(100)
            return True

        except ImportError:
            raise ImportError(
                "Hailo SDK (hailo_sdk_client) not installed. "
                "Please install the Hailo Dataflow Compiler."
            )
        except FileNotFoundError as e:
            self._log(f"File not found: {e}")
            raise
        except ValueError as e:
            self._log(f"Invalid calibration data: {e}")
            raise
        except RuntimeError as e:
            self._log(f"Compilation runtime error: {e}")
            raise
        except Exception as e:
            # Final fallback for compilation errors
            self._log(f"Compilation failed ({type(e).__name__}): {e}")
            raise

    def _load_calibration_data(
        self,
        calib_dir: str,
        input_size: Tuple[int, int] = (640, 640),
        max_images: int = 500
    ) -> np.ndarray:
        """Load calibration images from directory."""
        from PIL import Image

        calib_dir = Path(calib_dir)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

        images = []
        for f in sorted(calib_dir.iterdir()):
            if f.suffix.lower() in image_extensions:
                if len(images) >= max_images:
                    break

                img = Image.open(f).convert('RGB')
                img = img.resize(input_size)
                img_array = np.array(img).astype(np.float32) / 255.0
                images.append(img_array)

        if not images:
            raise ValueError(f"No calibration images found in {calib_dir}")

        self._log(f"Loaded {len(images)} calibration images")
        return np.stack(images, axis=0)

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
