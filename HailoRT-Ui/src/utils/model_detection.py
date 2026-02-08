"""
Model Detection Utilities

Module-level functions for detecting YOLO model version, task type,
and recommended settings from PyTorch checkpoint files.

These functions are designed to be importable by both the UI layer
(for early auto-configuration) and the service layer (for conversion).
"""

import os
import zipfile
from typing import Optional, Tuple

from utils.logger import setup_logger

logger = setup_logger('model_detection')


def detect_yolo_version_from_pt(pt_path: str) -> Optional[str]:
    """
    Detect YOLO version from PyTorch checkpoint file.

    Uses multiple detection methods in priority order:
    1. File path patterns (fast, no file I/O)
    2. Zipfile pickle data scan (no torch import needed)
    3. torch.load checkpoint inspection (most reliable)
    4. ModuleNotFoundError analysis (fallback)

    Args:
        pt_path: Path to .pt file.

    Returns:
        'v5', 'v8', 'v9', 'v10', or None if not detected.
    """
    # Method 0: File path patterns (fastest)
    path_lower = os.path.basename(pt_path).lower()
    if 'yolov9' in path_lower:
        return 'v9'
    if 'yolov10' in path_lower:
        return 'v10'

    # Method 1: Peek into checkpoint's pickle data for module references
    try:
        with zipfile.ZipFile(pt_path, 'r') as zf:
            for pkl_name in ['archive/data.pkl', 'data.pkl']:
                if pkl_name in zf.namelist():
                    with zf.open(pkl_name) as f:
                        content = f.read(4096).decode('latin-1', errors='ignore')

                        if 'ultralytics.nn' in content or 'ultralytics.engine' in content:
                            logger.debug("Detected YOLOv8 from module reference")
                            return 'v8'

                        if 'models.yolo' in content or 'models.common' in content:
                            logger.debug("Detected YOLOv5 from module reference")
                            return 'v5'
                    break
    except (zipfile.BadZipFile, KeyError, OSError):
        pass

    # Method 2: Try loading and inspect checkpoint structure
    try:
        import torch
        checkpoint = torch.load(pt_path, map_location='cpu', weights_only=False)

        if isinstance(checkpoint, dict):
            if 'train_args' in checkpoint:
                return 'v8'

            if 'model' in checkpoint:
                model = checkpoint['model']
                module_path = type(model).__module__
                class_name = type(model).__name__

                if 'ultralytics' in module_path:
                    return 'v8'
                elif 'models' in module_path:
                    return 'v5'
                elif 'yolo' in class_name.lower() or 'detect' in class_name.lower():
                    return 'v8'

            yolo_v5_keys = {'model', 'ema', 'updates', 'optimizer', 'epoch'}
            if yolo_v5_keys.issubset(set(checkpoint.keys())) and 'train_args' not in checkpoint:
                return 'v5'

        return None

    except ModuleNotFoundError as e:
        error_msg = str(e)
        if 'models' in error_msg and 'ultralytics' not in error_msg:
            return 'v5'
        if 'ultralytics' in error_msg:
            return 'v8'
        return None
    except (RuntimeError, KeyError, TypeError):
        return None


def detect_yolo_task_from_pt(pt_path: str) -> str:
    """
    Detect YOLO task type from PyTorch checkpoint file.

    Args:
        pt_path: Path to .pt file.

    Returns:
        'segment', 'detect', or 'classify'.
    """
    # Method 1: Check pickle data for task indicators
    try:
        with zipfile.ZipFile(pt_path, 'r') as zf:
            for pkl_name in ['archive/data.pkl', 'data.pkl']:
                if pkl_name in zf.namelist():
                    with zf.open(pkl_name) as f:
                        content = f.read(8192).decode('latin-1', errors='ignore')
                        if 'SegmentationModel' in content or 'Segment' in content:
                            return 'segment'
                        if 'ClassificationModel' in content or 'Classify' in content:
                            return 'classify'
                    break
    except (zipfile.BadZipFile, KeyError, OSError):
        pass

    # Method 2: Check file path patterns
    path_lower = pt_path.lower()
    if '-seg' in path_lower or 'segment' in path_lower or '_seg' in path_lower:
        return 'segment'
    if '-cls' in path_lower or 'classify' in path_lower or '_cls' in path_lower:
        return 'classify'

    # Method 3: Try loading and inspect model class name
    try:
        import torch
        checkpoint = torch.load(pt_path, map_location='cpu', weights_only=False)

        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model = checkpoint['model']
            class_name = type(model).__name__

            if 'Segment' in class_name:
                return 'segment'
            if 'Classif' in class_name:
                return 'classify'

    except (ModuleNotFoundError, RuntimeError, KeyError, TypeError):
        pass

    return 'detect'


def get_recommended_opset(yolo_version: Optional[str], task_type: str = 'detect') -> int:
    """
    Get recommended ONNX opset version based on YOLO version and task.

    Args:
        yolo_version: 'v5', 'v8', 'v9', 'v10', or None.
        task_type: 'detect', 'segment', 'classify'.

    Returns:
        Recommended opset version (11 for YOLOv5, 17 for v8+).
    """
    if yolo_version == 'v5':
        return 11
    else:
        return 17


def detect_yolo_from_pt(pt_path: str) -> dict:
    """
    Comprehensive YOLO model detection from PT file.

    Aggregates version, task, and opset detection into a single call.
    This is the primary function for UI auto-configuration.

    Args:
        pt_path: Path to .pt file.

    Returns:
        Dict with 'yolo_version', 'task_type', 'recommended_opset',
        'model_name_prefix' (e.g., 'yolov5s').
    """
    yolo_version = detect_yolo_version_from_pt(pt_path)
    task_type = detect_yolo_task_from_pt(pt_path)
    recommended_opset = get_recommended_opset(yolo_version, task_type)

    # Build model name prefix from filename
    model_name_prefix = _guess_model_name(pt_path, yolo_version)

    return {
        'yolo_version': yolo_version,
        'task_type': task_type,
        'recommended_opset': recommended_opset,
        'model_name_prefix': model_name_prefix,
    }


def _guess_model_name(pt_path: str, yolo_version: Optional[str]) -> str:
    """Guess a model name from filename and version."""
    basename = os.path.splitext(os.path.basename(pt_path))[0].lower()

    # Try to extract model size from filename (e.g., yolov5s, yolov8m)
    for size in ['n', 's', 'm', 'l', 'x']:
        for ver in ['yolov5', 'yolov8', 'yolov9', 'yolov10']:
            if f'{ver}{size}' in basename:
                return f'{ver}{size}'

    # Fallback based on detected version
    if yolo_version == 'v5':
        return 'yolov5s'
    elif yolo_version == 'v8':
        return 'yolov8s'
    elif yolo_version == 'v9':
        return 'yolov9s'
    elif yolo_version == 'v10':
        return 'yolov10s'
    else:
        return 'yolov5s'
