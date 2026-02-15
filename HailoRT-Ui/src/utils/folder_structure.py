"""
Folder structure detection for dataset/project directories.
Auto-detects calibration images, model paths, and data.yaml configuration.
"""

import os
from typing import Dict, Optional, Tuple, List

from utils.logger import setup_logger

logger = setup_logger(__name__)

# Calibration data source folders (in priority order)
CALIBRATION_FOLDERS = ["train/images", "valid/images", "test/images"]

# Image file extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def find_calibration_folder(base_path: str) -> Tuple[str, bool]:
    """
    Find the best available calibration folder.

    Checks train/images, valid/images, test/images in priority order.

    Args:
        base_path: Root directory to search.

    Returns:
        Tuple of (folder_path, exists_with_images).
    """
    for folder in CALIBRATION_FOLDERS:
        full_path = os.path.join(base_path, folder)
        if os.path.isdir(full_path):
            if _has_images(full_path):
                return full_path, True

    # Return first option as default (even if doesn't exist)
    return os.path.join(base_path, CALIBRATION_FOLDERS[0]), False


def _has_images(directory: str) -> bool:
    """Check if a directory contains image files."""
    try:
        for f in os.listdir(directory):
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                return True
    except OSError:
        pass
    return False


def _count_images(directory: str) -> int:
    """Count image files in a directory."""
    count = 0
    try:
        for f in os.listdir(directory):
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                count += 1
    except OSError:
        pass
    return count


def parse_data_yaml(base_path: str) -> Optional[dict]:
    """
    Parse data.yaml for dataset configuration.

    Args:
        base_path: Root directory containing data.yaml.

    Returns:
        Parsed config dict or None.
    """
    yaml_path = os.path.join(base_path, 'data.yaml')
    if not os.path.exists(yaml_path):
        return None

    try:
        import yaml
        with open(yaml_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except ImportError:
        logger.warning("PyYAML not installed, cannot parse data.yaml")
    except Exception as e:
        logger.error(f"Failed to parse data.yaml: {e}")

    return None


def detect_project_structure(base_path: str) -> Dict:
    """
    Detect dataset/project structure and return auto-fill paths.

    Scans the given directory for:
    - Calibration images (train/images, valid/images, test/images)
    - Model directories (models/pt, models/onnx, models/hef)
    - data.yaml (class count, class names)
    - PT model files

    Args:
        base_path: Root directory of the dataset/project.

    Returns:
        Dictionary with detected paths and metadata.
    """
    result = {
        'base': base_path,
        'valid': False,
        'summary': 'No dataset structure detected',
    }

    if not os.path.isdir(base_path):
        return result

    # Find calibration folder
    calib_path, calib_found = find_calibration_folder(base_path)
    result['calib_dir'] = calib_path
    result['calib_found'] = calib_found
    if calib_found:
        result['calib_count'] = _count_images(calib_path)

    # Check model directories
    models_dir = os.path.join(base_path, 'models')
    result['onnx_dir'] = os.path.join(models_dir, 'onnx')
    result['hef_dir'] = os.path.join(models_dir, 'hef')
    result['har_dir'] = os.path.join(models_dir, 'har')

    # Find PT files
    pt_dir = os.path.join(models_dir, 'pt')
    pt_files = _find_pt_files(pt_dir)
    if not pt_files:
        # Also check base_path directly for .pt files
        pt_files = _find_pt_files(base_path)
    result['pt_files'] = pt_files

    # Find existing ONNX files
    onnx_files = _find_files_by_ext(os.path.join(models_dir, 'onnx'), '.onnx')
    if not onnx_files:
        onnx_files = _find_files_by_ext(base_path, '.onnx')
    result['onnx_files'] = onnx_files

    # Find existing HEF files
    hef_files = _find_files_by_ext(os.path.join(models_dir, 'hef'), '.hef')
    if not hef_files:
        hef_files = _find_files_by_ext(base_path, '.hef')
    result['hef_files'] = hef_files

    # Find existing HAR files
    har_files = _find_files_by_ext(os.path.join(models_dir, 'har'), '.har')
    if not har_files:
        har_files = _find_files_by_ext(base_path, '.har')
    result['har_files'] = har_files

    # Parse data.yaml
    data_config = parse_data_yaml(base_path)
    if data_config:
        result['data_yaml'] = True
        nc = data_config.get('nc')
        if nc:
            result['num_classes'] = int(nc)
        names = data_config.get('names')
        if names:
            result['class_names'] = names

        # Check train path from data.yaml
        train_path = data_config.get('train')
        if train_path and not calib_found:
            if not os.path.isabs(train_path):
                train_path = os.path.join(base_path, train_path)
            if os.path.isdir(train_path) and _has_images(train_path):
                result['calib_dir'] = train_path
                result['calib_found'] = True
                result['calib_count'] = _count_images(train_path)
    else:
        result['data_yaml'] = False

    # Validate: at least calibration or model files found
    result['valid'] = (
        result.get('calib_found', False) or bool(pt_files) or
        result.get('data_yaml', False) or bool(onnx_files) or bool(hef_files)
    )

    # Structure summary for logging
    found_items = []
    if result.get('calib_found'):
        found_items.append(f"calibration ({result.get('calib_count', 0)} images)")
    if pt_files:
        found_items.append(f"PT ({len(pt_files)})")
    if onnx_files:
        found_items.append(f"ONNX ({len(onnx_files)})")
    if hef_files:
        found_items.append(f"HEF ({len(hef_files)})")
    if har_files:
        found_items.append(f"HAR ({len(har_files)})")
    if result.get('data_yaml'):
        nc = result.get('num_classes', '?')
        found_items.append(f"data.yaml (nc={nc})")
    result['summary'] = ', '.join(found_items) if found_items else 'No dataset structure detected'

    return result


def _find_files_by_ext(directory: str, ext: str) -> List[str]:
    """Find files with a specific extension in a directory."""
    files = []
    if not os.path.isdir(directory):
        return files
    try:
        for f in os.listdir(directory):
            if f.endswith(ext):
                files.append(os.path.join(directory, f))
    except OSError:
        pass
    return sorted(files)


def _find_pt_files(directory: str) -> List[str]:
    """Find PyTorch model files in a directory."""
    pt_files = []
    if not os.path.isdir(directory):
        return pt_files

    try:
        for f in os.listdir(directory):
            if f.endswith(('.pt', '.pth')):
                pt_files.append(os.path.join(directory, f))
    except OSError:
        pass

    return sorted(pt_files)
