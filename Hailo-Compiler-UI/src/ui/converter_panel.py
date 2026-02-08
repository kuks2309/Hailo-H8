"""
Converter utility functions for Hailo-Compiler-UI.
Contains YOLO project detection and data.yaml parsing.
"""

import os
import yaml

from ..core.converter import (
    detect_yolo_version_from_pt,
    detect_yolo_task_from_pt,
    get_recommended_opset
)


def parse_yolo_data_yaml(yaml_path: str) -> dict:
    """
    Parse YOLO data.yaml file to extract dataset configuration.

    Returns:
        dict with keys: nc (num classes), names (class names),
        train, val, test paths
    """
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return data or {}
    except (FileNotFoundError, yaml.YAMLError, Exception):
        return {}


def detect_yolo_project_from_pt(pt_path: str) -> dict:
    """
    Detect YOLO project structure and model info from PT file.

    Performs deep analysis of PT file to detect:
    - YOLO version (v5, v8, v9, v10) from file content
    - Model task type (detect, segment, classify) from model class
    - Recommended opset version based on version and task

    Expected project structure:
        project_root/
        ├── train/images/
        ├── valid/images/
        ├── models/pt/*.pt
        └── data.yaml

    Returns:
        dict with: project_root, data_yaml, train_images, nc, names,
                   model_type, yolo_version, recommended_opset
    """
    result = {
        'project_root': None,
        'data_yaml': None,
        'train_images': None,
        'nc': None,
        'names': None,
        'model_type': 'detect',
        'yolo_version': None,  # v5, v8, v9, v10
        'recommended_opset': 17  # default
    }

    if not os.path.exists(pt_path):
        return result

    pt_dir = os.path.dirname(pt_path)
    full_path_lower = pt_path.lower()

    # === Deep PT file analysis for YOLO version ===
    # This analyzes the actual model content, not just filename
    detected_version = detect_yolo_version_from_pt(pt_path)
    if detected_version:
        result['yolo_version'] = detected_version
    else:
        # Fallback: detect from path patterns
        if 'yolov5' in full_path_lower:
            result['yolo_version'] = 'v5'
        elif 'yolov10' in full_path_lower:
            result['yolo_version'] = 'v10'
        elif 'yolov9' in full_path_lower:
            result['yolo_version'] = 'v9'
        elif 'yolov8' in full_path_lower:
            result['yolo_version'] = 'v8'

    # === Deep PT file analysis for task type (detect/segment/classify) ===
    # This analyzes the actual model class, not just filename
    detected_task = detect_yolo_task_from_pt(pt_path)
    result['model_type'] = detected_task

    # === Get recommended opset based on version and task ===
    result['recommended_opset'] = get_recommended_opset(
        result['yolo_version'],
        result['model_type']
    )

    # === Find project root (look for data.yaml) ===
    # Case 1: PT is in models/pt/
    if pt_dir.endswith('/pt') or pt_dir.endswith('\\pt'):
        project_root = os.path.dirname(os.path.dirname(pt_dir))
    # Case 2: PT is in models/
    elif os.path.basename(pt_dir) == 'models':
        project_root = os.path.dirname(pt_dir)
    # Case 3: PT is in project root
    else:
        project_root = pt_dir

    # Verify project structure
    data_yaml = os.path.join(project_root, 'data.yaml')
    if os.path.exists(data_yaml):
        result['project_root'] = project_root
        result['data_yaml'] = data_yaml

        # Parse data.yaml
        data = parse_yolo_data_yaml(data_yaml)
        if 'nc' in data:
            result['nc'] = data['nc']
        if 'names' in data:
            result['names'] = data['names']

        # Find train/images for calibration
        train_images = os.path.join(project_root, 'train', 'images')
        if os.path.exists(train_images):
            result['train_images'] = train_images

    return result
