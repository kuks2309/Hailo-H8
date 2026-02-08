"""
Folder structure management for Hailo compilation projects.
Validates and creates the required directory structure.
"""

import os
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class FolderInfo:
    """Information about a required folder."""
    path: str
    description: str
    required: bool = True


# Required folder structure for Hailo compilation
REQUIRED_FOLDERS = [
    FolderInfo("models/onnx", "Original ONNX model files"),
    FolderInfo("models/har", "Hailo Archive files (intermediate)"),
    FolderInfo("models/hef", "Hailo Executable Format (final)"),
    FolderInfo("configs", "Hailo configuration files (.alls, .yaml)", required=False),
    FolderInfo("scripts", "Compilation and inference scripts", required=False),
    FolderInfo("train/images", "Training images (used for calibration)", required=False),
    FolderInfo("valid/images", "Validation images (used for calibration)", required=False),
    FolderInfo("test/images", "Test images for inference", required=False),
    FolderInfo("test/videos", "Test videos for inference", required=False),
    FolderInfo("test/results", "Inference output results", required=False),
    FolderInfo("logs", "Compilation and inference logs", required=False),
]

# Calibration data source folders (in priority order)
CALIBRATION_FOLDERS = ["train/images", "valid/images", "test/images"]


def find_calibration_folder(base_path: str) -> Tuple[str, bool]:
    """
    Find the best available calibration folder.

    Checks train/images, valid/images, test/images in priority order.

    Args:
        base_path: Root directory to search

    Returns:
        Tuple of (folder_path, exists)
    """
    for folder in CALIBRATION_FOLDERS:
        full_path = os.path.join(base_path, folder)
        if os.path.isdir(full_path):
            # Check if folder has images
            images = [f for f in os.listdir(full_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            if images:
                return full_path, True

    # Return first option as default (even if doesn't exist)
    return os.path.join(base_path, CALIBRATION_FOLDERS[0]), False


def validate_folder_structure(base_path: str) -> Tuple[bool, Dict[str, bool]]:
    """
    Validate if the folder structure exists.

    Args:
        base_path: Root directory to validate

    Returns:
        Tuple of (all_required_exist, {folder_path: exists})
    """
    results = {}
    all_required_ok = True

    for folder_info in REQUIRED_FOLDERS:
        full_path = os.path.join(base_path, folder_info.path)
        exists = os.path.isdir(full_path)
        results[folder_info.path] = exists

        if folder_info.required and not exists:
            all_required_ok = False

    # Check calibration source (at least one of train/valid/test images)
    calib_path, calib_exists = find_calibration_folder(base_path)
    results['calibration_source'] = calib_exists
    if not calib_exists:
        all_required_ok = False

    return all_required_ok, results


def get_missing_folders(base_path: str, required_only: bool = False) -> List[FolderInfo]:
    """
    Get list of missing folders.

    Args:
        base_path: Root directory to check
        required_only: If True, only return missing required folders

    Returns:
        List of FolderInfo for missing folders
    """
    missing = []

    for folder_info in REQUIRED_FOLDERS:
        if required_only and not folder_info.required:
            continue

        full_path = os.path.join(base_path, folder_info.path)
        if not os.path.isdir(full_path):
            missing.append(folder_info)

    return missing


def create_folder_structure(base_path: str, folders: List[FolderInfo] = None) -> Tuple[bool, List[str]]:
    """
    Create the folder structure.

    Args:
        base_path: Root directory where to create structure
        folders: List of folders to create (default: all required folders)

    Returns:
        Tuple of (success, list of created folders)
    """
    if folders is None:
        folders = REQUIRED_FOLDERS

    created = []

    try:
        for folder_info in folders:
            full_path = os.path.join(base_path, folder_info.path)
            if not os.path.exists(full_path):
                os.makedirs(full_path, exist_ok=True)
                created.append(folder_info.path)

        # Create README.md
        readme_path = os.path.join(base_path, "README.md")
        if not os.path.exists(readme_path):
            _create_readme(base_path)
            created.append("README.md")

        return True, created

    except Exception as e:
        return False, [str(e)]


def _create_readme(base_path: str):
    """Create a README.md file with folder structure documentation."""
    project_name = os.path.basename(base_path)

    content = f"""# {project_name} - Hailo-H8 Compilation Project

## Folder Structure

```
{project_name}/
├── train/
│   ├── images/          # Training images (used for calibration, 100-1000 images)
│   └── labels/          # Training labels
├── valid/
│   ├── images/          # Validation images (alternative calibration source)
│   └── labels/          # Validation labels
├── test/
│   ├── images/          # Test images for inference
│   ├── videos/          # Test videos for inference
│   └── results/         # Inference output results
├── models/
│   ├── onnx/            # Original ONNX model files
│   ├── har/             # Hailo Archive files (intermediate)
│   └── hef/             # Hailo Executable Format (final)
├── configs/             # Hailo configuration files (.alls, .yaml)
├── scripts/             # Compilation and inference scripts
└── logs/                # Compilation and inference logs
```

## Calibration Data Sources

Calibration images are searched in this priority order:
1. **train/images/** - Primary source (recommended)
2. **valid/images/** - Secondary source
3. **test/images/** - Fallback source

Add 100-1000 representative images for best quantization results.

## Data Preparation Checklist

1. **train/images/**: Add 100-1000 representative images for calibration
2. **models/onnx/**: Place your exported ONNX model here
3. **test/images/**: Add test images for validation

## Hailo Compilation Workflow

1. Parse ONNX → HAR
2. Optimize HAR with calibration data (from train/valid/test images)
3. Compile HAR → HEF
4. Test HEF on device
"""

    with open(os.path.join(base_path, "README.md"), 'w', encoding='utf-8') as f:
        f.write(content)


def get_project_paths(base_path: str) -> Dict[str, str]:
    """
    Get all relevant paths for a project.

    Args:
        base_path: Root directory of the project

    Returns:
        Dictionary of path names to full paths
    """
    # Find best calibration folder
    calib_path, _ = find_calibration_folder(base_path)

    return {
        'base': base_path,
        'calibration': calib_path,
        'train_images': os.path.join(base_path, 'train', 'images'),
        'valid_images': os.path.join(base_path, 'valid', 'images'),
        'onnx': os.path.join(base_path, 'models', 'onnx'),
        'har': os.path.join(base_path, 'models', 'har'),
        'hef': os.path.join(base_path, 'models', 'hef'),
        'configs': os.path.join(base_path, 'configs'),
        'scripts': os.path.join(base_path, 'scripts'),
        'test_images': os.path.join(base_path, 'test', 'images'),
        'test_videos': os.path.join(base_path, 'test', 'videos'),
        'test_results': os.path.join(base_path, 'test', 'results'),
        'logs': os.path.join(base_path, 'logs'),
    }
