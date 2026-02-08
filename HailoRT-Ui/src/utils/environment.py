"""
Environment checker for HailoRT-Ui.
Verifies required packages are installed and checks version compatibility.
"""

import importlib
from typing import Dict, Optional, Tuple

from utils.logger import setup_logger
from utils.exceptions import HailoEnvironmentError

logger = setup_logger(__name__)


# Package definitions for HailoRT-Ui
REQUIRED_PACKAGES = {
    'PyQt5': {'import_name': 'PyQt5', 'required': True, 'purpose': 'GUI framework'},
    'torch': {'import_name': 'torch', 'required': True, 'purpose': 'Model loading and export'},
    'numpy': {'import_name': 'numpy', 'required': True, 'purpose': 'Data processing'},
    'PIL': {'import_name': 'PIL', 'required': True, 'purpose': 'Image processing'},
    'cv2': {'import_name': 'cv2', 'required': True, 'purpose': 'Video capture and processing'},
    'hailo_platform': {'import_name': 'hailo_platform', 'required': False, 'purpose': 'Device communication'},
    'hailo_sdk_client': {'import_name': 'hailo_sdk_client', 'required': False, 'purpose': 'Model compilation'},
    'ultralytics': {'import_name': 'ultralytics', 'required': False, 'purpose': 'YOLOv8 model support'},
    'onnx': {'import_name': 'onnx', 'required': False, 'purpose': 'ONNX model manipulation'},
    'onnxsim': {'import_name': 'onnxsim', 'required': False, 'purpose': 'ONNX model simplification'},
}


def check_package(import_name: str) -> Tuple[bool, Optional[str]]:
    """
    Check if a package is installed and get its version.

    Args:
        import_name: The package import name (e.g., 'torch', 'cv2')

    Returns:
        Tuple of (installed: bool, version: str or None)
    """
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except ImportError:
        return False, None
    except Exception as e:
        logger.warning(f"Error checking package {import_name}: {e}")
        return False, None


def validate_onnx_version() -> dict:
    """
    Validate ONNX package version for Hailo SDK compatibility.

    ONNX >= 1.17 has known compatibility issues with Hailo SDK.
    Recommended version: onnx==1.16.0

    Returns:
        Dictionary with keys:
            - ok: bool - Whether version is compatible
            - version: str or None - ONNX version if installed
            - warning: str or None - Warning message if incompatible
    """
    try:
        import onnx
        version = onnx.__version__
        major_minor = tuple(map(int, version.split('.')[:2]))

        if major_minor >= (1, 17):
            return {
                'ok': False,
                'version': version,
                'warning': (
                    f"onnx {version} incompatible with hailo_sdk_client. "
                    f"Required: onnx==1.16.0. Run: pip install onnx==1.16.0"
                )
            }
        return {'ok': True, 'version': version, 'warning': None}
    except ImportError:
        return {'ok': False, 'version': None, 'warning': "onnx not installed"}
    except Exception as e:
        logger.warning(f"Error validating ONNX version: {e}")
        return {'ok': False, 'version': None, 'warning': f"Error checking ONNX: {e}"}


def check_environment() -> dict:
    """
    Check all required and optional packages.

    Returns:
        Dictionary with keys:
            - all_required_ok: bool - True if all required packages installed
            - packages: dict - Package name -> package info dict
            - missing_required: list - Names of missing required packages
            - missing_optional: list - Names of missing optional packages
            - onnx_validation: dict - ONNX version validation result
    """
    result = {
        'all_required_ok': True,
        'packages': {},
        'missing_required': [],
        'missing_optional': [],
        'onnx_validation': validate_onnx_version()
    }

    for pkg_key, pkg_config in REQUIRED_PACKAGES.items():
        import_name = pkg_config['import_name']
        required = pkg_config['required']
        purpose = pkg_config['purpose']

        installed, version = check_package(import_name)

        pkg_info = {
            'name': pkg_key,
            'import_name': import_name,
            'installed': installed,
            'version': version,
            'required': required,
            'purpose': purpose
        }

        result['packages'][pkg_key] = pkg_info

        if not installed:
            if required:
                result['missing_required'].append(pkg_key)
                result['all_required_ok'] = False
                logger.warning(f"Required package missing: {pkg_key} ({purpose})")
            else:
                result['missing_optional'].append(pkg_key)
                logger.info(f"Optional package missing: {pkg_key} ({purpose})")

    return result


def get_missing_packages_message(env_result: dict) -> str:
    """
    Generate a user-friendly message about missing packages.

    Args:
        env_result: Result dictionary from check_environment()

    Returns:
        Formatted string describing missing packages
    """
    lines = []

    if env_result['missing_required']:
        lines.append("Required packages missing:")
        for pkg_name in env_result['missing_required']:
            pkg_info = env_result['packages'].get(pkg_name, {})
            purpose = pkg_info.get('purpose', 'Unknown purpose')
            lines.append(f"  - {pkg_name}: {purpose}")
        lines.append("")
        lines.append("Install required packages:")
        lines.append("  pip install PyQt5 torch numpy Pillow opencv-python")
        lines.append("")

    if env_result['missing_optional']:
        lines.append("Optional packages missing (some features disabled):")
        for pkg_name in env_result['missing_optional']:
            pkg_info = env_result['packages'].get(pkg_name, {})
            purpose = pkg_info.get('purpose', 'Unknown purpose')
            lines.append(f"  - {pkg_name}: {purpose}")
        lines.append("")
        lines.append("To enable all features, install optional packages:")

        optional_installs = []
        if 'hailo_platform' in env_result['missing_optional']:
            optional_installs.append("  # HailoRT (device communication)")
            optional_installs.append("  pip install hailort-*.whl")
        if 'hailo_sdk_client' in env_result['missing_optional']:
            optional_installs.append("  # Hailo SDK (model compilation)")
            optional_installs.append("  pip install hailo_dataflow_compiler-*.whl")
        if 'ultralytics' in env_result['missing_optional']:
            optional_installs.append("  pip install ultralytics")
        if 'onnx' in env_result['missing_optional'] or 'onnxsim' in env_result['missing_optional']:
            optional_installs.append("  pip install onnx==1.16.0 onnxsim")

        lines.extend(optional_installs)

    # Add ONNX version warning if present
    onnx_validation = env_result.get('onnx_validation', {})
    if onnx_validation.get('warning'):
        lines.append("")
        lines.append("WARNING:")
        lines.append(f"  {onnx_validation['warning']}")

    return "\n".join(lines) if lines else "All packages installed correctly."


def can_compile_hef() -> bool:
    """
    Check if HEF compilation is available.

    Requires hailo_sdk_client package.

    Returns:
        True if hailo_sdk_client is installed
    """
    installed, _ = check_package('hailo_sdk_client')
    return installed


def can_use_device() -> bool:
    """
    Check if Hailo device communication is available.

    Requires hailo_platform package (HailoRT).

    Returns:
        True if hailo_platform is installed
    """
    installed, _ = check_package('hailo_platform')
    return installed


def can_detect_yolo() -> bool:
    """
    Check if YOLO auto-detection is available.

    Requires ultralytics package.

    Returns:
        True if ultralytics is installed
    """
    installed, _ = check_package('ultralytics')
    return installed


if __name__ == '__main__':
    # Test environment check
    print("=" * 60)
    print("HailoRT-Ui Environment Check")
    print("=" * 60)

    result = check_environment()

    print(f"\nAll required packages OK: {result['all_required_ok']}")

    print("\nPackages:")
    for key, info in result['packages'].items():
        status = "[OK]" if info['installed'] else "[MISSING]"
        version = info['version'] or "N/A"
        req_type = "(required)" if info['required'] else "(optional)"
        print(f"  {status} {key:20s} {version:15s} {req_type:12s} - {info['purpose']}")

    print("\nFeature availability:")
    print(f"  Device communication: {'YES' if can_use_device() else 'NO (mock mode)'}")
    print(f"  HEF compilation:      {'YES' if can_compile_hef() else 'NO'}")
    print(f"  YOLO auto-detection:  {'YES' if can_detect_yolo() else 'NO'}")

    # ONNX validation
    onnx_val = result['onnx_validation']
    if onnx_val['version']:
        onnx_status = "OK" if onnx_val['ok'] else "WARNING"
        print(f"\nONNX version: {onnx_val['version']} [{onnx_status}]")
        if onnx_val['warning']:
            print(f"  {onnx_val['warning']}")

    if not result['all_required_ok'] or result['missing_optional']:
        print("\n" + "=" * 60)
        print(get_missing_packages_message(result))
        print("=" * 60)
