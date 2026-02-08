"""
Environment checker for Hailo-Compiler-UI.
Verifies required packages are installed.
"""

import importlib
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class PackageInfo:
    """Information about a package."""
    name: str
    installed: bool
    version: Optional[str]
    required: bool
    install_cmd: str


# Package definitions
REQUIRED_PACKAGES = {
    'torch': {
        'import_name': 'torch',
        'required': True,
        'install_cmd': 'pip install torch torchvision'
    },
    'numpy': {
        'import_name': 'numpy',
        'required': True,
        'install_cmd': 'pip install numpy'
    },
    'PIL': {
        'import_name': 'PIL',
        'display_name': 'Pillow',
        'required': True,
        'install_cmd': 'pip install Pillow'
    },
    'hailo_sdk_client': {
        'import_name': 'hailo_sdk_client',
        'required': False,  # Only needed for HEF compilation
        'install_cmd': 'pip install hailo_dataflow_compiler-*.whl'
    },
    'ultralytics': {
        'import_name': 'ultralytics',
        'required': False,  # Optional for YOLO auto-detection
        'install_cmd': 'pip install ultralytics'
    },
    'onnx': {
        'import_name': 'onnx',
        'required': False,  # Optional for ONNX verification
        'install_cmd': 'pip install onnx'
    },
}


def check_package(import_name: str) -> tuple:
    """
    Check if a package is installed and get its version.

    Returns:
        (installed: bool, version: str or None)
    """
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except ImportError:
        return False, None


def validate_onnx_version() -> dict:
    """Validate ONNX package version for Hailo SDK compatibility."""
    try:
        import onnx
        version = onnx.__version__
        major_minor = tuple(map(int, version.split('.')[:2]))

        if major_minor >= (1, 17):
            return {
                'ok': False,
                'version': version,
                'warning': f"onnx {version} incompatible with hailo_sdk_client. Required: onnx==1.16.0. Run: pip install onnx==1.16.0"
            }
        return {'ok': True, 'version': version, 'warning': None}
    except ImportError:
        return {'ok': False, 'version': None, 'warning': "onnx not installed"}


def check_environment() -> Dict:
    """
    Check all required and optional packages.

    Returns:
        {
            'all_required_ok': bool,
            'packages': {
                'torch': PackageInfo(...),
                ...
            },
            'missing_required': ['package_name', ...],
            'missing_optional': ['package_name', ...],
            'onnx_validation': {
                'ok': bool,
                'version': str or None,
                'warning': str or None
            }
        }
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
        display_name = pkg_config.get('display_name', pkg_key)
        required = pkg_config['required']
        install_cmd = pkg_config['install_cmd']

        installed, version = check_package(import_name)

        pkg_info = PackageInfo(
            name=display_name,
            installed=installed,
            version=version,
            required=required,
            install_cmd=install_cmd
        )

        result['packages'][pkg_key] = pkg_info

        if not installed:
            if required:
                result['missing_required'].append(display_name)
                result['all_required_ok'] = False
            else:
                result['missing_optional'].append(display_name)

    return result


def get_missing_packages_message(env_result: Dict) -> str:
    """
    Generate a user-friendly message about missing packages.
    """
    lines = []

    if env_result['missing_required']:
        lines.append("Required packages missing:")
        for pkg_name in env_result['missing_required']:
            for key, info in env_result['packages'].items():
                if info.name == pkg_name:
                    lines.append(f"  - {pkg_name}: {info.install_cmd}")
                    break
        lines.append("")

    if env_result['missing_optional']:
        lines.append("Optional packages missing (some features disabled):")
        for pkg_name in env_result['missing_optional']:
            for key, info in env_result['packages'].items():
                if info.name == pkg_name:
                    lines.append(f"  - {pkg_name}: {info.install_cmd}")
                    break

    return "\n".join(lines)


def can_compile_hef() -> bool:
    """Check if HEF compilation is available."""
    installed, _ = check_package('hailo_sdk_client')
    return installed


def can_detect_yolo() -> bool:
    """Check if YOLO auto-detection is available."""
    installed, _ = check_package('ultralytics')
    return installed


if __name__ == '__main__':
    # Test environment check
    result = check_environment()
    print("Environment Check Result:")
    print(f"  All required OK: {result['all_required_ok']}")
    print("\nPackages:")
    for key, info in result['packages'].items():
        status = "✓" if info.installed else "✗"
        version = info.version or "N/A"
        req = "(required)" if info.required else "(optional)"
        print(f"  {status} {info.name}: {version} {req}")

    if not result['all_required_ok']:
        print("\n" + get_missing_packages_message(result))
