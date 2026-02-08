"""Shared utility functions for HailoRT-Ui."""

import os
import random

from utils.constants import (
    TEMP_NORMAL_THRESHOLD,
    TEMP_WARN_THRESHOLD,
    TEMP_COLOR_NORMAL,
    TEMP_COLOR_WARNING,
    TEMP_COLOR_CRITICAL,
)


def get_temperature_color(temp: float) -> tuple:
    """Get color based on temperature value.

    Args:
        temp: Temperature in Celsius

    Returns:
        tuple: (color_hex, status_text)
    """
    if temp < TEMP_NORMAL_THRESHOLD:
        return (TEMP_COLOR_NORMAL, "Normal")
    elif temp < TEMP_WARN_THRESHOLD:
        return (TEMP_COLOR_WARNING, "Warm")
    else:
        return (TEMP_COLOR_CRITICAL, "Hot!")


def get_data_path(base_path: str, *parts: str) -> str:
    """Construct path relative to data directory.

    Args:
        base_path: Base directory path
        *parts: Additional path components

    Returns:
        str: Full path to data directory
    """
    return os.path.join(base_path, 'data', *parts)


def get_models_path(base_path: str, model_type: str = '') -> str:
    """Get models directory path.

    Args:
        base_path: Base directory path
        model_type: Optional model type subdirectory

    Returns:
        str: Full path to models directory
    """
    return get_data_path(base_path, 'models', model_type)


def generate_mock_device_info() -> dict:
    """Generate mock device info for testing without hardware.

    Returns:
        dict: Mock device information
    """
    return {
        'device_name': 'Hailo-8 (Mock)',
        'architecture': 'hailo8',
        'serial': 'MOCK-12345678',
        'firmware': '4.23.0',
        'driver': '4.23.0',
        'temperature': 45 + random.uniform(-2, 5),
        'power': 2.5 + random.uniform(-0.5, 1.0),
        'utilization': random.uniform(0, 30)
    }
