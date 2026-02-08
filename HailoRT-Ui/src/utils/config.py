"""
Configuration Management
"""

import os
import yaml
from typing import Any, Dict


class Config:
    """Application configuration manager."""

    _instance = None
    _config: Dict[str, Any] = {}

    def __new__(cls, config_path: str = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            if config_path:
                cls._instance.load(config_path)
        return cls._instance

    def load(self, config_path: str):
        """Load configuration from YAML file."""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f) or {}
        else:
            self._config = self._default_config()
            self.save(config_path)

    def save(self, config_path: str):
        """Save configuration to YAML file."""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports dot notation)."""
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value: Any):
        """Set configuration value by key (supports dot notation)."""
        keys = key.split('.')
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    @staticmethod
    def _default_config() -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'paths': {
                'models': 'data/models',
                'calibration': 'data/calibration/images',
                'output': 'data/output',
            },
            'inference': {
                'confidence_threshold': 0.5,
                'iou_threshold': 0.45,
                'max_detections': 100,
            },
            'display': {
                'show_bboxes': True,
                'show_labels': True,
                'show_confidence': True,
                'show_fps': True,
            },
            'device': {
                'target': 'hailo8',
                'auto_connect': False,
            },
            'conversion': {
                'default_opset': 17,
                'available_opsets': [11, 12, 13, 17, 18],
                'default_batch_size': 1,
                'default_input_size': [640, 640],
                'available_targets': ['hailo8', 'hailo8l', 'hailo15h'],
                'default_target': 'hailo8',
                'model_types': ['detect', 'segment', 'classify'],
                'default_model_type': 'detect',
                'model_names': ['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l'],
                'default_model_name': 'yolov5s',
                'default_num_classes': 80,
                'calibration': {
                    'max_images': 500,
                    'layout': 'NHWC',
                    'data_format': 'float32',
                },
            },
            'file_extensions': {
                'pytorch': ['.pt', '.pth'],
                'onnx': ['.onnx'],
                'hef': ['.hef'],
                'images': ['.jpg', '.jpeg', '.png', '.bmp'],
                'video': ['.mp4', '.avi', '.mov', '.mkv'],
            },
        }
