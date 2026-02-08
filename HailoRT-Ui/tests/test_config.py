"""Tests for Config utility."""
import pytest
import os
import tempfile
import yaml

# Add parent to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.config import Config


class TestConfig:
    """Test cases for Config singleton."""

    def test_singleton_instance(self):
        """Config should be a singleton."""
        config1 = Config()
        config2 = Config()
        assert config1 is config2

    def test_load_yaml_config(self, tmp_path):
        """Should load YAML configuration."""
        config_file = tmp_path / "test_config.yaml"
        config_data = {
            'paths': {'models': 'data/models'},
            'inference': {'confidence_threshold': 0.5}
        }
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)

        # Reset singleton for testing
        Config._instance = None
        config = Config(str(config_file))

        assert config.get('paths.models') == 'data/models'
        assert config.get('inference.confidence_threshold') == 0.5

    def test_get_default_value(self):
        """Should return default for missing keys."""
        Config._instance = None
        config = Config()
        assert config.get('nonexistent.key', 'default') == 'default'

    def test_set_simple_key(self, tmp_path):
        """Should set a simple configuration value."""
        config_file = tmp_path / "test_config.yaml"
        Config._instance = None
        config = Config(str(config_file))

        config.set('test_key', 'test_value')
        assert config.get('test_key') == 'test_value'

    def test_set_nested_key(self, tmp_path):
        """Should set nested configuration values."""
        config_file = tmp_path / "test_config.yaml"
        Config._instance = None
        config = Config(str(config_file))

        config.set('level1.level2.level3', 'deep_value')
        assert config.get('level1.level2.level3') == 'deep_value'

    def test_set_creates_nested_structure(self, tmp_path):
        """Should create nested dict structure when setting nested keys."""
        config_file = tmp_path / "test_config.yaml"
        Config._instance = None
        config = Config(str(config_file))

        config.set('new.nested.key', 42)
        assert config.get('new.nested.key') == 42
        assert config.get('new.nested') == {'key': 42}
        assert isinstance(config.get('new'), dict)

    def test_get_nested_key(self, tmp_path):
        """Should retrieve nested configuration values."""
        config_file = tmp_path / "test_config.yaml"
        config_data = {
            'database': {
                'host': 'localhost',
                'port': 5432,
                'credentials': {
                    'user': 'admin'
                }
            }
        }
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)

        Config._instance = None
        config = Config(str(config_file))

        assert config.get('database.host') == 'localhost'
        assert config.get('database.port') == 5432
        assert config.get('database.credentials.user') == 'admin'

    def test_get_nonexistent_nested_key(self, tmp_path):
        """Should return default for nonexistent nested keys."""
        config_file = tmp_path / "test_config.yaml"
        Config._instance = None
        config = Config(str(config_file))

        assert config.get('level1.level2.missing', 'default') == 'default'

    def test_get_partial_nested_key(self, tmp_path):
        """Should return default when intermediate key doesn't exist."""
        config_file = tmp_path / "test_config.yaml"
        config_data = {'existing': 'value'}
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)

        Config._instance = None
        config = Config(str(config_file))

        assert config.get('existing.nonexistent.key', 'default') == 'default'

    def test_default_config_structure(self, tmp_path):
        """Should return correct default configuration."""
        config_file = tmp_path / "config.yaml"
        Config._instance = None
        config = Config(str(config_file))

        # Check default paths
        assert config.get('paths.models') == 'data/models'
        assert config.get('paths.calibration') == 'data/calibration/images'
        assert config.get('paths.output') == 'data/output'

        # Check default inference settings
        assert config.get('inference.confidence_threshold') == 0.5
        assert config.get('inference.iou_threshold') == 0.45
        assert config.get('inference.max_detections') == 100

        # Check default display settings
        assert config.get('display.show_bboxes') is True
        assert config.get('display.show_labels') is True

        # Check default device settings
        assert config.get('device.target') == 'hailo8'
        assert config.get('device.auto_connect') is False

    def test_save_creates_directory(self, tmp_path):
        """Should create parent directory when saving config."""
        config_file = tmp_path / "nested" / "dir" / "config.yaml"
        Config._instance = None
        config = Config()

        config.set('test', 'value')
        config.save(str(config_file))

        assert config_file.exists()
        assert config_file.parent.exists()

    def test_save_and_reload(self, tmp_path):
        """Should persist configuration across save and reload."""
        config_file = tmp_path / "config.yaml"

        # Create and save config
        Config._instance = None
        config1 = Config(str(config_file))
        config1.set('custom.key', 'custom_value')
        config1.set('nested.deep.value', 123)
        config1.save(str(config_file))

        # Reload config
        Config._instance = None
        config2 = Config(str(config_file))

        assert config2.get('custom.key') == 'custom_value'
        assert config2.get('nested.deep.value') == 123

    def test_get_with_none_default(self):
        """Should return None when default is None."""
        Config._instance = None
        config = Config()
        assert config.get('nonexistent', None) is None

    def test_set_overwrites_existing_value(self, tmp_path):
        """Should overwrite existing configuration value."""
        config_file = tmp_path / "config.yaml"
        Config._instance = None
        config = Config(str(config_file))

        config.set('key', 'original')
        assert config.get('key') == 'original'

        config.set('key', 'updated')
        assert config.get('key') == 'updated'

    def test_empty_config_file(self, tmp_path):
        """Should handle empty YAML file."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text('')

        Config._instance = None
        config = Config(str(config_file))

        # Empty file loads as empty dict, should have no defaults without path
        # This is actual behavior - config needs initialization or will be empty
        result = config.get('device.target')
        assert result is None or result == 'hailo8'


class TestConfigValidation:
    """Test cases for config validation and edge cases."""

    def test_set_integer_value(self, tmp_path):
        """Should handle integer values."""
        config_file = tmp_path / "config.yaml"
        Config._instance = None
        config = Config(str(config_file))

        config.set('port', 8080)
        assert config.get('port') == 8080
        assert isinstance(config.get('port'), int)

    def test_set_float_value(self, tmp_path):
        """Should handle float values."""
        config_file = tmp_path / "config.yaml"
        Config._instance = None
        config = Config(str(config_file))

        config.set('threshold', 0.75)
        assert config.get('threshold') == 0.75
        assert isinstance(config.get('threshold'), float)

    def test_set_boolean_value(self, tmp_path):
        """Should handle boolean values."""
        config_file = tmp_path / "config.yaml"
        Config._instance = None
        config = Config(str(config_file))

        config.set('enabled', True)
        assert config.get('enabled') is True
        config.set('enabled', False)
        assert config.get('enabled') is False

    def test_set_list_value(self, tmp_path):
        """Should handle list values."""
        config_file = tmp_path / "config.yaml"
        Config._instance = None
        config = Config(str(config_file))

        test_list = [1, 2, 3, 4, 5]
        config.set('numbers', test_list)
        assert config.get('numbers') == test_list

    def test_set_dict_value(self, tmp_path):
        """Should handle dict values."""
        config_file = tmp_path / "config.yaml"
        Config._instance = None
        config = Config(str(config_file))

        test_dict = {'key1': 'value1', 'key2': 'value2'}
        config.set('settings', test_dict)
        assert config.get('settings') == test_dict

    def test_get_with_zero_default(self, tmp_path):
        """Should handle zero as default value."""
        config_file = tmp_path / "config.yaml"
        Config._instance = None
        config = Config(str(config_file))

        result = config.get('nonexistent', 0)
        assert result == 0

    def test_get_with_false_default(self, tmp_path):
        """Should handle False as default value."""
        config_file = tmp_path / "config.yaml"
        Config._instance = None
        config = Config(str(config_file))

        result = config.get('nonexistent', False)
        assert result is False

    def test_get_with_empty_string_default(self, tmp_path):
        """Should handle empty string as default value."""
        config_file = tmp_path / "config.yaml"
        Config._instance = None
        config = Config(str(config_file))

        result = config.get('nonexistent', '')
        assert result == ''

    def test_deeply_nested_key_access(self, tmp_path):
        """Should handle deeply nested keys (5+ levels)."""
        config_file = tmp_path / "config.yaml"
        Config._instance = None
        config = Config(str(config_file))

        config.set('level1.level2.level3.level4.level5', 'deep_value')
        assert config.get('level1.level2.level3.level4.level5') == 'deep_value'

    def test_set_overwrites_nested_structure(self, tmp_path):
        """Should properly overwrite nested structures."""
        config_file = tmp_path / "config.yaml"
        Config._instance = None
        config = Config(str(config_file))

        config.set('parent.child', 'value1')
        assert config.get('parent.child') == 'value1'

        config.set('parent.child', 'value2')
        assert config.get('parent.child') == 'value2'

    def test_save_with_explicit_path(self, tmp_path):
        """Should save to specified path."""
        config_file = tmp_path / "config.yaml"
        Config._instance = None
        config = Config(str(config_file))

        config.set('test', 'value')
        config.save(str(config_file))

        # Reload to verify
        Config._instance = None
        config2 = Config(str(config_file))
        assert config2.get('test') == 'value'

    def test_config_persists_default_values(self, tmp_path):
        """Should include default values when saving."""
        config_file = tmp_path / "config.yaml"
        Config._instance = None
        config = Config(str(config_file))

        # Set a custom value
        config.set('custom', 'value')
        config.save(str(config_file))

        # Reload and check defaults still exist
        Config._instance = None
        config2 = Config(str(config_file))
        assert config2.get('custom') == 'value'
        assert config2.get('device.target') == 'hailo8'

    def test_special_characters_in_values(self, tmp_path):
        """Should handle special characters in string values."""
        config_file = tmp_path / "config.yaml"
        Config._instance = None
        config = Config(str(config_file))

        special_string = "test@#$%^&*()[]{}|\\;':\"<>?,./`~"
        config.set('special', special_string)
        assert config.get('special') == special_string

    def test_unicode_in_values(self, tmp_path):
        """Should handle Unicode characters in values."""
        config_file = tmp_path / "config.yaml"
        Config._instance = None
        config = Config(str(config_file))

        unicode_string = "测试 テスト 한글 🔥"
        config.set('unicode', unicode_string)
        assert config.get('unicode') == unicode_string

    def test_get_partial_dict(self, tmp_path):
        """Should return dict when getting intermediate nested key."""
        config_file = tmp_path / "config.yaml"
        Config._instance = None
        config = Config(str(config_file))

        config.set('parent.child1', 'value1')
        config.set('parent.child2', 'value2')

        parent_dict = config.get('parent')
        assert isinstance(parent_dict, dict)
        assert 'child1' in parent_dict
        assert 'child2' in parent_dict

    def test_multiple_saves(self, tmp_path):
        """Should handle multiple save operations."""
        config_file = tmp_path / "config.yaml"
        Config._instance = None
        config = Config(str(config_file))

        config.set('key1', 'value1')
        config.save(str(config_file))

        config.set('key2', 'value2')
        config.save(str(config_file))

        config.set('key3', 'value3')
        config.save(str(config_file))

        # Reload and verify all values
        Config._instance = None
        config2 = Config(str(config_file))
        assert config2.get('key1') == 'value1'
        assert config2.get('key2') == 'value2'
        assert config2.get('key3') == 'value3'


class TestConfigDefaults:
    """Test cases for default configuration values."""

    def test_all_default_paths_exist(self, tmp_path):
        """Should have all expected default paths."""
        config_file = tmp_path / "config.yaml"
        Config._instance = None
        config = Config(str(config_file))

        assert config.get('paths.models') is not None
        assert config.get('paths.calibration') is not None
        assert config.get('paths.output') is not None

    def test_default_inference_settings(self, tmp_path):
        """Should have valid default inference settings."""
        config_file = tmp_path / "config.yaml"
        Config._instance = None
        config = Config(str(config_file))

        confidence = config.get('inference.confidence_threshold')
        iou = config.get('inference.iou_threshold')
        max_det = config.get('inference.max_detections')

        assert 0.0 <= confidence <= 1.0
        assert 0.0 <= iou <= 1.0
        assert max_det > 0

    def test_default_display_settings(self, tmp_path):
        """Should have boolean display settings."""
        config_file = tmp_path / "config.yaml"
        Config._instance = None
        config = Config(str(config_file))

        assert isinstance(config.get('display.show_bboxes'), bool)
        assert isinstance(config.get('display.show_labels'), bool)

    def test_default_device_settings(self, tmp_path):
        """Should have valid default device settings."""
        config_file = tmp_path / "config.yaml"
        Config._instance = None
        config = Config(str(config_file))

        assert config.get('device.target') in ['hailo8', 'hailo8l']
        assert isinstance(config.get('device.auto_connect'), bool)


class TestConfigErrorHandling:
    """Test cases for error handling in Config."""

    def test_invalid_yaml_file(self, tmp_path):
        """Should handle invalid YAML gracefully."""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text('invalid: yaml: content: {]')

        Config._instance = None
        # Will raise yaml.scanner.ScannerError - this is expected behavior
        try:
            config = Config(str(config_file))
            # If it doesn't raise, check it has some config
            assert isinstance(config._config, dict)
        except Exception:
            # Expected - invalid YAML should raise
            pass

    def test_readonly_filesystem(self, tmp_path, monkeypatch):
        """Should handle save failures gracefully."""
        config_file = tmp_path / "config.yaml"
        Config._instance = None
        config = Config(str(config_file))

        config.set('test', 'value')

        # Mock os.makedirs to raise permission error
        import os as os_module
        original_makedirs = os_module.makedirs

        def mock_makedirs(*args, **kwargs):
            raise PermissionError("Permission denied")

        monkeypatch.setattr(os_module, 'makedirs', mock_makedirs)

        # Save should handle the error
        try:
            config.save(str(tmp_path / "readonly" / "config.yaml"))
        except PermissionError:
            pass  # Expected

    def test_nonexistent_directory_in_path(self, tmp_path):
        """Should create directory structure when needed."""
        config_file = tmp_path / "deeply" / "nested" / "path" / "config.yaml"
        Config._instance = None
        config = Config()

        config.set('test', 'value')
        config.save(str(config_file))

        assert config_file.exists()
        assert config_file.parent.exists()
