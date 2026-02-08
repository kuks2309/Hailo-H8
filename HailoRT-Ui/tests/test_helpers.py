"""Tests for utils/helpers.py"""
import pytest
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.helpers import (
    get_temperature_color,
    get_data_path,
    get_models_path,
    generate_mock_device_info
)


class TestGetTemperatureColor:
    """Tests for get_temperature_color function."""

    def test_normal_temperature(self):
        """Temperature below 50 should be green/Normal."""
        color, status = get_temperature_color(30.0)
        assert color == "#4caf50"
        assert status == "Normal"

    def test_warm_temperature(self):
        """Temperature 50-69 should be orange/Warm."""
        color, status = get_temperature_color(55.0)
        assert color == "#ff9800"
        assert status == "Warm"

    def test_hot_temperature(self):
        """Temperature 70+ should be red/Hot!."""
        color, status = get_temperature_color(75.0)
        assert color == "#f44336"
        assert status == "Hot!"

    def test_boundary_50(self):
        """Temperature exactly 50 should be Warm."""
        color, status = get_temperature_color(50.0)
        assert status == "Warm"
        assert color == "#ff9800"

    def test_boundary_70(self):
        """Temperature exactly 70 should be Hot!."""
        color, status = get_temperature_color(70.0)
        assert status == "Hot!"
        assert color == "#f44336"

    def test_boundary_49(self):
        """Temperature 49.9 should be Normal."""
        color, status = get_temperature_color(49.9)
        assert status == "Normal"
        assert color == "#4caf50"

    def test_boundary_69(self):
        """Temperature 69.9 should be Warm."""
        color, status = get_temperature_color(69.9)
        assert status == "Warm"
        assert color == "#ff9800"

    def test_zero_temperature(self):
        """Temperature 0 should be Normal."""
        color, status = get_temperature_color(0)
        assert status == "Normal"
        assert color == "#4caf50"

    def test_negative_temperature(self):
        """Negative temperature should be Normal."""
        color, status = get_temperature_color(-10.0)
        assert status == "Normal"
        assert color == "#4caf50"

    def test_extreme_temperature(self):
        """Extremely high temperature should be Hot!."""
        color, status = get_temperature_color(120.0)
        assert status == "Hot!"
        assert color == "#f44336"


class TestGetDataPath:
    """Tests for get_data_path function."""

    def test_returns_string(self):
        """Should return a string."""
        result = get_data_path("/home/user/project")
        assert isinstance(result, str)

    def test_path_contains_data(self):
        """Path should contain 'data' directory."""
        result = get_data_path("/home/user/project")
        assert "data" in result

    def test_with_single_part(self):
        """Should join base_path, data, and single part."""
        result = get_data_path("/home/user/project", "models")
        expected = os.path.join("/home/user/project", "data", "models")
        assert result == expected

    def test_with_multiple_parts(self):
        """Should join base_path, data, and multiple parts."""
        result = get_data_path("/home/user/project", "models", "yolov8")
        expected = os.path.join("/home/user/project", "data", "models", "yolov8")
        assert result == expected

    def test_with_no_additional_parts(self):
        """Should return just base_path/data when no parts given."""
        result = get_data_path("/home/user/project")
        expected = os.path.join("/home/user/project", "data")
        assert result == expected

    def test_handles_trailing_slash(self):
        """Should handle base_path with trailing slash."""
        result = get_data_path("/home/user/project/", "models")
        assert "data" in result
        assert "models" in result


class TestGetModelsPath:
    """Tests for get_models_path function."""

    def test_returns_string(self):
        """Should return a string."""
        result = get_models_path("/home/user/project")
        assert isinstance(result, str)

    def test_contains_models_directory(self):
        """Path should contain 'models' directory."""
        result = get_models_path("/home/user/project")
        assert "models" in result

    def test_with_model_type(self):
        """Should include model_type subdirectory."""
        result = get_models_path("/home/user/project", "yolov8")
        expected = os.path.join("/home/user/project", "data", "models", "yolov8")
        assert result == expected

    def test_without_model_type(self):
        """Should return models directory without subdirectory."""
        result = get_models_path("/home/user/project")
        expected = os.path.join("/home/user/project", "data", "models", "")
        assert result == expected

    def test_empty_model_type(self):
        """Should handle empty string model_type."""
        result = get_models_path("/home/user/project", "")
        expected = os.path.join("/home/user/project", "data", "models", "")
        assert result == expected


class TestGenerateMockDeviceInfo:
    """Tests for generate_mock_device_info function."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        result = generate_mock_device_info()
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        """Should have all required keys."""
        result = generate_mock_device_info()
        required_keys = ['device_name', 'architecture', 'serial', 'firmware',
                         'driver', 'temperature', 'power', 'utilization']
        for key in required_keys:
            assert key in result

    def test_device_name(self):
        """Device name should indicate mock device."""
        result = generate_mock_device_info()
        assert "Mock" in result["device_name"]

    def test_architecture(self):
        """Architecture should be hailo8."""
        result = generate_mock_device_info()
        assert result["architecture"] == "hailo8"

    def test_serial_format(self):
        """Serial should contain MOCK prefix."""
        result = generate_mock_device_info()
        assert result["serial"].startswith("MOCK")

    def test_temperature_in_range(self):
        """Temperature should be in reasonable range (43-50)."""
        result = generate_mock_device_info()
        assert 43 <= result["temperature"] <= 50

    def test_power_in_range(self):
        """Power should be in reasonable range (2.0-3.5W)."""
        result = generate_mock_device_info()
        assert 2.0 <= result["power"] <= 3.5

    def test_utilization_percentage(self):
        """Utilization should be 0-30%."""
        result = generate_mock_device_info()
        assert 0 <= result["utilization"] <= 30

    def test_randomness(self):
        """Multiple calls should produce different values."""
        result1 = generate_mock_device_info()
        result2 = generate_mock_device_info()
        # At least one of temperature, power, or utilization should differ
        assert (result1["temperature"] != result2["temperature"] or
                result1["power"] != result2["power"] or
                result1["utilization"] != result2["utilization"])

    def test_firmware_version(self):
        """Firmware version should be present."""
        result = generate_mock_device_info()
        assert result["firmware"] == "4.23.0"

    def test_driver_version(self):
        """Driver version should be present."""
        result = generate_mock_device_info()
        assert result["driver"] == "4.23.0"


class TestPathHelpersIntegration:
    """Integration tests for path helper functions."""

    def test_get_models_path_uses_get_data_path(self):
        """get_models_path should use get_data_path internally."""
        base = '/test/base'
        model_type = 'pt'

        models_path = get_models_path(base, model_type)
        data_path = get_data_path(base, 'models', model_type)

        assert models_path == data_path

    def test_path_construction_consistency(self):
        """Path helpers should produce consistent results."""
        base = '/project/root'

        # These should produce the same path
        path1 = get_data_path(base, 'models', 'onnx')
        path2 = get_models_path(base, 'onnx')

        assert path1 == path2

    def test_nested_model_types(self):
        """Should handle nested model type paths."""
        base = '/base'
        result = get_models_path(base, 'pt/yolov8')
        assert 'models' in result
        assert 'pt' in result
        assert 'yolov8' in result


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_get_temperature_color_with_float_precision(self):
        """Should handle float precision correctly at boundaries."""
        # Test boundary with high precision
        color1, _ = get_temperature_color(49.999999)
        color2, _ = get_temperature_color(50.000001)

        assert color1 == "#4caf50"  # Normal
        assert color2 == "#ff9800"  # Warn

    def test_path_helpers_with_unicode(self):
        """Should handle Unicode characters in paths."""
        result = get_data_path('/base/测试', 'models')
        assert 'data' in result
        assert 'models' in result
        assert '测试' in result

    def test_path_helpers_with_spaces(self):
        """Should handle spaces in paths."""
        result = get_data_path('/base/my folder', 'models')
        assert 'data' in result
        assert 'models' in result
        assert 'my folder' in result

    def test_get_data_path_with_dots(self):
        """Should handle relative path components."""
        result = get_data_path('..', 'models')
        assert 'data' in result
        assert 'models' in result

    def test_mock_device_info_multiple_generations(self):
        """Should generate valid data consistently."""
        results = [generate_mock_device_info() for _ in range(100)]

        # All temperatures should be in valid range
        for result in results:
            assert 43 <= result['temperature'] <= 50
            assert 2.0 <= result['power'] <= 3.5
            assert 0 <= result['utilization'] <= 30

        # Static fields should be consistent
        for result in results:
            assert result['device_name'] == 'Hailo-8 (Mock)'
            assert result['architecture'] == 'hailo8'
            assert result['serial'] == 'MOCK-12345678'

    def test_temperature_color_return_type(self):
        """Should return tuple with exactly 2 elements."""
        result = get_temperature_color(50.0)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], str)

    def test_get_data_path_with_empty_parts(self):
        """Should handle empty string parts."""
        result = get_data_path('/base', '', 'models')
        assert 'data' in result
        assert 'models' in result

    def test_all_temperature_states(self):
        """Should cover all three temperature states."""
        # Normal
        color_n, status_n = get_temperature_color(25.0)
        assert status_n == "Normal"

        # Warm
        color_w, status_w = get_temperature_color(60.0)
        assert status_w == "Warm"

        # Hot
        color_h, status_h = get_temperature_color(80.0)
        assert status_h == "Hot!"

        # All should have different colors
        assert color_n != color_w
        assert color_w != color_h
        assert color_n != color_h
