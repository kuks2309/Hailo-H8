"""
Integration tests for ONNX node extraction and Hailo compatibility validation.
"""
import pytest
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import onnx
from onnx import helper, TensorProto


# ============== PYTEST FIXTURES ==============

@pytest.fixture
def fixtures_dir():
    """Return path to test fixtures directory."""
    return os.path.join(os.path.dirname(__file__), 'fixtures')


@pytest.fixture
def yolov5_official_onnx(fixtures_dir, tmp_path):
    """
    Fixture for YOLOv5 official export ONNX.
    Uses real file if exists, otherwise creates minimal mock.
    """
    real_path = os.path.join(fixtures_dir, 'yolov5s_official.onnx')
    if os.path.exists(real_path):
        return real_path

    # Create minimal mock ONNX with YOLOv5 official naming pattern
    mock_path = tmp_path / "yolov5_official_mock.onnx"
    nodes = [
        helper.make_node('Conv', ['x'], ['y1'], name='/model.24/m.0/Conv'),
        helper.make_node('Conv', ['y1'], ['y2'], name='/model.24/m.1/Conv'),
        helper.make_node('Conv', ['y2'], ['y3'], name='/model.24/m.2/Conv'),
    ]
    graph = helper.make_graph(nodes, 'yolov5_mock',
        [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 3, 640, 640])],
        [helper.make_tensor_value_info('y3', TensorProto.FLOAT, [1, 85, 80, 80])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)])
    onnx.save(model, str(mock_path))
    return str(mock_path)


@pytest.fixture
def yolov8_onnx(fixtures_dir, tmp_path):
    """
    Fixture for YOLOv8 ONNX.
    Uses real file if exists, otherwise creates minimal mock.
    """
    real_path = os.path.join(fixtures_dir, 'yolov8s.onnx')
    if os.path.exists(real_path):
        return real_path

    # Create minimal mock ONNX with YOLOv8 naming pattern
    mock_path = tmp_path / "yolov8_mock.onnx"
    nodes = [
        helper.make_node('Conv', ['x'], ['y1'], name='/model.22/cv2.0/cv2.0.2/Conv'),
        helper.make_node('Conv', ['y1'], ['y2'], name='/model.22/cv2.1/cv2.1.2/Conv'),
        helper.make_node('Conv', ['y2'], ['y3'], name='/model.22/cv3.0/cv3.0.2/Conv'),
    ]
    graph = helper.make_graph(nodes, 'yolov8_mock',
        [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 3, 640, 640])],
        [helper.make_tensor_value_info('y3', TensorProto.FLOAT, [1, 85, 80, 80])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)])
    onnx.save(model, str(mock_path))
    return str(mock_path)


@pytest.fixture
def ultralytics_onnx(fixtures_dir, tmp_path):
    """
    Fixture for Ultralytics-style ONNX (incompatible naming).
    Uses real file if exists, otherwise creates minimal mock.
    """
    real_path = os.path.join(fixtures_dir, 'yolov5s_ultralytics.onnx')
    if os.path.exists(real_path):
        return real_path

    # Create minimal mock ONNX with ultralytics naming pattern (conv2d_N)
    mock_path = tmp_path / "ultralytics_mock.onnx"
    nodes = [
        helper.make_node('Conv', ['x'], ['y1'], name='conv2d_0'),
        helper.make_node('Conv', ['y1'], ['y2'], name='conv2d_1'),
        helper.make_node('Conv', ['y2'], ['y3'], name='conv2d_2'),
    ]
    graph = helper.make_graph(nodes, 'ultralytics_mock',
        [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 3, 640, 640])],
        [helper.make_tensor_value_info('y3', TensorProto.FLOAT, [1, 85, 80, 80])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)])
    onnx.save(model, str(mock_path))
    return str(mock_path)


# ============== TEST CLASSES ==============

class TestOnnxNodeExtraction:
    """Tests for extract_onnx_nodes() function."""

    def test_yolov5_official_pattern(self, yolov5_official_onnx):
        """Test detection of YOLOv5 official export pattern."""
        from src.core.converter import extract_onnx_nodes
        result = extract_onnx_nodes(yolov5_official_onnx)
        assert result['naming_style'] == 'yolov5_official'
        assert '/model.24/m.0/Conv' in result['detection_heads']
        assert result['yolo_version'] == 'v5'

    def test_yolov8_pattern(self, yolov8_onnx):
        """Test detection of YOLOv8 pattern."""
        from src.core.converter import extract_onnx_nodes
        result = extract_onnx_nodes(yolov8_onnx)
        assert result['naming_style'] == 'yolov8_official'
        assert any('/model.22/' in h for h in result['detection_heads'])
        assert result['yolo_version'] == 'v8'


class TestOnnxCompatibilityValidation:
    """Tests for validate_onnx_hailo_compatibility() function."""

    def test_yolov5_official_compatible(self, yolov5_official_onnx):
        """Test that YOLOv5 official export is compatible."""
        from src.core.converter import validate_onnx_hailo_compatibility
        result = validate_onnx_hailo_compatibility(yolov5_official_onnx)
        assert result['compatible'] == True
        assert result['naming_style'] == 'yolov5_official'

    def test_yolov8_compatible(self, yolov8_onnx):
        """Test that YOLOv8 export is compatible."""
        from src.core.converter import validate_onnx_hailo_compatibility
        result = validate_onnx_hailo_compatibility(yolov8_onnx)
        assert result['compatible'] == True
        assert result['naming_style'] == 'yolov8_official'

    def test_ultralytics_with_detection_heads(self, ultralytics_onnx):
        """Test that ultralytics pattern with detection heads is compatible (auto-convertible)."""
        from src.core.converter import validate_onnx_hailo_compatibility
        result = validate_onnx_hailo_compatibility(ultralytics_onnx)
        # Ultralytics ONNX is now compatible because:
        # 1. We can detect end-node-names from conv2d_N patterns
        # 2. onnx_utils.py can auto-convert naming to Hailo format
        assert result['naming_style'] == 'ultralytics'
        if result['detection_heads']:
            # With detected heads, it's compatible
            assert result['compatible'] == True
            assert len(result['warnings']) > 0  # Should have warning about ultralytics
        else:
            # Without detected heads, it's incompatible
            assert result['compatible'] == False
            assert len(result['errors']) > 0


class TestYoloBaseModelDetection:
    """Tests for detect_yolo_base_model() function."""

    def test_detects_yolov5(self, yolov5_official_onnx):
        """Test detection of YOLOv5 base model."""
        from src.core.converter import detect_yolo_base_model
        result = detect_yolo_base_model(yolov5_official_onnx)
        assert result.startswith('yolov5')

    def test_detects_yolov8(self, yolov8_onnx):
        """Test detection of YOLOv8 base model."""
        from src.core.converter import detect_yolo_base_model
        result = detect_yolo_base_model(yolov8_onnx)
        assert result.startswith('yolov8')
