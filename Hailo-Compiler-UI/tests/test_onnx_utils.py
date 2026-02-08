"""
Unit tests for onnx_utils module.
Tests ONNX patching, naming style detection, and auto-conversion.
"""
import pytest
import os
import sys
import tempfile
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import onnx
from onnx import helper, TensorProto, numpy_helper


# ============== PYTEST FIXTURES ==============

@pytest.fixture
def pytorch_generic_onnx(tmp_path):
    """Create ONNX with PyTorch generic naming (Conv_N pattern)."""
    # Create weights for Conv nodes
    weights = []
    nodes = []

    for i in range(60):  # Simulate YOLOv5 structure
        weight_name = f'conv{i}_weight'
        weight_data = np.random.randn(64, 3, 3, 3).astype(np.float32)
        weights.append(numpy_helper.from_array(weight_data, weight_name))

        inp = 'x' if i == 0 else f'y{i-1}'
        out = f'y{i}'
        nodes.append(helper.make_node(
            'Conv', [inp, weight_name], [out],
            name=f'Conv_{i}',
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1]
        ))

    graph = helper.make_graph(
        nodes, 'pytorch_generic_mock',
        [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 3, 640, 640])],
        [helper.make_tensor_value_info('y59', TensorProto.FLOAT, [1, 64, 640, 640])],
        weights
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])

    mock_path = tmp_path / "pytorch_generic.onnx"
    onnx.save(model, str(mock_path))
    return str(mock_path)


@pytest.fixture
def yolov5_official_onnx(tmp_path):
    """Create ONNX with YOLOv5 official naming pattern."""
    weights = []
    nodes = []

    # Create 60 Conv nodes with YOLOv5 naming
    yolov5_names = [
        '/model.0/conv/Conv', '/model.1/conv/Conv', '/model.2/cv1/conv/Conv',
    ]
    # Add backbone names
    for i in range(57):
        yolov5_names.append(f'/model.backbone/conv{i}/Conv')
    # Replace last 3 with detection heads
    yolov5_names[-3:] = [
        '/model.24/m.0/Conv', '/model.24/m.1/Conv', '/model.24/m.2/Conv'
    ]

    for i, name in enumerate(yolov5_names):
        weight_name = f'conv{i}_weight'
        weight_data = np.random.randn(64, 3, 3, 3).astype(np.float32)
        weights.append(numpy_helper.from_array(weight_data, weight_name))

        inp = 'x' if i == 0 else f'y{i-1}'
        out = f'y{i}'
        nodes.append(helper.make_node(
            'Conv', [inp, weight_name], [out],
            name=name,
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1]
        ))

    graph = helper.make_graph(
        nodes, 'yolov5_official_mock',
        [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 3, 640, 640])],
        [helper.make_tensor_value_info('y59', TensorProto.FLOAT, [1, 64, 640, 640])],
        weights
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)])

    mock_path = tmp_path / "yolov5_official.onnx"
    onnx.save(model, str(mock_path))
    return str(mock_path)


# ============== TEST CLASSES ==============

class TestNamingStyleDetection:
    """Tests for detect_onnx_naming_style function."""

    def test_detect_pytorch_generic(self, pytorch_generic_onnx):
        """Test detection of PyTorch generic naming."""
        from src.core.onnx_utils import detect_onnx_naming_style
        style = detect_onnx_naming_style(pytorch_generic_onnx)
        assert style == 'pytorch_generic'

    def test_detect_yolov5_official(self, yolov5_official_onnx):
        """Test detection of YOLOv5 official naming."""
        from src.core.onnx_utils import detect_onnx_naming_style
        style = detect_onnx_naming_style(yolov5_official_onnx)
        assert style == 'yolov5_official'

    def test_is_hailo_compatible(self, yolov5_official_onnx, pytorch_generic_onnx):
        """Test Hailo compatibility check."""
        from src.core.onnx_utils import is_hailo_compatible_naming
        assert is_hailo_compatible_naming(yolov5_official_onnx) == True
        assert is_hailo_compatible_naming(pytorch_generic_onnx) == False


class TestOnnxPatching:
    """Tests for ONNX patching functions."""

    def test_downgrade_opset_version(self, pytorch_generic_onnx):
        """Test opset version downgrade."""
        from src.core.onnx_utils import downgrade_opset_version

        model = onnx.load(pytorch_generic_onnx)
        assert model.opset_import[0].version == 17

        downgrade_opset_version(model, target_opset=11)
        assert model.opset_import[0].version == 11

    def test_patch_adds_kernel_shape(self, tmp_path):
        """Test that patching adds missing kernel_shape to Conv nodes."""
        from src.core.onnx_utils import patch_onnx_for_hailo

        # Create Conv node without kernel_shape attribute
        weight_data = np.random.randn(64, 3, 3, 3).astype(np.float32)
        weight = numpy_helper.from_array(weight_data, 'weight')

        node = helper.make_node(
            'Conv', ['x', 'weight'], ['y'],
            name='conv_no_kernel_shape'
            # No kernel_shape attribute
        )

        graph = helper.make_graph(
            [node], 'test',
            [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 3, 640, 640])],
            [helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 64, 638, 638])],
            [weight]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])

        # Verify no kernel_shape before patching
        conv_node = model.graph.node[0]
        has_kernel_shape_before = any(a.name == 'kernel_shape' for a in conv_node.attribute)
        assert has_kernel_shape_before == False

        # Apply patching
        patch_onnx_for_hailo(model)

        # Verify kernel_shape added after patching
        has_kernel_shape_after = any(a.name == 'kernel_shape' for a in conv_node.attribute)
        assert has_kernel_shape_after == True


class TestNodeRenaming:
    """Tests for ONNX node renaming functionality."""

    def test_rename_to_hailo_style(self, pytorch_generic_onnx):
        """Test renaming PyTorch nodes to Hailo style."""
        from src.core.onnx_utils import rename_onnx_nodes_to_hailo_style, detect_onnx_naming_style

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'converted.onnx')
            converted_path, end_nodes = rename_onnx_nodes_to_hailo_style(
                pytorch_generic_onnx,
                output_path=output_path,
                model_type='detect'
            )

            # Verify conversion
            assert os.path.exists(converted_path)
            assert len(end_nodes) == 3

            # Verify naming style changed
            new_style = detect_onnx_naming_style(converted_path)
            assert new_style == 'yolov5_official'

            # Verify end nodes have correct naming
            for node in end_nodes:
                assert '/model.24/m.' in node

    def test_already_compatible_unchanged(self, yolov5_official_onnx):
        """Test that already compatible ONNX is not modified."""
        from src.core.onnx_utils import rename_onnx_nodes_to_hailo_style

        # Should return same path since already compatible
        result_path, end_nodes = rename_onnx_nodes_to_hailo_style(
            yolov5_official_onnx,
            model_type='detect'
        )

        assert result_path == yolov5_official_onnx
        assert len(end_nodes) == 3


class TestEndNodeExtraction:
    """Tests for end node name extraction."""

    def test_get_end_nodes_detect(self, yolov5_official_onnx):
        """Test end node extraction for detection model."""
        from src.core.onnx_utils import get_end_node_names

        end_nodes = get_end_node_names(yolov5_official_onnx, model_type='detect')
        assert len(end_nodes) == 3
        assert '/model.24/m.0/Conv' in end_nodes
        assert '/model.24/m.1/Conv' in end_nodes
        assert '/model.24/m.2/Conv' in end_nodes


class TestValidation:
    """Tests for ONNX validation functions."""

    def test_validate_compatible_onnx(self, yolov5_official_onnx):
        """Test validation of compatible ONNX."""
        from src.core.onnx_utils import validate_onnx_hailo_compatibility

        result = validate_onnx_hailo_compatibility(yolov5_official_onnx)
        assert result['compatible'] == True
        assert result['naming_style'] == 'yolov5_official'
        assert len(result['errors']) == 0

    def test_validate_incompatible_onnx(self, pytorch_generic_onnx):
        """Test validation of incompatible ONNX."""
        from src.core.onnx_utils import validate_onnx_hailo_compatibility

        result = validate_onnx_hailo_compatibility(pytorch_generic_onnx)
        assert result['compatible'] == False
        assert result['naming_style'] == 'pytorch_generic'
        assert len(result['errors']) > 0
        assert result['recommended_action'] is not None
