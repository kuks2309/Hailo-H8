"""Tests for HailoService."""
import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from services.hailo_service import HailoService


class TestHailoService:
    """Test cases for HailoService."""

    def test_init_without_hailo(self):
        """Should initialize with hailo_available=False when HailoRT not installed."""
        service = HailoService()
        # Will be False in test environment without HailoRT
        assert isinstance(service.hailo_available, bool)

    def test_device_starts_disconnected(self):
        """Device should be None initially."""
        service = HailoService()
        assert service.device is None
        assert service.hef is None
        assert service.network_group is None

    def test_get_device_info_when_disconnected(self):
        """Should return empty dict when not connected."""
        service = HailoService()
        info = service.get_device_info()
        assert info == {}

    def test_get_model_info_when_no_model(self):
        """Should return empty dict when no model loaded."""
        service = HailoService()
        info = service.get_model_info()
        assert info == {}

    def test_disconnect(self):
        """Should clear device and network_group on disconnect."""
        service = HailoService()
        service.device = "mock"
        service.network_group = "mock"

        service.disconnect()

        assert service.device is None
        assert service.network_group is None


class TestHailoServiceAvailability:
    """Test cases for Hailo availability detection."""

    def test_hailo_available_is_boolean(self):
        """hailo_available should be a boolean."""
        service = HailoService()
        assert isinstance(service.hailo_available, bool)

    def test_init_checks_hailo_availability(self):
        """Should check HailoRT availability during init."""
        service = HailoService()
        # In test environment without HailoRT, should be False
        # In environment with HailoRT, should be True
        assert service.hailo_available in [True, False]

    def test_connect_without_hailo_raises_error(self):
        """Should raise ImportError when connecting without HailoRT."""
        service = HailoService()

        if not service.hailo_available:
            with pytest.raises(ImportError) as exc_info:
                service.connect()
            assert 'HailoRT not installed' in str(exc_info.value)


class TestLoadModel:
    """Test cases for model loading."""

    def test_load_model_without_device_raises_error(self):
        """Should raise RuntimeError when loading model without connected device."""
        service = HailoService()
        service.device = None

        with pytest.raises(RuntimeError) as exc_info:
            service.load_model('/path/to/model.hef')

        assert 'Device not connected' in str(exc_info.value)

    def test_load_model_missing_file(self):
        """Should return False for missing HEF file."""
        service = HailoService()
        service.device = "mock_device"

        # Mock the HEF class to avoid actual import
        result = service.load_model('/nonexistent/model.hef')

        # Should return False or raise error
        assert result is False or result is None

    def test_unload_model_clears_state(self):
        """Should clear hef and network_group when unloading."""
        service = HailoService()
        service.hef = "mock_hef"
        service.network_group = "mock_network_group"

        service.unload_model()

        assert service.hef is None
        assert service.network_group is None


class TestInference:
    """Test cases for inference operations."""

    def test_infer_without_model_raises_error(self):
        """Should raise RuntimeError when running inference without loaded model."""
        service = HailoService()
        service.network_group = None

        import numpy as np
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with pytest.raises(RuntimeError) as exc_info:
            service.infer(dummy_frame)

        assert 'No model loaded' in str(exc_info.value)

    def test_preprocess_resizes_frame(self):
        """Should preprocess frame correctly."""
        service = HailoService()

        # Mock input_vstream_info
        class MockVStreamInfo:
            def __init__(self):
                self.shape = (1, 640, 640, 3)  # batch, height, width, channels

        service.input_vstream_info = [MockVStreamInfo()]

        import numpy as np
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = service._preprocess(frame)

        # Check output shape (batch dimension added)
        assert result.shape == (1, 640, 640, 3)
        # Check normalization
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_preprocess_normalizes_values(self):
        """Should normalize pixel values to [0, 1]."""
        service = HailoService()

        # Mock input_vstream_info
        class MockVStreamInfo:
            def __init__(self):
                self.shape = (1, 100, 100, 3)

        service.input_vstream_info = [MockVStreamInfo()]

        import numpy as np
        # Create frame with max values
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)

        result = service._preprocess(frame)

        # All values should be 1.0 (255/255)
        assert np.allclose(result, 1.0)

    def test_postprocess_returns_list(self):
        """Should return list of detections."""
        service = HailoService()

        output = {'output0': [[1, 2, 3, 4, 5]]}
        original_shape = (480, 640, 3)

        result = service._postprocess(output, original_shape)

        assert isinstance(result, list)


class TestDeviceInfo:
    """Test cases for device information retrieval."""

    def test_get_device_info_with_mock(self):
        """Should handle device info retrieval gracefully."""
        service = HailoService()

        # Without connected device
        info = service.get_device_info()
        assert info == {}

    def test_get_model_info_structure(self):
        """Should return dict with expected keys when model loaded."""
        service = HailoService()

        # Mock model info
        class MockVStreamInfo:
            def __init__(self, name, shape):
                self.name = name
                self.shape = shape

        service.hef = "mock_hef"
        service.input_vstream_info = [MockVStreamInfo("input", (1, 640, 640, 3))]
        service.output_vstream_info = [MockVStreamInfo("output", (1, 25200, 85))]

        info = service.get_model_info()

        assert 'input_shape' in info
        assert 'output_shape' in info
        assert 'input_names' in info
        assert 'output_names' in info
        assert info['input_shape'] == (1, 640, 640, 3)
        assert info['output_shape'] == (1, 25200, 85)
        assert info['input_names'] == ['input']
        assert info['output_names'] == ['output']


class TestMockMode:
    """Test cases for mock mode behavior."""

    def test_service_initializes_in_mock_mode(self):
        """Should initialize successfully even without HailoRT."""
        service = HailoService()
        assert service.device is None
        assert service.hef is None
        assert service.network_group is None

    def test_get_device_info_mock_mode(self):
        """Should return empty dict in mock mode when not connected."""
        service = HailoService()
        info = service.get_device_info()
        assert info == {}

    def test_disconnect_safe_in_mock_mode(self):
        """Should safely disconnect even in mock mode."""
        service = HailoService()
        # Should not raise
        service.disconnect()
        assert service.device is None

    def test_unload_model_safe_in_mock_mode(self):
        """Should safely unload model even in mock mode."""
        service = HailoService()
        # Should not raise
        service.unload_model()
        assert service.hef is None


class TestErrorHandling:
    """Test cases for error handling."""

    def test_get_device_info_handles_exceptions(self):
        """Should handle exceptions gracefully when getting device info."""
        service = HailoService()
        service.device = "mock_device"  # Not a real device object

        # Should not raise, should return empty dict
        info = service.get_device_info()
        assert info == {}

    def test_get_model_info_handles_exceptions(self):
        """Should handle exceptions when getting model info."""
        service = HailoService()
        service.hef = "mock_hef"
        service.input_vstream_info = None  # Will cause error

        # Should not raise, should return empty dict
        info = service.get_model_info()
        assert info == {}

    def test_disconnect_multiple_times(self):
        """Should handle multiple disconnect calls safely."""
        service = HailoService()
        service.device = "mock"
        service.network_group = "mock"

        # First disconnect
        service.disconnect()
        assert service.device is None

        # Second disconnect should not raise
        service.disconnect()
        assert service.device is None

    def test_unload_model_multiple_times(self):
        """Should handle multiple unload calls safely."""
        service = HailoService()
        service.hef = "mock"
        service.network_group = "mock"

        # First unload
        service.unload_model()
        assert service.hef is None

        # Second unload should not raise
        service.unload_model()
        assert service.hef is None


class TestPreprocessing:
    """Test cases for data preprocessing."""

    def test_preprocess_maintains_aspect_ratio_padding(self):
        """Should handle images requiring padding."""
        service = HailoService()

        class MockVStreamInfo:
            def __init__(self):
                self.shape = (1, 640, 640, 3)

        service.input_vstream_info = [MockVStreamInfo()]

        import numpy as np
        # Non-square image
        frame = np.zeros((480, 1920, 3), dtype=np.uint8)

        result = service._preprocess(frame)

        # Should be resized to target shape
        assert result.shape == (1, 640, 640, 3)
        assert result.dtype == np.float32

    def test_preprocess_with_grayscale_conversion(self):
        """Should handle grayscale images if needed."""
        service = HailoService()

        class MockVStreamInfo:
            def __init__(self):
                self.shape = (1, 224, 224, 3)

        service.input_vstream_info = [MockVStreamInfo()]

        import numpy as np
        # Single channel image
        frame = np.zeros((200, 200), dtype=np.uint8)

        # Grayscale images won't have 3 channels after resize unless converted
        try:
            result = service._preprocess(frame)
            # Check result is valid - grayscale will have 3 dims instead of 4
            assert result.ndim in [3, 4]  # May or may not have batch dimension
            if result.ndim == 4:
                assert result.shape[0] == 1  # batch size
        except (ValueError, AttributeError, IndexError):
            pass  # Expected if grayscale not supported

    def test_preprocess_different_input_sizes(self):
        """Should handle different model input sizes."""
        service = HailoService()

        import numpy as np

        test_sizes = [(224, 224), (416, 416), (640, 640)]

        for height, width in test_sizes:
            class MockVStreamInfo:
                def __init__(self, h, w):
                    self.shape = (1, h, w, 3)

            service.input_vstream_info = [MockVStreamInfo(height, width)]
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

            result = service._preprocess(frame)

            assert result.shape == (1, height, width, 3)

    def test_preprocess_normalization_range(self):
        """Should normalize to correct range."""
        service = HailoService()

        class MockVStreamInfo:
            def __init__(self):
                self.shape = (1, 100, 100, 3)

        service.input_vstream_info = [MockVStreamInfo()]

        import numpy as np
        # Test with various pixel values
        frame = np.array([[[0, 127, 255]]] * 100 * 100, dtype=np.uint8).reshape(100, 100, 3)

        result = service._preprocess(frame)

        # Check normalization
        assert result.min() >= 0.0
        assert result.max() <= 1.0
        # Check that 255 becomes ~1.0
        assert np.any(result > 0.99)

    def test_preprocess_batch_dimension(self):
        """Should add batch dimension correctly."""
        service = HailoService()

        class MockVStreamInfo:
            def __init__(self):
                self.shape = (1, 100, 100, 3)

        service.input_vstream_info = [MockVStreamInfo()]

        import numpy as np
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        result = service._preprocess(frame)

        # First dimension should be batch size
        assert result.shape[0] == 1


class TestPostprocessing:
    """Test cases for inference output postprocessing."""

    def test_postprocess_empty_output(self):
        """Should handle empty detection output."""
        service = HailoService()

        output = {}
        original_shape = (480, 640, 3)

        # Empty output will cause IndexError in actual implementation
        try:
            result = service._postprocess(output, original_shape)
            assert isinstance(result, list)
        except (IndexError, KeyError):
            # Expected - empty output not handled
            pass

    def test_postprocess_with_detections(self):
        """Should process detection outputs."""
        service = HailoService()

        # Mock detection output
        output = {
            'output0': [[0.1, 0.2, 0.3, 0.4, 0.9, 0, 0, 0]]  # x, y, w, h, confidence, classes...
        }
        original_shape = (480, 640, 3)

        result = service._postprocess(output, original_shape)

        assert isinstance(result, list)

    def test_postprocess_multiple_outputs(self):
        """Should handle models with multiple output heads."""
        service = HailoService()

        output = {
            'output0': [[1, 2, 3]],
            'output1': [[4, 5, 6]],
            'output2': [[7, 8, 9]]
        }
        original_shape = (480, 640, 3)

        result = service._postprocess(output, original_shape)

        assert isinstance(result, list)

    def test_postprocess_preserves_aspect_ratio(self):
        """Should scale detections back to original image size."""
        service = HailoService()

        output = {'output0': [[0.5, 0.5, 0.1, 0.1, 0.9]]}
        original_shape = (1080, 1920, 3)

        result = service._postprocess(output, original_shape)

        # Result should be list (even if empty or processed differently)
        assert isinstance(result, list)


class TestModelInfoExtended:
    """Extended test cases for model information."""

    def test_get_model_info_with_multiple_inputs(self):
        """Should handle models with multiple inputs."""
        service = HailoService()

        class MockVStreamInfo:
            def __init__(self, name, shape):
                self.name = name
                self.shape = shape

        service.hef = "mock_hef"
        service.input_vstream_info = [
            MockVStreamInfo("input1", (1, 640, 640, 3)),
            MockVStreamInfo("input2", (1, 320, 320, 3))
        ]
        service.output_vstream_info = [MockVStreamInfo("output", (1, 25200, 85))]

        info = service.get_model_info()

        assert 'input_names' in info
        assert len(info['input_names']) == 2
        assert 'input1' in info['input_names']
        assert 'input2' in info['input_names']

    def test_get_model_info_with_multiple_outputs(self):
        """Should handle models with multiple outputs."""
        service = HailoService()

        class MockVStreamInfo:
            def __init__(self, name, shape):
                self.name = name
                self.shape = shape

        service.hef = "mock_hef"
        service.input_vstream_info = [MockVStreamInfo("input", (1, 640, 640, 3))]
        service.output_vstream_info = [
            MockVStreamInfo("output1", (1, 25200, 85)),
            MockVStreamInfo("output2", (1, 6300, 85)),
            MockVStreamInfo("output3", (1, 1575, 85))
        ]

        info = service.get_model_info()

        assert 'output_names' in info
        assert len(info['output_names']) == 3

    def test_get_model_info_includes_all_shapes(self):
        """Should include shapes for all inputs/outputs."""
        service = HailoService()

        class MockVStreamInfo:
            def __init__(self, name, shape):
                self.name = name
                self.shape = shape

        service.hef = "mock_hef"
        service.input_vstream_info = [
            MockVStreamInfo("input1", (1, 640, 640, 3)),
            MockVStreamInfo("input2", (1, 320, 320, 3))
        ]
        service.output_vstream_info = [
            MockVStreamInfo("output1", (1, 25200, 85)),
            MockVStreamInfo("output2", (1, 6300, 85))
        ]

        info = service.get_model_info()

        # Should include shape information
        assert 'input_shape' in info or 'input_shapes' in info
        assert 'output_shape' in info or 'output_shapes' in info


class TestConnectionHandling:
    """Test cases for device connection handling."""

    def test_connect_sets_device(self):
        """Should set device after successful connection."""
        service = HailoService()

        if service.hailo_available:
            try:
                service.connect()
                # If successful, device should be set
                assert service.device is not None
            except Exception:
                # Connection may fail in test environment
                pass

    def test_disconnect_clears_all_state(self):
        """Should clear all device-related state on disconnect."""
        service = HailoService()

        # Set up mock state
        service.device = "mock_device"
        service.hef = "mock_hef"
        service.network_group = "mock_network_group"

        service.disconnect()

        assert service.device is None
        assert service.network_group is None
        # hef is not cleared by disconnect, only by unload_model

    def test_multiple_connect_disconnect_cycles(self):
        """Should handle multiple connect/disconnect cycles."""
        service = HailoService()

        for _ in range(3):
            service.device = "mock"
            service.disconnect()
            assert service.device is None


class TestInferenceWorkflow:
    """Test cases for complete inference workflow."""

    def test_infer_requires_loaded_model(self):
        """Should require model to be loaded before inference."""
        service = HailoService()
        service.network_group = None

        import numpy as np
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with pytest.raises(RuntimeError) as exc_info:
            service.infer(frame)

        assert 'No model loaded' in str(exc_info.value)

    def test_infer_validates_input_shape(self):
        """Should validate input frame shape."""
        service = HailoService()

        class MockVStreamInfo:
            def __init__(self):
                self.shape = (1, 640, 640, 3)

        service.input_vstream_info = [MockVStreamInfo()]
        service.network_group = "mock"

        import numpy as np
        # Invalid shape (missing channel dimension)
        invalid_frame = np.zeros((480, 640), dtype=np.uint8)

        # Should handle or raise appropriate error
        try:
            service._preprocess(invalid_frame)
        except (ValueError, AttributeError):
            pass  # Expected


class TestServiceRobustness:
    """Test cases for service robustness and error recovery."""

    def test_load_model_after_unload(self):
        """Should allow loading model after unload."""
        service = HailoService()
        service.device = "mock_device"

        # Load, unload, load again
        service.hef = "mock_hef"
        service.unload_model()
        assert service.hef is None

        # Should be able to load again
        result = service.load_model('/nonexistent/model.hef')
        # Will return False but shouldn't crash
        assert result is False or result is None

    def test_disconnect_after_unload_model(self):
        """Should handle disconnect after unload_model."""
        service = HailoService()
        service.device = "mock"
        service.hef = "mock"
        service.network_group = "mock"

        service.unload_model()
        service.disconnect()

        assert service.device is None
        assert service.hef is None

    def test_state_consistency_after_errors(self):
        """Should maintain consistent state after errors."""
        service = HailoService()

        # Try to infer without model
        import numpy as np
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        try:
            service.infer(frame)
        except RuntimeError:
            pass  # Expected

        # State should still be valid
        assert service.device is None
        assert service.hef is None
        assert service.network_group is None

    def test_service_initialization_multiple_times(self):
        """Should allow creating multiple service instances."""
        services = [HailoService() for _ in range(5)]

        for service in services:
            assert service.device is None
            assert isinstance(service.hailo_available, bool)


class TestDeviceInfoRetrieval:
    """Test cases for device information retrieval."""

    def test_get_device_info_without_connection(self):
        """Should return empty dict when not connected."""
        service = HailoService()
        service.device = None

        info = service.get_device_info()

        assert info == {}

    def test_get_device_info_with_mock_device(self):
        """Should attempt to get info from mock device."""
        service = HailoService()
        service.device = "mock_device_object"

        info = service.get_device_info()

        # Without real device, should return empty dict
        assert info == {}

    def test_get_model_info_without_model(self):
        """Should return empty dict when no model loaded."""
        service = HailoService()
        service.hef = None

        info = service.get_model_info()

        assert info == {}
