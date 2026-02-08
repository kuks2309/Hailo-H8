"""
Model Converter for Hailo-Compiler-UI.
Handles PT -> ONNX -> HEF conversion pipeline.
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Callable
import numpy as np

from .exceptions import (
    ModelLoadError,
    ExportError,
    CompilationError,
    CalibrationError
)
from .onnx_utils import (
    detect_onnx_naming_style,
    is_hailo_compatible_naming,
    rename_onnx_nodes_to_hailo_style,
    get_end_node_names
)


def _configure_torch_for_yolo():
    """
    Configure PyTorch 2.6+ for YOLO model loading.
    PyTorch 2.6 changed weights_only default to True for security.
    This adds YOLO model classes to safe globals for torch.load().
    """
    try:
        import torch

        # Set environment variable for subprocess calls (yolo CLI)
        os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'

        # For PyTorch 2.6+, add safe globals
        if hasattr(torch, 'serialization') and hasattr(torch.serialization, 'add_safe_globals'):
            try:
                # Try to import YOLO model classes and add them as safe
                from ultralytics.nn.tasks import DetectionModel, SegmentationModel, ClassificationModel
                torch.serialization.add_safe_globals([DetectionModel, SegmentationModel, ClassificationModel])
            except ImportError:
                pass

            try:
                # Also try yolov5 models
                from models.yolo import DetectionModel as YOLOv5Detection
                from models.yolo import SegmentationModel as YOLOv5Segment
                torch.serialization.add_safe_globals([YOLOv5Detection, YOLOv5Segment])
            except ImportError:
                pass

    except ImportError:
        pass


# Configure torch on module load
_configure_torch_for_yolo()


def detect_yolo_version_from_pt(pt_path: str) -> str:
    """
    Detect YOLO version from PyTorch checkpoint file.

    Args:
        pt_path: Path to .pt file

    Returns:
        'v5', 'v8', 'v9', 'v10', or None if not a YOLO model
    """
    import zipfile

    # Method 1: Check file path patterns first (fast)
    path_lower = pt_path.lower()
    if 'yolov5' in path_lower:
        return 'v5'
    elif 'yolov10' in path_lower:
        return 'v10'
    elif 'yolov9' in path_lower:
        return 'v9'
    elif 'yolov8' in path_lower:
        return 'v8'

    # Method 2: Peek into checkpoint's pickle data for module references
    try:
        with zipfile.ZipFile(pt_path, 'r') as zf:
            for pkl_name in ['archive/data.pkl', 'data.pkl']:
                if pkl_name in zf.namelist():
                    with zf.open(pkl_name) as f:
                        content = f.read(4096).decode('latin-1', errors='ignore')
                        if 'ultralytics.nn' in content or 'ultralytics.engine' in content:
                            return 'v8'
                        if 'models.yolo' in content or 'models.common' in content:
                            return 'v5'
                    break
    except (zipfile.BadZipFile, KeyError, Exception):
        pass

    # Method 3: Try loading and inspect checkpoint structure
    try:
        import torch
        checkpoint = torch.load(pt_path, map_location='cpu', weights_only=False)

        if isinstance(checkpoint, dict):
            if 'train_args' in checkpoint:
                return 'v8'

            if 'model' in checkpoint:
                model = checkpoint['model']
                module_path = type(model).__module__

                if 'ultralytics' in module_path:
                    return 'v8'
                elif 'models' in module_path:
                    return 'v5'

            # YOLOv5 checkpoint key pattern
            yolo_v5_keys = {'model', 'ema', 'updates', 'optimizer', 'epoch'}
            if yolo_v5_keys.issubset(set(checkpoint.keys())) and 'train_args' not in checkpoint:
                return 'v5'

        return None

    except ModuleNotFoundError as e:
        error_msg = str(e)
        if 'models' in error_msg and 'ultralytics' not in error_msg:
            return 'v5'
        if 'ultralytics' in error_msg:
            return 'v8'
        return None
    except (RuntimeError, KeyError, TypeError):
        return None


def detect_yolo_task_from_pt(pt_path: str) -> str:
    """
    Detect YOLO task type from PyTorch checkpoint file.

    Args:
        pt_path: Path to .pt file

    Returns:
        'segment', 'detect', 'classify', or 'detect' as default
    """
    import zipfile

    # Method 1: Check pickle data for task indicators (most reliable)
    try:
        with zipfile.ZipFile(pt_path, 'r') as zf:
            for pkl_name in ['archive/data.pkl', 'data.pkl']:
                if pkl_name in zf.namelist():
                    with zf.open(pkl_name) as f:
                        content = f.read(8192).decode('latin-1', errors='ignore')
                        if 'SegmentationModel' in content or 'Segment' in content:
                            return 'segment'
                        if 'ClassificationModel' in content or 'Classify' in content:
                            return 'classify'
                    break
    except (zipfile.BadZipFile, KeyError, Exception):
        pass

    # Method 2: Check file path patterns
    path_lower = pt_path.lower()
    if '-seg' in path_lower or 'segment' in path_lower or '_seg' in path_lower:
        return 'segment'
    if '-cls' in path_lower or 'classify' in path_lower or '_cls' in path_lower:
        return 'classify'

    # Method 3: Try loading and inspect model class name
    try:
        import torch
        checkpoint = torch.load(pt_path, map_location='cpu', weights_only=False)

        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model = checkpoint['model']
            class_name = type(model).__name__

            if 'Segment' in class_name:
                return 'segment'
            if 'Classif' in class_name:
                return 'classify'

    except (ModuleNotFoundError, RuntimeError, KeyError, TypeError):
        pass

    return 'detect'


def get_recommended_opset(yolo_version: str, task_type: str) -> int:
    """
    Get recommended ONNX opset version based on YOLO version and task.

    Args:
        yolo_version: 'v5', 'v8', 'v9', 'v10', or None
        task_type: 'detect', 'segment', 'classify'

    Returns:
        Recommended opset version (11 for YOLOv5, 17 for others)
    """
    if yolo_version == 'v5':
        # YOLOv5: opset 11 is Hailo standard for best compatibility
        return 11
    else:
        # YOLOv8+: opset 17 recommended
        return 17


def extract_onnx_nodes(onnx_path: str) -> dict:
    """
    Extract input/output and internal node names from ONNX model.
    Specifically looks for YOLOv5/v8 detection head nodes for Hailo compatibility.

    Supports two ONNX export patterns:
    1. YOLOv5 official export.py: /model.24/m.0/Conv, /model.24/proto/cv3/conv/Conv
    2. Ultralytics/torch.onnx: conv2d_60, conv2d_61, conv2d_62

    Returns:
        dict with 'input_nodes', 'output_nodes', 'detection_heads', 'proto_node', 'naming_style'
    """
    try:
        import onnx
        model = onnx.load(onnx_path)

        input_nodes = [inp.name for inp in model.graph.input]
        output_nodes = [out.name for out in model.graph.output]

        detection_heads = []
        proto_node = None
        naming_style = 'unknown'

        # Pattern 1: YOLOv5 official export.py (module path naming)
        # /model.24/m.0/Conv, /model.24/m.1/Conv, /model.24/m.2/Conv
        # /model.24/proto/cv3/conv/Conv
        model24_detect = []
        model24_proto = None

        for node in model.graph.node:
            # Detection heads: /model.24/m.X/Conv
            if '/model.24/m.' in node.name and '/Conv' in node.name:
                model24_detect.append(node.name)
            # Proto: /model.24/proto/cv3/conv/Conv
            if '/model.24/proto/' in node.name and '/Conv' in node.name:
                model24_proto = node.name

        if model24_detect:
            # Sort by m.X number (m.0, m.1, m.2)
            model24_detect = sorted(model24_detect, key=lambda x: int(x.split('/m.')[1].split('/')[0]))
            detection_heads = model24_detect
            proto_node = model24_proto
            naming_style = 'yolov5_official'

        # Pattern 3: YOLOv8 (module path naming - any export method)
        # /model.22/cv2.0/cv2.0.2/Conv, /model.22/cv2.1/cv2.1.2/Conv, /model.22/cv2.2/cv2.2.2/Conv
        # /model.22/cv3.0/cv3.0.2/Conv, /model.22/cv3.1/cv3.1.2/Conv, /model.22/cv3.2/cv3.2.2/Conv
        if not detection_heads:
            model22_cv2 = []
            model22_cv3 = []

            for node in model.graph.node:
                if '/model.22/cv2.' in node.name and '/Conv' in node.name:
                    model22_cv2.append(node.name)
                if '/model.22/cv3.' in node.name and '/Conv' in node.name:
                    model22_cv3.append(node.name)

            if model22_cv2 or model22_cv3:
                # YOLOv8 detection heads
                detection_heads = sorted(model22_cv2 + model22_cv3)
                naming_style = 'yolov8_official'

        if not detection_heads:
            # Pattern 2: Ultralytics/torch.onnx (auto-generated naming)
            # YOLOv5-seg: conv2d_60, conv2d_61, conv2d_62 are detection heads
            # Proto is silu_59 or similar

            # First, look for conv2d_N pattern (PyTorch 2.x Ultralytics export)
            conv2d_nodes = []
            for node in model.graph.node:
                if node.op_type == 'Conv' and 'conv2d_' in node.name.lower():
                    if node.output:
                        conv2d_nodes.append({
                            'name': node.name,
                            'output': node.output[0]
                        })

            if conv2d_nodes:
                # conv2d_60, conv2d_61, conv2d_62 are the detection heads
                detection_heads = [n['output'] for n in conv2d_nodes]
                naming_style = 'ultralytics'

                # Proto node: look for silu_59 in outputs or find Sigmoid before mask
                for out in output_nodes:
                    if 'silu' in out.lower() or out == 'output1':
                        proto_node = out
                        break

                # If no proto in outputs, look for last getitem before conv2d nodes
                if not proto_node:
                    for node in model.graph.node:
                        if node.op_type == 'Conv' and 'Conv_108' in node.name:
                            if node.output:
                                proto_node = node.output[0]
                                break

            # Fallback: look for Conv_N pattern (older naming)
            if not detection_heads:
                conv_nodes = []
                for node in model.graph.node:
                    if node.op_type == 'Conv' and node.name.startswith('Conv_'):
                        if node.output:
                            conv_nodes.append({
                                'name': node.name,
                                'output': node.output[0]
                            })

                if len(conv_nodes) >= 3:
                    # Take last 3 Conv nodes as detection heads
                    detection_heads = [n['output'] for n in conv_nodes[-3:]]
                    naming_style = 'ultralytics'

            # Another fallback: node_conv2d_N pattern
            if not detection_heads:
                for node in model.graph.node:
                    if node.op_type == 'Conv' and 'node_conv2d_' in node.name:
                        if node.output:
                            detection_heads.append(node.output[0])
                if detection_heads:
                    naming_style = 'ultralytics'

        # Determine YOLO version from naming_style
        if naming_style == 'yolov5_official':
            yolo_version = 'v5'
        elif naming_style == 'yolov8_official':
            yolo_version = 'v8'
        else:
            yolo_version = 'unknown'

        return {
            'input_nodes': input_nodes,
            'output_nodes': output_nodes,
            'detection_heads': detection_heads,
            'proto_node': proto_node,
            'naming_style': naming_style,
            'yolo_version': yolo_version
        }
    except Exception as e:
        return {'input_nodes': [], 'output_nodes': [], 'detection_heads': [], 'error': str(e)}


def extract_onnx_output_nodes(onnx_path: str) -> dict:
    """Legacy wrapper for extract_onnx_nodes."""
    return extract_onnx_nodes(onnx_path)


def validate_onnx_hailo_compatibility(onnx_path: str) -> dict:
    """
    Validate ONNX model compatibility with Hailo Model Zoo.

    Returns:
        dict with keys:
        - 'compatible': bool
        - 'naming_style': str
        - 'yolo_version': str ('v5', 'v8', or 'unknown')
        - 'detection_heads': list
        - 'warnings': list of warning messages
        - 'errors': list of error messages
        - 'recommended_action': str
    """
    result = {
        'compatible': False,
        'naming_style': 'unknown',
        'yolo_version': 'unknown',
        'detection_heads': [],
        'warnings': [],
        'errors': [],
        'recommended_action': '',
        'alternative_commands': []
    }

    try:
        node_info = extract_onnx_nodes(onnx_path)
        result['naming_style'] = node_info.get('naming_style', 'unknown')
        result['yolo_version'] = node_info.get('yolo_version', 'unknown')
        result['detection_heads'] = node_info.get('detection_heads', [])

        if result['naming_style'] == 'ultralytics':
            # Ultralytics style CAN be compiled if we have detected end-node-names
            if result['detection_heads']:
                result['compatible'] = True
                result['warnings'].append(
                    f"Ultralytics ONNX detected - will use --end-node-names: {result['detection_heads'][:3]}..."
                )
            else:
                result['compatible'] = False
                result['errors'].append(
                    "ONNX uses auto-generated node names but end-nodes could not be detected"
                )
                result['recommended_action'] = (
                    "Use Netron (https://netron.app) to inspect ONNX and find Conv nodes before output.\n"
                    "Or use Hailo Model Zoo standard model."
                )
                result['alternative_commands'] = [
                    {'name': 'Hailo Model Zoo (recommended)', 'command': 'hailomz compile {model}_seg --hw-arch hailo8 --calib-path <images> --classes {classes}'}
                ]
        elif result['naming_style'] in ('yolov5_official', 'yolov8_official'):
            result['compatible'] = True
            if not result['detection_heads']:
                result['warnings'].append("No detection heads found - manual node specification may be required")
        else:
            result['compatible'] = False
            result['errors'].append("Unknown ONNX structure - manual node specification may be required")
            result['recommended_action'] = (
                "Choose one of the following:\n"
                "1. Use YOLOv5 official export.py: python export.py --weights <model>.pt --include onnx --opset 11 --simplify\n"
                "2. Use Hailo Model Zoo: hailomz compile yolov5s_seg --hw-arch hailo8 --calib-path <images>\n"
                "3. Manually specify --end-node-names with detected node names"
            )
            result['alternative_commands'] = [
                {'name': 'YOLOv5 Official Export', 'command': 'python export.py --weights <model>.pt --include onnx --opset 11 --simplify'},
                {'name': 'Hailo Model Zoo', 'command': 'hailomz compile {model} --hw-arch hailo8 --calib-path <images> --classes {classes}'},
                {'name': 'Manual Node Specification', 'command': 'hailomz compile {model} --ckpt <onnx> --hw-arch hailo8 --calib-path <images> --end-node-names <node1> <node2> <node3>'}
            ]

    except Exception as e:
        result['errors'].append(f"Failed to analyze ONNX: {str(e)}")

    return result


def verify_onnx_for_hailo(onnx_path: str) -> dict:
    """
    Comprehensive ONNX verification before HEF compilation.

    Returns:
        dict with keys: 'valid', 'issues', 'suggestions', 'metadata'
    """
    issues = []
    suggestions = []
    metadata = {}

    try:
        import onnx
        from onnx import checker

        # 1. ONNX file validity
        model = onnx.load(onnx_path)
        try:
            checker.check_model(model)
        except Exception as e:
            issues.append(f"ONNX validation failed: {e}")

        # 2. Opset version check
        opset = model.opset_import[0].version
        metadata['opset'] = opset
        if opset < 11:
            issues.append(f"Opset {opset} too old (minimum: 11)")
        elif opset > 18:
            suggestions.append(f"Opset {opset} may not be fully supported")

        # 3. Node naming compatibility
        compat = validate_onnx_hailo_compatibility(onnx_path)
        metadata['naming_style'] = compat.get('naming_style', 'unknown')
        metadata['detection_heads'] = compat.get('detection_heads', [])
        metadata['proto_node'] = compat.get('proto_node')
        if not compat['compatible']:
            issues.extend(compat['errors'])
            if compat.get('recommended_action'):
                suggestions.append(compat['recommended_action'])

        # 4. Input shape format
        for inp in model.graph.input:
            shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            metadata['input_shape'] = shape
            if len(shape) == 4:
                batch, channels, height, width = shape
                if channels not in [1, 3]:
                    suggestions.append(f"Unusual channel count: {channels}")
                if height != width:
                    suggestions.append(f"Non-square input ({height}x{width})")

        # 5. Output node count
        output_count = len(model.graph.output)
        metadata['output_count'] = output_count

        # 6. Check for dynamic shapes
        for inp in model.graph.input:
            for dim in inp.type.tensor_type.shape.dim:
                if dim.dim_param:
                    issues.append(f"Dynamic shape detected: {dim.dim_param}")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'suggestions': suggestions,
            'metadata': metadata
        }

    except Exception as e:
        return {
            'valid': False,
            'issues': [f"Failed to load ONNX: {e}"],
            'suggestions': ["Ensure the file is a valid ONNX model"],
            'metadata': {}
        }


def detect_yolo_base_model(onnx_path: str) -> str:
    """
    Detect YOLO base model name from ONNX structure.

    NOTE: Conv node count thresholds are approximations and MAY NEED TUNING.
    """
    try:
        import onnx
        model = onnx.load(onnx_path)

        # Count Conv nodes
        conv_count = sum(1 for node in model.graph.node if node.op_type == 'Conv')

        # Detect YOLO version from node patterns
        node_info = extract_onnx_nodes(onnx_path)
        yolo_version = node_info.get('yolo_version', 'unknown')

        # Determine model size based on Conv node count
        if conv_count < 100:
            size = 'n'
        elif conv_count < 200:
            size = 's'
        elif conv_count < 300:
            size = 'm'
        elif conv_count < 400:
            size = 'l'
        else:
            size = 'x'

        if yolo_version == 'v5':
            return f'yolov5{size}'
        elif yolo_version == 'v8':
            return f'yolov8{size}'
        else:
            return 'yolov5s'

    except Exception:
        return 'yolov5s'


def generate_hailo_yaml(
    base_model: str,
    onnx_path: str,
    output_yaml_path: str,
    num_classes: int = 80,
    model_type: str = 'detect'
) -> str:
    """
    Generate Hailo-compatible YAML config with actual ONNX node names.

    This solves the PyTorch 1.x vs 2.x node naming incompatibility by
    reading actual node names from the ONNX file.

    Args:
        base_model: Base model name (yolov5s, yolov8s, etc.)
        onnx_path: Path to ONNX model
        output_yaml_path: Where to save generated YAML
        num_classes: Number of classes
        model_type: 'detect' or 'segment'

    Returns:
        Path to generated YAML file
    """
    import yaml

    # Extract actual node names from ONNX (including internal detection heads)
    nodes = extract_onnx_nodes(onnx_path)
    input_nodes = nodes.get('input_nodes', [])
    detection_heads = nodes.get('detection_heads', [])
    proto_node = nodes.get('proto_node')

    # Determine input node (usually 'images' or first input)
    input_node = input_nodes[0] if input_nodes else 'images'

    # Determine base YAML based on model
    if 'v8' in base_model:
        base_yaml = 'base/yolov8_seg.yaml' if model_type == 'segment' else 'base/yolov8.yaml'
        meta_arch = 'yolov8_seg' if model_type == 'segment' else 'yolov8'
    else:  # v5
        base_yaml = 'base/yolov5_seg.yaml' if model_type == 'segment' else 'base/yolov5.yaml'
        meta_arch = 'yolov5_seg' if model_type == 'segment' else 'yolov5'

    # Build end nodes list for parser
    # For YOLOv5 seg: [proto_node, detection_head_P5, detection_head_P4, detection_head_P3]
    if model_type == 'segment' and detection_heads:
        # Reverse order: P5(large), P4(medium), P3(small) for Hailo
        end_nodes = detection_heads[::-1]  # Reverse: conv2d_62, conv2d_61, conv2d_60
        if proto_node:
            end_nodes = [proto_node] + end_nodes
    elif detection_heads:
        end_nodes = detection_heads[::-1]
    else:
        end_nodes = None

    # Build YAML config
    config = {
        'base': [base_yaml],
        'network': {
            'network_name': f"{base_model}_{model_type}_custom"
        },
        'parser': {
            'nodes': [
                input_node if input_node != 'images' else 'images',
                end_nodes
            ],
            'normalization_params': {
                'normalize_in_net': True,
                'mean_list': [0, 0, 0],
                'std_list': [255.0, 255.0, 255.0]
            }
        },
        'postprocessing': {
            'meta_arch': meta_arch,
            'classes': num_classes
        }
    }

    if model_type == 'segment':
        config['postprocessing']['mask_threshold'] = 0.5
        config['postprocessing']['nms_iou_thresh'] = 0.6
        config['postprocessing']['score_threshold'] = 0.001

    # Write YAML
    os.makedirs(os.path.dirname(output_yaml_path), exist_ok=True)
    with open(output_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    return output_yaml_path


class ModelConverter:
    """
    Model conversion service for PT -> ONNX -> HEF pipeline.
    """

    def __init__(self):
        self.progress_callback: Optional[Callable[[int], None]] = None
        self.log_callback: Optional[Callable[[str], None]] = None

    def set_callbacks(self, progress_cb=None, log_cb=None):
        """Set progress and log callbacks."""
        self.progress_callback = progress_cb
        self.log_callback = log_cb

    def _log(self, message: str):
        """Log message via callback."""
        if self.log_callback:
            self.log_callback(message)

    def _progress(self, value: int):
        """Update progress via callback."""
        if self.progress_callback:
            self.progress_callback(value)

    def _patch_torch_load_for_yolo(self):
        """
        Patch torch.load for PyTorch 2.6+ compatibility with YOLO models.
        PyTorch 2.6+ defaults weights_only=True which breaks YOLO loading.
        """
        try:
            import torch

            # Check if already patched
            if hasattr(torch, '_original_load_for_yolo'):
                return

            _original_load = torch.load

            def patched_load(*args, **kwargs):
                if 'weights_only' not in kwargs:
                    kwargs['weights_only'] = False
                return _original_load(*args, **kwargs)

            torch._original_load_for_yolo = _original_load
            torch.load = patched_load
            self._log("Patched torch.load for PyTorch 2.6+ compatibility")

            # Also add safe globals
            if hasattr(torch.serialization, 'add_safe_globals'):
                try:
                    from yolov5.models.yolo import DetectionModel, SegmentationModel
                    torch.serialization.add_safe_globals([DetectionModel, SegmentationModel])
                except ImportError:
                    pass

        except Exception as e:
            self._log(f"Warning: Could not patch torch.load: {e}")

    def _detect_yolo_version(self, pt_path: str) -> Optional[str]:
        """
        Detect YOLO version from checkpoint file.
        Returns: 'v5', 'v8', or None if not a YOLO model
        """
        import zipfile

        # Method 1: Peek into checkpoint's pickle data for module references
        try:
            with zipfile.ZipFile(pt_path, 'r') as zf:
                for pkl_name in ['archive/data.pkl', 'data.pkl']:
                    if pkl_name in zf.namelist():
                        with zf.open(pkl_name) as f:
                            content = f.read(4096).decode('latin-1', errors='ignore')
                            if 'ultralytics.nn' in content or 'ultralytics.engine' in content:
                                self._log("Detected YOLOv8 model from module reference")
                                return 'v8'
                            if 'models.yolo' in content or 'models.common' in content:
                                self._log("Detected YOLOv5 model from module reference")
                                return 'v5'
                        break
        except (zipfile.BadZipFile, KeyError, Exception):
            pass

        # Method 2: Try loading and inspect checkpoint structure
        try:
            import torch
            checkpoint = torch.load(pt_path, map_location='cpu', weights_only=False)

            if isinstance(checkpoint, dict):
                if 'train_args' in checkpoint:
                    self._log("Detected YOLOv8 model from train_args")
                    return 'v8'

                if 'model' in checkpoint:
                    model = checkpoint['model']
                    module_path = type(model).__module__
                    class_name = type(model).__name__

                    if 'ultralytics' in module_path:
                        self._log(f"Detected YOLOv8 model: {module_path}.{class_name}")
                        return 'v8'
                    elif 'models' in module_path:
                        self._log(f"Detected YOLOv5 model: {module_path}.{class_name}")
                        return 'v5'

                yolo_v5_keys = {'model', 'ema', 'updates', 'optimizer', 'epoch'}
                if yolo_v5_keys.issubset(set(checkpoint.keys())) and 'train_args' not in checkpoint:
                    self._log("Detected YOLOv5 model from checkpoint keys")
                    return 'v5'

            return None

        except ModuleNotFoundError as e:
            error_msg = str(e)
            if 'models' in error_msg and 'ultralytics' not in error_msg:
                self._log(f"Detected YOLOv5 model (missing module: {e})")
                return 'v5'
            if 'ultralytics' in error_msg:
                self._log(f"Detected YOLOv8 model (missing module: {e})")
                return 'v8'
            return None
        except (RuntimeError, KeyError, TypeError):
            return None

    def _detect_yolo_task(self, pt_path: str) -> str:
        """
        Detect YOLO task type from checkpoint file.
        Returns: 'segment', 'detect', 'classify', or 'detect' as default

        Segmentation models require higher opset version (18+) due to Resize operators.
        """
        import zipfile

        # Method 1: Check pickle data for task indicators
        try:
            with zipfile.ZipFile(pt_path, 'r') as zf:
                for pkl_name in ['archive/data.pkl', 'data.pkl']:
                    if pkl_name in zf.namelist():
                        with zf.open(pkl_name) as f:
                            content = f.read(8192).decode('latin-1', errors='ignore')
                            if 'SegmentationModel' in content or 'Segment' in content:
                                self._log("Detected segmentation model from pickle data")
                                return 'segment'
                            if 'ClassificationModel' in content or 'Classify' in content:
                                self._log("Detected classification model from pickle data")
                                return 'classify'
                        break
        except (zipfile.BadZipFile, KeyError, Exception):
            pass

        # Method 2: Check file path patterns
        path_lower = pt_path.lower()
        if '-seg' in path_lower or 'segment' in path_lower or '_seg' in path_lower:
            self._log("Detected segmentation model from file path")
            return 'segment'
        if '-cls' in path_lower or 'classify' in path_lower or '_cls' in path_lower:
            self._log("Detected classification model from file path")
            return 'classify'

        # Method 3: Try loading and inspect model class name
        try:
            import torch
            checkpoint = torch.load(pt_path, map_location='cpu', weights_only=False)

            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                model = checkpoint['model']
                class_name = type(model).__name__

                if 'Segment' in class_name:
                    self._log(f"Detected segmentation model: {class_name}")
                    return 'segment'
                if 'Classif' in class_name:
                    self._log(f"Detected classification model: {class_name}")
                    return 'classify'

        except (ModuleNotFoundError, RuntimeError, KeyError, TypeError):
            pass

        self._log("Defaulting to detection model")
        return 'detect'

    def _is_yolo_model(self, pt_path: str) -> bool:
        """Check if model is a YOLO model."""
        version = self._detect_yolo_version(pt_path)
        if version:
            return True

        # Fallback: check file path patterns
        path_lower = pt_path.lower()
        yolo_path_indicators = ['yolo', 'best.pt', 'last.pt', 'weights/']
        if any(indicator in path_lower for indicator in yolo_path_indicators):
            self._log("Detected YOLO model from file path pattern")
            return True

        return False

    def _detect_pytorch_version(self) -> tuple:
        """
        Detect PyTorch version and warn about ONNX compatibility issues.
        Returns: (major_version: int, has_onnx_naming_issue: bool)
        """
        try:
            import torch
            version = torch.__version__
            major = int(version.split('.')[0])
            has_issue = major >= 2
            return major, has_issue
        except Exception:
            return 0, False

    def convert_pt_to_onnx(
        self,
        pt_path: str,
        onnx_path: str,
        input_size: Tuple[int, int] = (640, 640),
        batch_size: int = 1,
        opset_version: int = 17
    ) -> str:
        """
        Convert PyTorch model to ONNX format.

        Args:
            pt_path: Path to PyTorch model (.pt file)
            onnx_path: Output path for ONNX model
            input_size: Model input size (height, width)
            batch_size: Batch size for export
            opset_version: ONNX opset version

        Returns:
            Path to exported ONNX file

        Raises:
            ModelLoadError: If model loading fails
            ExportError: If ONNX export fails
        """
        # Validate input
        if not os.path.exists(pt_path):
            raise ModelLoadError(f"Model file not found: {pt_path}", pt_path)

        self._log(f"Loading PyTorch model: {pt_path}")
        self._progress(10)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

        # Detect task type and adjust opset for segmentation models
        task_type = self._detect_yolo_task(pt_path)
        self._log(f"Detected task type: {task_type}")

        # Segmentation models require opset 18+ due to Resize operator
        if task_type == 'segment' and opset_version < 18:
            self._log(f"Segmentation model detected: upgrading opset {opset_version} → 18 (Resize operator requirement)")
            opset_version = 18

        try:
            # Check if it's a YOLO model
            if self._is_yolo_model(pt_path):
                self._log(f"Detected YOLO model ({task_type}), using export with opset {opset_version}...")
                return self._export_yolo_to_onnx(
                    pt_path, onnx_path, input_size, opset_version
                )
            else:
                self._log("Using generic PyTorch export...")
                return self._export_generic_to_onnx(
                    pt_path, onnx_path, input_size, batch_size, opset_version
                )

        except (ModelLoadError, ExportError):
            raise
        except Exception as e:
            raise ExportError(f"Conversion failed: {e}", stage='unknown')

    def _export_yolo_to_onnx(
        self,
        pt_path: str,
        onnx_path: str,
        input_size: Tuple[int, int],
        opset_version: int
    ) -> str:
        """Export YOLO model using appropriate method based on version."""
        version = self._detect_yolo_version(pt_path)
        self._log(f"YOLO version detected: {version or 'unknown'}")
        self._progress(30)

        if version == 'v5':
            return self._export_yolov5_to_onnx(pt_path, onnx_path, input_size, opset_version)
        else:
            return self._export_yolov8_to_onnx(pt_path, onnx_path, input_size, opset_version)

    def _export_yolov5_to_onnx(
        self,
        pt_path: str,
        onnx_path: str,
        input_size: Tuple[int, int],
        opset_version: int
    ) -> str:
        """
        Export YOLOv5 model to ONNX using official YOLOv5 export.py.

        IMPORTANT: For Hailo compatibility, we MUST use YOLOv5 official export.py.
        This produces node names like /model.24/m.0/Conv that Hailo Model Zoo expects.

        DO NOT use Ultralytics CLI (yolo export) - it produces conv2d_N naming
        which is incompatible with Hailo.

        Opset versions:
        - Detection models: opset 11 (Hailo standard)
        - Segmentation models: opset 18 (required for Resize operator)
        """
        # Use passed opset_version (already adjusted for segment models by caller)
        # Minimum opset 11 for Hailo compatibility
        hailo_opset = max(opset_version, 11)
        self._log(f"Using YOLOv5 official export (opset {hailo_opset})...")

        try:
            import subprocess
            import shutil
            import sys

            self._progress(20)

            # Set environment for PyTorch 2.6+ compatibility
            env = os.environ.copy()
            env['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'

            # Use our wrapper script that patches torch.load for PyTorch 2.6+
            wrapper_script = Path(__file__).parent / 'yolov5_export_wrapper.py'

            self._progress(30)

            if wrapper_script.exists():
                # Use wrapper script (handles PyTorch 2.6+ weights_only issue)
                cmd = [
                    sys.executable, str(wrapper_script),
                    "--weights", pt_path,
                    "--img-size", str(input_size[0]), str(input_size[1]),
                    "--batch-size", "1",
                    "--device", "cpu",
                    "--include", "onnx",
                    "--opset", str(hailo_opset),
                    "--simplify"
                ]
                self._log(f"Running wrapper: {' '.join(cmd)}")
                self._progress(50)

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600,
                    env=env
                )

                if result.returncode != 0:
                    self._log(f"Wrapper stderr: {result.stderr}")
                    raise ExportError(
                        f"YOLOv5 export failed: {result.stderr}",
                        stage='yolov5_export'
                    )
            else:
                # Fallback: try direct import with patched torch.load
                self._log("Using direct yolov5 package export...")
                self._patch_torch_load_for_yolo()
                try:
                    from yolov5.export import run as yolov5_export
                    yolov5_export(
                        weights=pt_path,
                        imgsz=(input_size[0], input_size[1]),
                        batch_size=1,
                        device='cpu',
                        include=['onnx'],
                        opset=hailo_opset,
                        simplify=True
                    )
                except ImportError:
                    raise ExportError(
                        "YOLOv5 not installed.\n"
                        "Please install: pip install yolov5",
                        stage='yolov5_export'
                    )

            self._progress(80)

            # YOLOv5 export creates ONNX next to PT file
            # Move to desired output path
            default_onnx = Path(pt_path).with_suffix('.onnx')
            if default_onnx.exists() and str(default_onnx) != onnx_path:
                os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
                shutil.move(str(default_onnx), onnx_path)
                self._log(f"Moved ONNX to: {onnx_path}")
            elif not default_onnx.exists() and not Path(onnx_path).exists():
                raise ExportError("ONNX file not created", stage='yolov5_export')

            # Verify ONNX node naming
            self._log("Verifying ONNX compatibility...")
            nodes = extract_onnx_nodes(onnx_path)
            if nodes.get('naming_style') == 'yolov5_official':
                self._log(f"✓ ONNX has Hailo-compatible node naming: {nodes.get('detection_heads', [])[:2]}...")
            else:
                self._log(f"⚠ Warning: ONNX naming style is '{nodes.get('naming_style')}', may need --end-node-names")

            self._log(f"Exported to: {onnx_path}")
            self._progress(100)
            return onnx_path

        except subprocess.TimeoutExpired:
            raise ExportError("Export timed out (10 min)", stage='yolov5_export')
        except ExportError:
            raise
        except Exception as e:
            self._log(f"YOLOv5 export failed: {e}")
            raise ExportError(f"YOLOv5 export failed: {e}", stage='yolov5_export')

    def _export_yolov8_to_onnx(
        self,
        pt_path: str,
        onnx_path: str,
        input_size: Tuple[int, int],
        opset_version: int
    ) -> str:
        """Export YOLOv8 model using Ultralytics."""
        self._log("Using YOLOv8 export method...")

        try:
            from ultralytics import YOLO

            self._log("Loading YOLO model...")
            model = YOLO(pt_path)

            self._progress(50)
            self._log(f"Exporting to ONNX (opset={opset_version})...")

            export_path = model.export(
                format='onnx',
                imgsz=input_size[0],
                opset=opset_version,
                simplify=True,
                dynamic=False
            )

            self._progress(80)

            if str(export_path) != onnx_path:
                import shutil
                shutil.move(str(export_path), onnx_path)

            self._log(f"Exported to: {onnx_path}")
            self._progress(100)
            return onnx_path

        except ImportError:
            raise ExportError(
                "YOLOv8 export requires 'ultralytics' package. Install with: pip install ultralytics",
                stage='yolov8_import'
            )
        except Exception as e:
            raise ExportError(f"YOLOv8 export failed: {e}", stage='yolov8_export')

    def _export_generic_to_onnx(
        self,
        pt_path: str,
        onnx_path: str,
        input_size: Tuple[int, int],
        batch_size: int,
        opset_version: int
    ) -> str:
        """Export generic PyTorch model to ONNX."""
        try:
            import torch
        except ImportError:
            raise ExportError("PyTorch not installed", stage='import')

        self._progress(30)
        self._log("Loading PyTorch model...")

        try:
            checkpoint = torch.load(pt_path, map_location='cpu', weights_only=False)
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {e}", pt_path)

        # Extract model from checkpoint
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                model = checkpoint['model']
            elif 'state_dict' in checkpoint:
                raise ModelLoadError(
                    "Model file contains only state_dict. Full model required.",
                    pt_path
                )
            else:
                model = checkpoint
        else:
            model = checkpoint

        # Handle wrapped models
        if hasattr(model, 'model'):
            model = model.model

        model.eval()
        model.float()

        self._progress(50)
        self._log("Creating dummy input tensor...")

        # Create dummy input
        dummy_input = torch.randn(batch_size, 3, input_size[0], input_size[1])

        self._progress(60)
        self._log(f"Exporting to ONNX (opset={opset_version})...")

        try:
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                opset_version=opset_version,
                input_names=['images'],
                output_names=['output'],
                dynamic_axes={
                    'images': {0: 'batch'},
                    'output': {0: 'batch'}
                },
                do_constant_folding=True
            )
        except Exception as e:
            raise ExportError(f"ONNX export failed: {e}", stage='torch_export')

        self._progress(90)
        self._log(f"Successfully exported to: {onnx_path}")
        self._progress(100)
        return onnx_path

    def compile_onnx_to_hef(
        self,
        onnx_path: Optional[str],
        hef_path: str,
        calib_dir: str,
        target: str = "hailo8",
        calib_size: Tuple[int, int] = (640, 640),
        model_type: str = "detect",
        num_classes: int = 80,
        model_name: str = "yolov5s"
    ) -> str:
        """
        Compile ONNX model to Hailo HEF format.

        Args:
            onnx_path: Path to custom ONNX model, or None/empty to use Model Zoo standard
            hef_path: Output path for HEF model
            calib_dir: Directory containing calibration images
            target: Target Hailo device (hailo8, hailo8l, hailo15h)
            calib_size: Size to resize calibration images
            model_type: Model task type ('detect', 'segment', 'classify')
            num_classes: Number of classes in the model
            model_name: Base model name for hailomz (e.g., 'yolov5s', 'yolov8s')

        Returns:
            Path to compiled HEF file

        Raises:
            ModelLoadError: If ONNX file not found
            CalibrationError: If calibration data issues
            CompilationError: If HEF compilation fails
        """
        # Validate calibration directory
        if not os.path.exists(calib_dir):
            raise CalibrationError(
                f"Calibration directory not found: {calib_dir}",
                calib_dir, 0
            )

        self._log(f"Starting ONNX to HEF compilation...")
        self._log(f"Target: {target}, Model type: {model_type}")
        self._progress(10)

        # Model Zoo standard mode - no custom ONNX
        if not onnx_path or onnx_path.strip() == "":
            self._log("Model Zoo mode: Using standard pre-validated ONNX")
            self._compile_modelzoo_standard(
                hef_path, calib_dir, target, num_classes, model_name, model_type
            )
            return hef_path

        # Custom ONNX mode - validate file exists
        if not os.path.exists(onnx_path):
            raise ModelLoadError(f"ONNX file not found: {onnx_path}", onnx_path)

        # Extract end-node-names for custom ONNX
        nodes_info = extract_onnx_nodes(onnx_path)
        detection_heads = nodes_info.get('detection_heads', [])
        proto_node = nodes_info.get('proto_node')
        naming_style = nodes_info.get('naming_style', 'unknown')

        # Auto-detect segmentation model from proto node
        if proto_node and model_type != 'segment':
            self._log(f"Proto node detected: {proto_node}")
            self._log(f"Auto-switching model_type: {model_type} → segment")
            model_type = 'segment'

        # Auto-convert ONNX if naming is not Hailo compatible
        # This must happen BEFORE hailomz/SDK decision to ensure fallback uses converted ONNX
        if naming_style in ('ultralytics', 'pytorch_generic', 'unknown') and naming_style != 'yolov5_official':
            self._log(f"ONNX naming style '{naming_style}' is not Hailo compatible")
            self._log("Auto-converting node names to Hailo format...")

            # Determine model type for conversion
            conv_model_type = 'seg' if model_type == 'segment' else 'detect'

            try:
                converted_path, end_nodes_converted = rename_onnx_nodes_to_hailo_style(
                    onnx_path,
                    output_path=None,  # Auto-generate path with _hailo suffix
                    model_type=conv_model_type
                )
                self._log(f"Converted ONNX saved to: {converted_path}")
                self._log(f"End node names: {end_nodes_converted}")

                # Update variables for both hailomz and SDK fallback
                onnx_path = converted_path
                detection_heads = end_nodes_converted[:-1] if conv_model_type == 'seg' else end_nodes_converted
                proto_node = end_nodes_converted[-1] if conv_model_type == 'seg' else None

            except Exception as e:
                self._log(f"Auto-conversion failed: {e}, continuing with original ONNX")

        # Update end_nodes after potential conversion
        end_nodes = None
        if detection_heads:
            # CRITICAL: Proto node MUST come FIRST for segment models
            # Order: [proto_node, detection_head_0, detection_head_1, detection_head_2]
            if proto_node:
                end_nodes = [proto_node] + detection_heads
            else:
                end_nodes = detection_heads.copy()

        # Try hailomz CLI first for YOLO models
        if model_type in ('segment', 'detect'):
            try:
                return self._compile_with_hailomz(
                    onnx_path, hef_path, calib_dir, target, num_classes, model_name, model_type, calib_size
                )
            except CompilationError as e:
                # For custom ONNX models, ALWAYS fall back to SDK on any hailomz error
                # hailomz uses pre-configured .alls files designed for standard 80-class models
                # which often fail with custom models (different layer structure, NMS config, etc.)
                self._log(f"hailomz failed, falling back to Python SDK for custom model...")
                self._log(f"  hailomz error: {str(e)[:200]}...")
                return self._compile_with_sdk(
                    onnx_path, hef_path, calib_dir, target, calib_size,
                    end_node_names=end_nodes,
                    model_name=f"{model_name}_{model_type}_custom"
                )

        # Fallback to Python SDK for other models
        return self._compile_with_sdk(
            onnx_path, hef_path, calib_dir, target, calib_size,
            end_node_names=end_nodes,
            model_name=model_name
        )

    def _run_hailomz_command(self, cmd: list, hef_path: str) -> bool:
        """
        Execute hailomz command and handle output.

        Args:
            cmd: Command list to execute
            hef_path: Expected output HEF file path

        Returns:
            True if successful, raises CompilationError otherwise
        """
        import subprocess

        try:
            # Run hailomz compile
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            self._progress(80)

            if result.returncode != 0:
                self._log(f"hailomz stderr: {result.stderr}")
                raise CompilationError(
                    f"hailomz compile failed: {result.stderr}",
                    sdk_error=result.stderr
                )

            self._log(result.stdout)

            # hailomz outputs HEF to current directory with model name
            # Move to desired location if needed
            if os.path.exists(hef_path):
                self._log(f"Successfully compiled to: {hef_path}")
                self._progress(100)
                return True
            else:
                # Try to find generated HEF file
                import glob
                hef_files = glob.glob("*.hef")
                if hef_files:
                    import shutil
                    os.makedirs(os.path.dirname(hef_path), exist_ok=True)
                    shutil.move(hef_files[0], hef_path)
                    self._log(f"Moved {hef_files[0]} to {hef_path}")
                    self._progress(100)
                    return True
                else:
                    raise CompilationError(
                        f"HEF file not found after compilation: {hef_path}",
                        sdk_error="Output file missing"
                    )

        except subprocess.TimeoutExpired:
            raise CompilationError("Compilation timed out (1 hour)", sdk_error="Timeout")
        except FileNotFoundError:
            raise CompilationError(
                "hailomz command not found. Install Hailo Model Zoo: "
                "pip install hailo_model_zoo",
                sdk_error="hailomz not installed"
            )

    def _compile_modelzoo_standard(
        self,
        hef_path: str,
        calib_dir: str,
        target: str,
        num_classes: int,
        model_name: str,
        model_type: str
    ) -> bool:
        """
        Compile using Hailo Model Zoo standard model (auto-download).
        Does NOT use --ckpt parameter - hailomz downloads verified ONNX automatically.

        Args:
            hef_path: Output path for HEF model
            calib_dir: Directory containing calibration images
            target: Target Hailo device (hailo8, hailo8l, hailo15h)
            num_classes: Number of classes in the model
            model_name: Base model name (e.g., 'yolov5s', 'yolov8s')
            model_type: Model task type ('detect', 'segment', 'classify')

        Returns:
            True if successful
        """
        # Build model identifier (e.g., "yolov5s_seg" for segmentation)
        if model_type == 'segment':
            hailomz_model = f"{model_name}_seg"
        else:
            hailomz_model = model_name

        cmd = [
            "hailomz", "compile", hailomz_model,
            "--hw-arch", target,
            "--calib-path", calib_dir,
            "--classes", str(num_classes)
        ]
        # NOTE: No --ckpt, no --end-node-names (standard model has correct naming)

        self._log(f"Using Model Zoo standard model: {hailomz_model}")
        self._log(f"Command: {' '.join(cmd)}")
        self._progress(30)

        return self._run_hailomz_command(cmd, hef_path)

    def _compile_with_hailomz(
        self,
        onnx_path: str,
        hef_path: str,
        calib_dir: str,
        target: str,
        num_classes: int,
        model_name: str,
        model_type: str = "detect",
        calib_size: Tuple[int, int] = (640, 640)
    ) -> str:
        """
        Compile YOLO model using hailomz CLI.

        For custom ONNX models, prepares calibration data as numpy file
        to ensure correct NHWC format (Hailo expects height, width, channels).
        """
        import subprocess
        import tempfile

        # Determine model name for hailomz based on type
        if model_type == 'segment':
            hailomz_model = f"{model_name}_seg"
        else:
            hailomz_model = model_name  # detection: yolov5s, yolov8s
        self._log(f"Using hailomz compile for {model_type} model: {hailomz_model}")
        self._progress(10)

        # Extract ONNX node names to determine end-node-names
        self._log("Extracting ONNX node names...")
        nodes_info = extract_onnx_nodes(onnx_path)
        naming_style = nodes_info.get('naming_style', 'unknown')
        detection_heads = nodes_info.get('detection_heads', [])
        proto_node = nodes_info.get('proto_node')

        self._log(f"  Naming style: {naming_style}")
        self._log(f"  Detection heads: {detection_heads}")
        self._log(f"  Proto node: {proto_node}")

        self._progress(15)

        # Note: ONNX conversion now happens in compile_onnx_to_hef() before calling this method
        # This ensures the SDK fallback uses the same converted ONNX path

        self._progress(20)

        # Prepare calibration data as numpy file to ensure correct NHWC format
        # This bypasses hailomz's internal image loading which may have format issues
        self._log("Preparing calibration data (NHWC format for Hailo)...")

        # Create temp directory for calibration numpy file
        hef_dir = os.path.dirname(hef_path) or '.'
        calib_npy_dir = os.path.join(hef_dir, '.calib_cache')

        try:
            calib_npy_path = self._prepare_calibration_npy(
                calib_dir, calib_npy_dir, calib_size, max_images=500
            )
        except CalibrationError:
            raise
        except Exception as e:
            self._log(f"Warning: Could not prepare numpy calibration, using image folder: {e}")
            calib_npy_path = None

        self._progress(30)

        # Build command
        cmd = [
            "hailomz", "compile", hailomz_model,
            "--ckpt", onnx_path,
            "--hw-arch", target,
            "--classes", str(num_classes)
        ]

        # Use numpy file if available, otherwise use image folder
        if calib_npy_path and os.path.exists(calib_npy_path):
            cmd.extend(["--calib-path", calib_npy_path])
            self._log(f"Using numpy calibration: {calib_npy_path}")
        else:
            cmd.extend(["--calib-path", calib_dir])
            self._log(f"Using image folder: {calib_dir}")

        # Add --end-node-names for both segment and detect models if detected
        # This is required for custom ONNX exports with non-standard node names
        # Order matters: proto_node MUST come first for segment models
        if detection_heads:
            end_nodes = []
            if proto_node:
                end_nodes.append(proto_node)  # Proto node FIRST for segment models
            end_nodes.extend(detection_heads)  # Then detection heads
            if end_nodes:
                cmd.append("--end-node-names")
                cmd.extend(end_nodes)
                self._log(f"  Using end-node-names: {end_nodes}")

        self._log(f"Command: {' '.join(cmd)}")
        self._progress(35)

        self._run_hailomz_command(cmd, hef_path)
        return hef_path

    def _compile_with_sdk(
        self,
        onnx_path: str,
        hef_path: str,
        calib_dir: str,
        target: str,
        calib_size: Tuple[int, int],
        end_node_names: list = None,
        model_name: str = None
    ) -> str:
        """
        Compile model using Hailo Python SDK.

        This method provides more control over the compilation process
        and properly handles calibration data format (NHWC).

        Args:
            onnx_path: Path to ONNX model
            hef_path: Output path for HEF
            calib_dir: Directory with calibration images
            target: Target device (hailo8, hailo8l, etc.)
            calib_size: Input size (height, width)
            end_node_names: Optional list of end node names for network cutting
            model_name: Optional model name for HAR file
        """
        # CRITICAL: Skip GPU subprocess check (the real fix)
        # See: hailo_model_optimization/acceleras/utils/tf_utils.py line 65
        import os as _os
        _os.environ['HAILO_DISABLE_MO_SUB_PROCESS'] = '1'
        _os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        _os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        try:
            from hailo_sdk_client import ClientRunner
        except ImportError:
            raise CompilationError(
                "Hailo SDK (hailo_sdk_client) not installed. "
                "Please install the Hailo Dataflow Compiler.",
                sdk_error="ImportError"
            )

        try:
            self._log("Initializing Hailo compiler (CPU mode)...")
            runner = ClientRunner(hw_arch=target)

            self._progress(20)
            self._log(f"Translating ONNX model: {onnx_path}")

            # Prepare network name
            net_name = model_name or Path(onnx_path).stem

            # Translate ONNX to Hailo format with optional end nodes
            translate_kwargs = {
                'net_name': net_name
            }
            if end_node_names:
                translate_kwargs['end_node_names'] = end_node_names
                self._log(f"Using end-node-names: {end_node_names}")

            hn, npz = runner.translate_onnx_model(onnx_path, **translate_kwargs)

            self._progress(40)
            self._log("Loading calibration data...")

            # Load calibration data in NHWC format (Hailo standard)
            # Shape: (N, H, W, C) = (N, 640, 640, 3)
            calib_data = self._load_calibration_data(calib_dir, calib_size, layout='NHWC')
            self._log(f"Calibration data shape: {calib_data.shape}")

            self._progress(50)
            self._log(f"Running optimization with {len(calib_data)} calibration images...")

            # Optimize (quantization) - GPU subprocess disabled via HAILO_DISABLE_MO_SUB_PROCESS
            runner.optimize(calib_data)

            self._progress(80)
            self._log("Compiling to HEF...")

            # Compile
            hef = runner.compile()

            # Ensure output directory exists
            os.makedirs(os.path.dirname(hef_path), exist_ok=True)

            # Save HEF
            with open(hef_path, 'wb') as f:
                f.write(hef)

            self._log(f"Successfully compiled to: {hef_path}")
            self._progress(100)
            return hef_path

        except (ModelLoadError, CalibrationError, CompilationError):
            raise
        except Exception as e:
            raise CompilationError(f"Compilation failed: {e}", sdk_error=str(e))

    def _load_calibration_data(
        self,
        calib_dir: str,
        input_size: Tuple[int, int] = (640, 640),
        max_images: int = 500,
        layout: str = 'NHWC'
    ) -> np.ndarray:
        """
        Load calibration images from directory.

        Args:
            calib_dir: Directory containing calibration images
            input_size: Target size (height, width)
            max_images: Maximum number of images to load
            layout: Output layout - 'NHWC' (H, W, C) or 'NCHW' (C, H, W)
                   Hailo SDK expects NHWC (640, 640, 3)

        Returns:
            numpy array of shape (N, H, W, C) for NHWC or (N, C, H, W) for NCHW
        """
        from PIL import Image

        calib_path = Path(calib_dir)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

        images = []
        skipped = 0

        for f in sorted(calib_path.iterdir()):
            if f.suffix.lower() in image_extensions:
                if len(images) >= max_images:
                    break

                try:
                    img = Image.open(f).convert('RGB')
                    img = img.resize(input_size)
                    # PIL loads as (H, W, C) which is NHWC without batch
                    img_array = np.array(img).astype(np.float32) / 255.0

                    # Convert to NCHW if requested (for PyTorch-style models)
                    if layout == 'NCHW':
                        img_array = np.transpose(img_array, (2, 0, 1))  # (H,W,C) -> (C,H,W)

                    images.append(img_array)
                except Exception as e:
                    self._log(f"Warning: Skipping {f.name}: {e}")
                    skipped += 1

        if not images:
            raise CalibrationError(
                f"No valid calibration images found in {calib_dir}",
                calib_dir, 0
            )

        if skipped > 0:
            self._log(f"Skipped {skipped} invalid images")

        self._log(f"Loaded {len(images)} calibration images ({layout} format)")
        return np.stack(images, axis=0)

    def _prepare_calibration_npy(
        self,
        calib_dir: str,
        output_dir: str,
        input_size: Tuple[int, int] = (640, 640),
        max_images: int = 500
    ) -> str:
        """
        Prepare calibration data as numpy file for hailomz.

        Hailo expects NHWC format (height, width, channels).
        This bypasses hailomz's internal image loading which may have format issues.

        Args:
            calib_dir: Directory containing calibration images
            output_dir: Directory to save numpy file
            input_size: Target size (height, width)
            max_images: Maximum number of images

        Returns:
            Path to saved numpy file
        """
        # Load images in NHWC format (Hailo standard)
        calib_data = self._load_calibration_data(
            calib_dir, input_size, max_images, layout='NHWC'
        )

        # Save to numpy file
        os.makedirs(output_dir, exist_ok=True)
        npy_path = os.path.join(output_dir, 'calibration_data.npy')
        np.save(npy_path, calib_data)

        self._log(f"Saved calibration data: {npy_path}")
        self._log(f"  Shape: {calib_data.shape} (NHWC format)")
        return npy_path
