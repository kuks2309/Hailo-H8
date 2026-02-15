"""
ONNX Utilities for Hailo Compatibility.

Provides functions for:
- Detecting ONNX naming styles (YOLOv5 official vs PyTorch generic)
- Converting node names to Hailo-compatible format
- Patching PyTorch 2.x ONNX for Hailo SDK compatibility
- Validating ONNX compatibility with Hailo Model Zoo
- Extracting input/output nodes and detection heads from ONNX models
"""

import os
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any

import numpy as np

from utils.logger import setup_logger

logger = setup_logger('onnx_utils')

# Optional ONNX import - graceful degradation when not installed
try:
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


def patch_onnx_for_hailo(model: Any) -> None:
    """
    Patch ONNX model in-place for Hailo SDK compatibility.

    Fixes PyTorch 2.x export issues:
    1. Add missing kernel_shape attribute to Conv nodes
    2. Fix Resize nodes for opset 11 format (scales instead of sizes)
    3. Fix Split nodes to use attribute-based split (opset 11 style)
    4. Remove unsupported attributes (allowzero from Reshape)

    Args:
        model: Loaded ONNX model object (onnx.ModelProto).
    """
    if not ONNX_AVAILABLE:
        logger.warning("ONNX not installed - cannot patch model")
        return

    # Build initializer lookup
    initializers: Dict[str, Any] = {init.name: init for init in model.graph.initializer}

    # Track new initializers to add
    new_initializers: List[Any] = []

    # Create empty ROI tensor for Resize nodes (opset 11 format)
    empty_roi_name: str = '_hailo_empty_roi'
    if empty_roi_name not in initializers:
        empty_roi = numpy_helper.from_array(
            np.array([], dtype=np.float32), empty_roi_name
        )
        new_initializers.append(empty_roi)
        initializers[empty_roi_name] = empty_roi

    for node in model.graph.node:
        if node.op_type == 'Conv':
            # Fix 1: Add kernel_shape if missing
            has_kernel_shape: bool = any(a.name == 'kernel_shape' for a in node.attribute)
            if not has_kernel_shape and len(node.input) >= 2:
                weight_name: str = node.input[1]
                if weight_name in initializers:
                    weight = initializers[weight_name]
                    # Weight shape: (out_channels, in_channels, kH, kW)
                    if hasattr(weight, 'dims') and len(weight.dims) >= 4:
                        kernel_shape: List[int] = list(weight.dims[2:])
                        node.attribute.append(
                            helper.make_attribute('kernel_shape', kernel_shape)
                        )

        elif node.op_type == 'Resize':
            # Fix 2: Convert to opset 11 format
            # Opset 11 Resize inputs: [X, scales] (simpler format)
            # PyTorch 2.x exports with: [X, '', scales_or_empty, sizes] format

            inputs: List[str] = list(node.input)

            # Remove unsupported attributes for opset 11
            # These were added in later opsets (18+)
            unsupported_attrs: List[str] = [
                'antialias', 'keep_aspect_ratio_policy', 'axes', 'exclude_outside'
            ]
            attrs_to_remove: List[int] = [
                i for i, a in enumerate(node.attribute) if a.name in unsupported_attrs
            ]
            for i in reversed(attrs_to_remove):
                del node.attribute[i]

            # Find the actual scales or sizes tensor
            data_input: str = inputs[0]
            scales_input: Optional[str] = None
            sizes_input: Optional[str] = None

            for i, inp in enumerate(inputs[1:], 1):
                if inp and inp in initializers:
                    tensor = initializers[inp]
                    arr = numpy_helper.to_array(tensor)
                    if len(arr) > 0:
                        if i == 2:  # scales position
                            scales_input = inp
                        elif i == 3:  # sizes position
                            sizes_input = inp

            # For opset 11, we need: [X, roi, scales] or [X, roi, scales, sizes]
            # Create proper ROI tensor (empty float array)
            if scales_input or sizes_input:
                # Build new inputs
                new_inputs: List[str] = [data_input, empty_roi_name]
                if scales_input:
                    new_inputs.append(scales_input)
                elif sizes_input:
                    # Convert sizes to scales if needed
                    # For now, add empty scales and then sizes
                    empty_scales_name: str = f'_hailo_empty_scales_{node.name}'
                    if empty_scales_name not in initializers:
                        empty_scales = numpy_helper.from_array(
                            np.array([], dtype=np.float32), empty_scales_name
                        )
                        new_initializers.append(empty_scales)
                        initializers[empty_scales_name] = empty_scales
                    new_inputs.append(empty_scales_name)
                    new_inputs.append(sizes_input)

                # Update node inputs
                del node.input[:]
                node.input.extend(new_inputs)

        elif node.op_type == 'Split':
            # Fix 3: Convert Split from input-based to attribute-based (opset 11)
            # Opset 13+: split sizes as input[1]
            # Opset 11: split sizes as 'split' attribute

            has_split_attr: bool = any(a.name == 'split' for a in node.attribute)
            if not has_split_attr and len(node.input) >= 2:
                split_input: str = node.input[1]
                if split_input in initializers:
                    split_tensor = initializers[split_input]
                    split_values: List[int] = numpy_helper.to_array(split_tensor).tolist()

                    # Add as attribute
                    node.attribute.append(
                        helper.make_attribute('split', split_values)
                    )
                    # Remove the input (keep only data input)
                    del node.input[1:]

        elif node.op_type == 'Reshape':
            # Fix 4: Remove 'allowzero' attribute (not supported in opset 11)
            attrs_to_remove = [
                i for i, a in enumerate(node.attribute) if a.name == 'allowzero'
            ]
            for i in reversed(attrs_to_remove):
                del node.attribute[i]

    # Add new initializers
    for init in new_initializers:
        model.graph.initializer.append(init)


def downgrade_opset_version(model: Any, target_opset: int = 11) -> None:
    """
    Downgrade ONNX opset version.

    Note: This only changes the version number. Actual operator compatibility
    must be ensured by patch_onnx_for_hailo().

    Args:
        model: Loaded ONNX model object (onnx.ModelProto).
        target_opset: Target opset version (default: 11).
    """
    if not ONNX_AVAILABLE:
        logger.warning("ONNX not installed - cannot downgrade opset")
        return

    if model.opset_import:
        current_opset: int = model.opset_import[0].version
        if current_opset > target_opset:
            model.opset_import[0].version = target_opset


def detect_onnx_naming_style(onnx_path: str) -> str:
    """
    Detect the naming style of ONNX Conv nodes.

    Args:
        onnx_path: Path to the ONNX model file.

    Returns:
        'yolov5_official': /model.24/m.X/Conv (Hailo compatible)
        'yolov8_official': /model.22/cvX.X/Conv (Hailo compatible)
        'pytorch_generic': conv2d_N, node_conv2d_N, node_Conv_N (needs conversion)
        'unknown': Unable to determine or ONNX not installed
    """
    if not ONNX_AVAILABLE:
        logger.warning("ONNX not installed - cannot detect naming style")
        return 'unknown'

    try:
        model = onnx.load(onnx_path)
        conv_nodes = [n for n in model.graph.node if n.op_type == 'Conv']

        if not conv_nodes:
            return 'unknown'

        # Check multiple Conv nodes for pattern detection
        conv_names: List[str] = [n.name for n in conv_nodes[:10]]  # Check first 10

        for name in conv_names:
            # YOLOv5 official export style
            if name.startswith('/model.'):
                return 'yolov5_official'

            # YOLOv8 style
            if '/model.22/' in name:
                return 'yolov8_official'

        # Check for PyTorch generic patterns in any of the names
        for name in conv_names:
            name_lower: str = name.lower()
            # Patterns: conv2d_N, node_conv2d_N, node_Conv_N, Conv_N
            if 'conv2d' in name_lower:
                return 'pytorch_generic'
            if name.startswith('Conv_') or name.startswith('node_Conv_'):
                return 'pytorch_generic'

        return 'unknown'

    except Exception:
        return 'unknown'


def is_hailo_compatible_naming(onnx_path: str) -> bool:
    """
    Check if ONNX node naming is compatible with Hailo Model Zoo.

    Args:
        onnx_path: Path to the ONNX model file.

    Returns:
        True if naming style is Hailo-compatible, False otherwise.
    """
    style: str = detect_onnx_naming_style(onnx_path)
    return style in ('yolov5_official', 'yolov8_official')


def build_yolov5_conv_mapping(num_convs: int, model_type: str = 'seg') -> Dict[int, str]:
    """
    Build mapping from sequential conv index to YOLOv5 layer path.

    Based on YOLOv5s/m structure analysis.
    Segmentation model has additional proto layer.

    Args:
        num_convs: Total number of Conv nodes in the model.
        model_type: 'seg' for segmentation or 'detect' for detection.

    Returns:
        Dictionary mapping conv index to YOLOv5-style layer path.
    """
    mapping: Dict[int, str] = {}

    # For segmentation models (yolov5s-seg has ~63 Conv nodes)
    if model_type == 'seg':
        if num_convs >= 60:
            mapping[num_convs - 4] = '/model.24/proto/cv3/conv/Conv'
            mapping[num_convs - 3] = '/model.24/m.0/Conv'
            mapping[num_convs - 2] = '/model.24/m.1/Conv'
            mapping[num_convs - 1] = '/model.24/m.2/Conv'
    else:
        # Detection model
        if num_convs >= 50:
            mapping[num_convs - 3] = '/model.24/m.0/Conv'
            mapping[num_convs - 2] = '/model.24/m.1/Conv'
            mapping[num_convs - 1] = '/model.24/m.2/Conv'

    # Backbone layers (first ~20 convs)
    backbone_map: List[str] = [
        '/model.0/conv/Conv',           # 0
        '/model.1/conv/Conv',           # 1
        '/model.2/cv1/conv/Conv',       # 2
        '/model.2/m/m.0/cv1/conv/Conv', # 3
        '/model.2/m/m.0/cv2/conv/Conv', # 4
        '/model.2/cv2/conv/Conv',       # 5
        '/model.3/conv/Conv',           # 6
        '/model.4/cv1/conv/Conv',       # 7
        '/model.4/m/m.0/cv1/conv/Conv', # 8
        '/model.4/m/m.0/cv2/conv/Conv', # 9
        '/model.4/m/m.1/cv1/conv/Conv', # 10
        '/model.4/m/m.1/cv2/conv/Conv', # 11
        '/model.4/cv2/conv/Conv',       # 12
        '/model.5/conv/Conv',           # 13
        '/model.6/cv1/conv/Conv',       # 14
        '/model.6/m/m.0/cv1/conv/Conv', # 15
        '/model.6/m/m.0/cv2/conv/Conv', # 16
        '/model.6/m/m.1/cv1/conv/Conv', # 17
        '/model.6/m/m.1/cv2/conv/Conv', # 18
        '/model.6/m/m.2/cv1/conv/Conv', # 19
        '/model.6/m/m.2/cv2/conv/Conv', # 20
    ]

    for i, path in enumerate(backbone_map):
        if i not in mapping:
            mapping[i] = path

    return mapping


def rename_onnx_nodes_to_hailo_style(
    onnx_path: str,
    output_path: Optional[str] = None,
    model_type: str = 'seg'
) -> Tuple[str, List[str]]:
    """
    Rename ONNX nodes from PyTorch style to Hailo-compatible YOLOv5 style.

    Args:
        onnx_path: Path to input ONNX file.
        output_path: Path for output ONNX file (default: adds _hailo suffix).
        model_type: 'seg' or 'detect'.

    Returns:
        Tuple of (output_path, end_node_names).
        Returns (onnx_path, []) if ONNX is not installed.
    """
    if not ONNX_AVAILABLE:
        logger.warning("ONNX not installed - cannot rename nodes")
        return onnx_path, []

    model = onnx.load(onnx_path)

    # Get all Conv nodes
    conv_nodes = [n for n in model.graph.node if n.op_type == 'Conv']
    num_convs: int = len(conv_nodes)

    # Check current naming style
    first_conv: str = conv_nodes[0].name if conv_nodes else ''
    if first_conv.startswith('/model.'):
        # Already Hailo compatible
        end_nodes: List[str] = get_end_node_names_from_model(model, model_type)
        return onnx_path, end_nodes

    # Build mapping
    mapping: Dict[int, str] = build_yolov5_conv_mapping(num_convs, model_type)

    # Create mapping ONLY for Conv output tensors (not node names)
    # This maps old output tensor name -> new output tensor name
    output_name_mapping: Dict[str, str] = {}

    # First pass: Rename Conv nodes and their outputs, build output mapping
    for idx, node in enumerate(conv_nodes):
        if idx in mapping:
            new_name: str = mapping[idx]
        else:
            # Generate sequential name for unmapped nodes
            new_name = f'/model.backbone/conv{idx}/Conv'

        # Rename node
        node.name = new_name

        # Rename output tensor and track the mapping
        if node.output:
            old_output: str = node.output[0]
            new_output: str = new_name.replace('/', '_').lstrip('_') + '_output'
            output_name_mapping[old_output] = new_output
            node.output[0] = new_output

    # Second pass: Update inputs that reference renamed Conv outputs
    # IMPORTANT: Only update inputs, NOT outputs of non-Conv nodes
    for node in model.graph.node:
        # Update inputs that reference renamed Conv outputs
        for i, inp in enumerate(node.input):
            if inp in output_name_mapping:
                node.input[i] = output_name_mapping[inp]

    # Update graph outputs if they reference renamed Conv outputs
    for output in model.graph.output:
        if output.name in output_name_mapping:
            output.name = output_name_mapping[output.name]

    # Apply PyTorch 2.x compatibility patches
    # This fixes: missing kernel_shape, Resize format, Split format, etc.
    patch_onnx_for_hailo(model)

    # Downgrade opset version for better Hailo compatibility
    # Hailo SDK works best with opset 11-13
    downgrade_opset_version(model, target_opset=11)

    # Determine output path
    if output_path is None:
        base: Path = Path(onnx_path)
        output_path = str(base.parent / f"{base.stem}_hailo{base.suffix}")

    # Validate the model before saving
    try:
        onnx.checker.check_model(model)
    except Exception as e:
        # Log warning but continue - Hailo may still accept it
        logger.warning(f"ONNX model validation warning (continuing anyway): {e}")

    # Save modified model
    onnx.save(model, output_path)

    # Get end node names from modified model
    end_nodes = get_end_node_names_from_model(model, model_type)

    return output_path, end_nodes


def get_end_node_names_from_model(model: Any, model_type: str = 'seg') -> List[str]:
    """
    Extract the end node names for hailomz compile command from a loaded model.

    Args:
        model: Loaded ONNX model object (onnx.ModelProto).
        model_type: 'seg' for segmentation (last 4 Conv) or 'detect' (last 3 Conv).

    Returns:
        List of end node names. Empty list if ONNX is not available.
    """
    if not ONNX_AVAILABLE:
        return []

    conv_nodes = [n for n in model.graph.node if n.op_type == 'Conv']

    if model_type == 'seg':
        # Last 4 Conv nodes for segmentation
        end_nodes: List[str] = [n.name for n in conv_nodes[-4:]]
    else:
        # Last 3 Conv nodes for detection
        end_nodes = [n.name for n in conv_nodes[-3:]]

    return end_nodes


def get_end_node_names(onnx_path: str, model_type: str = 'seg') -> List[str]:
    """
    Extract the end node names for hailomz compile command from a file path.

    Args:
        onnx_path: Path to the ONNX model file.
        model_type: 'seg' for segmentation or 'detect' for detection.

    Returns:
        List of end node names. Empty list if ONNX is not available.
    """
    if not ONNX_AVAILABLE:
        logger.warning("ONNX not installed - cannot extract end node names")
        return []

    model = onnx.load(onnx_path)
    return get_end_node_names_from_model(model, model_type)


def validate_onnx_hailo_compatibility(onnx_path: str) -> Dict[str, Any]:
    """
    Validate if ONNX is compatible with Hailo Model Zoo.

    Args:
        onnx_path: Path to the ONNX model file.

    Returns:
        dict with 'compatible', 'naming_style', 'errors', 'warnings', 'recommended_action'.
        Returns a default dict with compatible=False if ONNX is not installed.
    """
    result: Dict[str, Any] = {
        'compatible': True,
        'naming_style': 'unknown',
        'errors': [],
        'warnings': [],
        'recommended_action': None
    }

    if not ONNX_AVAILABLE:
        result['compatible'] = False
        result['errors'].append("ONNX package is not installed")
        result['recommended_action'] = "Install ONNX: pip install onnx"
        return result

    try:
        naming_style: str = detect_onnx_naming_style(onnx_path)
        result['naming_style'] = naming_style

        if naming_style == 'pytorch_generic':
            result['compatible'] = False
            result['errors'].append(
                f"ONNX node naming style '{naming_style}' is not compatible with Hailo Model Zoo"
            )
            result['recommended_action'] = (
                "Use auto-conversion to rename nodes to Hailo-compatible format, "
                "or re-export using YOLOv5 official export.py"
            )
        elif naming_style == 'unknown':
            result['warnings'].append(
                "Unable to determine ONNX naming style. Compilation may fail."
            )

        # Check opset version
        model = onnx.load(onnx_path)
        opset: int = model.opset_import[0].version if model.opset_import else 0

        if opset > 13:
            result['warnings'].append(
                f"ONNX opset {opset} may have limited Hailo support. Recommended: opset 11-13"
            )

    except Exception as e:
        result['compatible'] = False
        result['errors'].append(f"Failed to validate ONNX: {e}")

    return result


def verify_onnx_for_hailo(onnx_path: str) -> Dict[str, Any]:
    """
    Comprehensive ONNX verification before HEF compilation.

    Performs 6 checks:
    1. ONNX file validity (checker)
    2. Opset version range
    3. Node naming compatibility
    4. Input shape format (channels, square)
    5. Output node count
    6. Dynamic shape detection

    Args:
        onnx_path: Path to the ONNX model file.

    Returns:
        dict with 'valid', 'issues', 'suggestions', 'metadata'.
    """
    issues: List[str] = []
    suggestions: List[str] = []
    metadata: Dict[str, Any] = {}

    if not ONNX_AVAILABLE:
        return {
            'valid': False,
            'issues': ['ONNX package is not installed'],
            'suggestions': ['Install ONNX: pip install onnx'],
            'metadata': {}
        }

    try:
        from onnx import checker

        # 1. ONNX file validity
        model = onnx.load(onnx_path)
        try:
            checker.check_model(model)
        except Exception as e:
            issues.append(f"ONNX validation failed: {e}")

        # 2. Opset version check
        opset = model.opset_import[0].version if model.opset_import else 0
        metadata['opset'] = opset
        if opset < 11:
            issues.append(f"Opset {opset} too old (minimum: 11)")
        elif opset > 18:
            suggestions.append(f"Opset {opset} may not be fully supported")

        # 3. Node naming compatibility
        compat = validate_onnx_hailo_compatibility(onnx_path)
        metadata['naming_style'] = compat.get('naming_style', 'unknown')
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
            'suggestions': ['Ensure the file is a valid ONNX model'],
            'metadata': {}
        }


def detect_yolo_base_model(onnx_path: str) -> str:
    """
    Detect YOLO base model name from ONNX structure.

    Determines model size (n/s/m/l/x) from Conv node count
    and YOLO version from node naming patterns.

    Args:
        onnx_path: Path to the ONNX model file.

    Returns:
        Model name string (e.g., 'yolov5s', 'yolov8m').
        Defaults to 'yolov5s' on failure.
    """
    if not ONNX_AVAILABLE:
        return 'yolov5s'

    try:
        model = onnx.load(onnx_path)
        conv_count = sum(1 for node in model.graph.node if node.op_type == 'Conv')

        node_info = extract_onnx_nodes(onnx_path)
        yolo_version = node_info.get('yolo_version', 'unknown')

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

    Args:
        base_model: Base model name (yolov5s, yolov8s, etc.).
        onnx_path: Path to ONNX model.
        output_yaml_path: Where to save generated YAML.
        num_classes: Number of classes.
        model_type: 'detect' or 'segment'.

    Returns:
        Path to generated YAML file.
    """
    import yaml

    nodes = extract_onnx_nodes(onnx_path)
    input_nodes = nodes.get('input_nodes', [])
    detection_heads = nodes.get('detection_heads', [])
    proto_node = nodes.get('proto_node')

    input_node = input_nodes[0] if input_nodes else 'images'

    if 'v8' in base_model:
        base_yaml = 'base/yolov8_seg.yaml' if model_type == 'segment' else 'base/yolov8.yaml'
        meta_arch = 'yolov8_seg' if model_type == 'segment' else 'yolov8'
    else:
        base_yaml = 'base/yolov5_seg.yaml' if model_type == 'segment' else 'base/yolov5.yaml'
        meta_arch = 'yolov5_seg' if model_type == 'segment' else 'yolov5'

    if model_type == 'segment' and detection_heads:
        end_nodes = detection_heads[::-1]
        if proto_node:
            end_nodes = [proto_node] + end_nodes
    elif detection_heads:
        end_nodes = detection_heads[::-1]
    else:
        end_nodes = None

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

    out_dir = os.path.dirname(output_yaml_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    return output_yaml_path


def extract_onnx_nodes(onnx_path: str) -> Dict[str, Any]:
    """
    Extract input/output and internal node names from ONNX model.
    Specifically looks for YOLOv5/v8 detection head nodes for Hailo compatibility.

    Supports two ONNX export patterns:
    1. YOLOv5 official export.py: /model.24/m.0/Conv, /model.24/proto/cv3/conv/Conv
    2. Ultralytics/torch.onnx: conv2d_60, conv2d_61, conv2d_62

    Args:
        onnx_path: Path to the ONNX model file.

    Returns:
        dict with 'input_nodes', 'output_nodes', 'detection_heads', 'proto_node',
        'naming_style', 'yolo_version'. Returns empty lists and error on failure.
    """
    if not ONNX_AVAILABLE:
        logger.warning("ONNX not installed - cannot extract nodes")
        return {
            'input_nodes': [],
            'output_nodes': [],
            'detection_heads': [],
            'proto_node': None,
            'naming_style': 'unknown',
            'yolo_version': 'unknown',
            'error': 'ONNX package is not installed'
        }

    try:
        model = onnx.load(onnx_path)

        input_nodes: List[str] = [inp.name for inp in model.graph.input]
        output_nodes: List[str] = [out.name for out in model.graph.output]

        detection_heads: List[str] = []
        proto_node: Optional[str] = None
        naming_style: str = 'unknown'

        # Pattern 1: YOLOv5 official export.py (module path naming)
        # /model.24/m.0/Conv, /model.24/m.1/Conv, /model.24/m.2/Conv
        # /model.24/proto/cv3/conv/Conv
        model24_detect: List[str] = []
        model24_proto: Optional[str] = None

        for node in model.graph.node:
            # Detection heads: /model.24/m.X/Conv
            if '/model.24/m.' in node.name and '/Conv' in node.name:
                model24_detect.append(node.name)
            # Proto: /model.24/proto/cv3/conv/Conv
            if '/model.24/proto/' in node.name and '/Conv' in node.name:
                model24_proto = node.name

        if model24_detect:
            # Sort by m.X number (m.0, m.1, m.2)
            model24_detect = sorted(
                model24_detect,
                key=lambda x: int(x.split('/m.')[1].split('/')[0])
            )
            detection_heads = model24_detect
            proto_node = model24_proto
            naming_style = 'yolov5_official'

        # Pattern 3: YOLOv8 (module path naming - any export method)
        # /model.22/cv2.0/cv2.0.2/Conv, /model.22/cv2.1/cv2.1.2/Conv, ...
        # /model.22/cv3.0/cv3.0.2/Conv, /model.22/cv3.1/cv3.1.2/Conv, ...
        if not detection_heads:
            model22_cv2: List[str] = []
            model22_cv3: List[str] = []
            model22_cv4: List[str] = []
            model22_proto: Optional[str] = None

            for node in model.graph.node:
                # Only match final Conv in each branch (exclude intermediate /conv/Conv)
                if '/model.22/cv2.' in node.name and node.name.endswith('/Conv') and '/conv/Conv' not in node.name:
                    model22_cv2.append(node.name)
                if '/model.22/cv3.' in node.name and node.name.endswith('/Conv') and '/conv/Conv' not in node.name:
                    model22_cv3.append(node.name)
                if '/model.22/cv4.' in node.name and node.name.endswith('/Conv') and '/conv/Conv' not in node.name:
                    model22_cv4.append(node.name)
                if '/model.22/proto/' in node.name and ('/Mul' in node.name or node.name.endswith('/Conv')):
                    model22_proto = node.name

            if model22_cv2 or model22_cv3:
                # YOLOv8 detection + segmentation heads
                detection_heads = sorted(model22_cv2 + model22_cv3 + model22_cv4)
                naming_style = 'yolov8_official'
                if model22_proto:
                    proto_node = model22_proto

        if not detection_heads:
            # Pattern 2: Ultralytics/torch.onnx (auto-generated naming)
            # YOLOv5-seg: conv2d_60, conv2d_61, conv2d_62 are detection heads
            # Proto is silu_59 or similar

            # First, look for conv2d_N pattern (PyTorch 2.x Ultralytics export)
            conv2d_nodes: List[Dict[str, str]] = []
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
                conv_nodes: List[Dict[str, str]] = []
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
            yolo_version: str = 'v5'
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
        logger.error(f"Failed to extract ONNX nodes: {e}")
        return {
            'input_nodes': [],
            'output_nodes': [],
            'detection_heads': [],
            'proto_node': None,
            'naming_style': 'unknown',
            'yolo_version': 'unknown',
            'error': str(e)
        }
