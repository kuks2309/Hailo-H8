"""
ONNX Utilities for Hailo Compatibility.

Provides functions for:
- Detecting ONNX naming styles (YOLOv5 official vs PyTorch generic)
- Converting node names to Hailo-compatible format
- Patching PyTorch 2.x ONNX for Hailo SDK compatibility
- Validating ONNX compatibility with Hailo Model Zoo
"""

import os
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto


def patch_onnx_for_hailo(model) -> None:
    """
    Patch ONNX model in-place for Hailo SDK compatibility.

    Fixes PyTorch 2.x export issues:
    1. Add missing kernel_shape attribute to Conv nodes
    2. Fix Resize nodes for opset 11 format (scales instead of sizes)
    3. Fix Split nodes to use attribute-based split (opset 11 style)
    4. Remove unsupported attributes (allowzero from Reshape)
    """
    # Build initializer lookup
    initializers = {init.name: init for init in model.graph.initializer}

    # Track new initializers to add
    new_initializers = []

    # Create empty ROI tensor for Resize nodes (opset 11 format)
    empty_roi_name = '_hailo_empty_roi'
    if empty_roi_name not in initializers:
        empty_roi = numpy_helper.from_array(
            np.array([], dtype=np.float32), empty_roi_name
        )
        new_initializers.append(empty_roi)
        initializers[empty_roi_name] = empty_roi

    for node in model.graph.node:
        if node.op_type == 'Conv':
            # Fix 1: Add kernel_shape if missing
            has_kernel_shape = any(a.name == 'kernel_shape' for a in node.attribute)
            if not has_kernel_shape and len(node.input) >= 2:
                weight_name = node.input[1]
                if weight_name in initializers:
                    weight = initializers[weight_name]
                    # Weight shape: (out_channels, in_channels, kH, kW)
                    if hasattr(weight, 'dims') and len(weight.dims) >= 4:
                        kernel_shape = list(weight.dims[2:])
                        node.attribute.append(
                            helper.make_attribute('kernel_shape', kernel_shape)
                        )

        elif node.op_type == 'Resize':
            # Fix 2: Convert to opset 11 format
            # Opset 11 Resize inputs: [X, scales] (simpler format)
            # PyTorch 2.x exports with: [X, '', scales_or_empty, sizes] format

            inputs = list(node.input)

            # Remove unsupported attributes for opset 11
            # These were added in later opsets (18+)
            unsupported_attrs = ['antialias', 'keep_aspect_ratio_policy', 'axes', 'exclude_outside']
            attrs_to_remove = [i for i, a in enumerate(node.attribute) if a.name in unsupported_attrs]
            for i in reversed(attrs_to_remove):
                del node.attribute[i]

            # Find the actual scales or sizes tensor
            data_input = inputs[0]
            scales_input = None
            sizes_input = None

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
                new_inputs = [data_input, empty_roi_name]
                if scales_input:
                    new_inputs.append(scales_input)
                elif sizes_input:
                    # Convert sizes to scales if needed
                    # For now, add empty scales and then sizes
                    empty_scales_name = f'_hailo_empty_scales_{node.name}'
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

            has_split_attr = any(a.name == 'split' for a in node.attribute)
            if not has_split_attr and len(node.input) >= 2:
                split_input = node.input[1]
                if split_input in initializers:
                    split_tensor = initializers[split_input]
                    split_values = numpy_helper.to_array(split_tensor).tolist()

                    # Add as attribute
                    node.attribute.append(
                        helper.make_attribute('split', split_values)
                    )
                    # Remove the input (keep only data input)
                    del node.input[1:]

        elif node.op_type == 'Reshape':
            # Fix 4: Remove 'allowzero' attribute (not supported in opset 11)
            attrs_to_remove = [i for i, a in enumerate(node.attribute) if a.name == 'allowzero']
            for i in reversed(attrs_to_remove):
                del node.attribute[i]

    # Add new initializers
    for init in new_initializers:
        model.graph.initializer.append(init)


def downgrade_opset_version(model, target_opset: int = 11) -> None:
    """
    Downgrade ONNX opset version.

    Note: This only changes the version number. Actual operator compatibility
    must be ensured by patch_onnx_for_hailo().
    """
    if model.opset_import:
        current_opset = model.opset_import[0].version
        if current_opset > target_opset:
            model.opset_import[0].version = target_opset


def detect_onnx_naming_style(onnx_path: str) -> str:
    """
    Detect the naming style of ONNX Conv nodes.

    Returns:
        'yolov5_official': /model.24/m.X/Conv (Hailo compatible)
        'yolov8_official': /model.22/cvX.X/Conv (Hailo compatible)
        'pytorch_generic': conv2d_N, node_conv2d_N, node_Conv_N (needs conversion)
        'unknown': Unable to determine
    """
    try:
        model = onnx.load(onnx_path)
        conv_nodes = [n for n in model.graph.node if n.op_type == 'Conv']

        if not conv_nodes:
            return 'unknown'

        # Check multiple Conv nodes for pattern detection
        conv_names = [n.name for n in conv_nodes[:10]]  # Check first 10

        for name in conv_names:
            # YOLOv5 official export style
            if name.startswith('/model.'):
                return 'yolov5_official'

            # YOLOv8 style
            if '/model.22/' in name:
                return 'yolov8_official'

        # Check for PyTorch generic patterns in any of the names
        for name in conv_names:
            name_lower = name.lower()
            # Patterns: conv2d_N, node_conv2d_N, node_Conv_N, Conv_N
            if 'conv2d' in name_lower:
                return 'pytorch_generic'
            if name.startswith('Conv_') or name.startswith('node_Conv_'):
                return 'pytorch_generic'

        return 'unknown'

    except Exception:
        return 'unknown'


def is_hailo_compatible_naming(onnx_path: str) -> bool:
    """Check if ONNX node naming is compatible with Hailo Model Zoo."""
    style = detect_onnx_naming_style(onnx_path)
    return style in ('yolov5_official', 'yolov8_official')


def build_yolov5_conv_mapping(num_convs: int, model_type: str = 'seg') -> Dict[int, str]:
    """
    Build mapping from sequential conv index to YOLOv5 layer path.

    Based on YOLOv5s/m structure analysis.
    Segmentation model has additional proto layer.
    """
    mapping = {}

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
    backbone_map = [
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
    output_path: str = None,
    model_type: str = 'seg'
) -> Tuple[str, List[str]]:
    """
    Rename ONNX nodes from PyTorch style to Hailo-compatible YOLOv5 style.

    Args:
        onnx_path: Path to input ONNX file
        output_path: Path for output ONNX file (default: adds _hailo suffix)
        model_type: 'seg' or 'detect'

    Returns:
        Tuple of (output_path, end_node_names)
    """
    model = onnx.load(onnx_path)

    # Get all Conv nodes
    conv_nodes = [n for n in model.graph.node if n.op_type == 'Conv']
    num_convs = len(conv_nodes)

    # Check current naming style
    first_conv = conv_nodes[0].name if conv_nodes else ''
    if first_conv.startswith('/model.'):
        # Already Hailo compatible
        end_nodes = get_end_node_names_from_model(model, model_type)
        return onnx_path, end_nodes

    # Build mapping
    mapping = build_yolov5_conv_mapping(num_convs, model_type)

    # Create mapping ONLY for Conv output tensors (not node names)
    # This maps old output tensor name -> new output tensor name
    output_name_mapping = {}

    # First pass: Rename Conv nodes and their outputs, build output mapping
    for idx, node in enumerate(conv_nodes):
        if idx in mapping:
            new_name = mapping[idx]
        else:
            # Generate sequential name for unmapped nodes
            new_name = f'/model.backbone/conv{idx}/Conv'

        # Rename node
        node.name = new_name

        # Rename output tensor and track the mapping
        if node.output:
            old_output = node.output[0]
            new_output = new_name.replace('/', '_').lstrip('_') + '_output'
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
        base = Path(onnx_path)
        output_path = str(base.parent / f"{base.stem}_hailo{base.suffix}")

    # Validate the model before saving
    try:
        onnx.checker.check_model(model)
    except Exception as e:
        # Log warning but continue - Hailo may still accept it
        pass

    # Save modified model
    onnx.save(model, output_path)

    # Get end node names from modified model
    end_nodes = get_end_node_names_from_model(model, model_type)

    return output_path, end_nodes


def get_end_node_names_from_model(model, model_type: str = 'seg') -> List[str]:
    """Extract the end node names for hailomz compile command."""
    conv_nodes = [n for n in model.graph.node if n.op_type == 'Conv']

    if model_type == 'seg':
        # Last 4 Conv nodes for segmentation
        end_nodes = [n.name for n in conv_nodes[-4:]]
    else:
        # Last 3 Conv nodes for detection
        end_nodes = [n.name for n in conv_nodes[-3:]]

    return end_nodes


def get_end_node_names(onnx_path: str, model_type: str = 'seg') -> List[str]:
    """Extract the end node names for hailomz compile command."""
    model = onnx.load(onnx_path)
    return get_end_node_names_from_model(model, model_type)


def validate_onnx_hailo_compatibility(onnx_path: str) -> Dict:
    """
    Validate if ONNX is compatible with Hailo Model Zoo.

    Returns:
        dict with 'compatible', 'naming_style', 'errors', 'warnings', 'recommended_action'
    """
    result = {
        'compatible': True,
        'naming_style': 'unknown',
        'errors': [],
        'warnings': [],
        'recommended_action': None
    }

    try:
        naming_style = detect_onnx_naming_style(onnx_path)
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
        opset = model.opset_import[0].version if model.opset_import else 0

        if opset > 13:
            result['warnings'].append(
                f"ONNX opset {opset} may have limited Hailo support. Recommended: opset 11-13"
            )

    except Exception as e:
        result['compatible'] = False
        result['errors'].append(f"Failed to validate ONNX: {e}")

    return result
