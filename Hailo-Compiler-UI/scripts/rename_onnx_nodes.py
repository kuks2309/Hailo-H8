#!/usr/bin/env python3
"""
ONNX Node Renaming Script for Hailo Compatibility.

Converts PyTorch 2.x style node names (node_conv2d_N, conv2d_N) to
YOLOv5 official export style (/model.X/conv/Conv) for Hailo Model Zoo compatibility.

Usage:
    python rename_onnx_nodes.py input.onnx output.onnx [--model-type seg|detect]

Reference:
    - Hailo DFC Guide (withus_dfc_yolov5m)
    - zaiv training guide
"""

import argparse
import sys
from pathlib import Path

import onnx
from onnx import helper


def analyze_yolov5_structure(model):
    """
    Analyze YOLOv5 ONNX structure and create node mapping.

    YOLOv5 backbone/neck structure:
    - model.0-9: Backbone (stem + stages)
    - model.10-23: Neck (PANet)
    - model.24: Detection head (Detect/Segment)
    """
    conv_nodes = []
    other_nodes = []

    for node in model.graph.node:
        if node.op_type == 'Conv':
            conv_nodes.append(node)
        else:
            other_nodes.append(node)

    print(f"Total Conv nodes: {len(conv_nodes)}")
    print(f"Total other nodes: {len(other_nodes)}")

    return conv_nodes, other_nodes


def build_yolov5_conv_mapping(num_convs: int, model_type: str = 'detect'):
    """
    Build mapping from sequential conv index to YOLOv5 layer path.

    Based on YOLOv5s/m structure analysis.
    Segmentation model has additional proto layer.
    """
    # YOLOv5s/m backbone+neck structure (approximate mapping)
    # This maps sequential Conv indices to YOLOv5 model path

    mapping = {}

    # Standard YOLOv5 structure (simplified)
    # Backbone: model.0 - model.9
    # Neck: model.10 - model.23
    # Head: model.24

    # For segmentation models (yolov5s-seg has ~63 Conv nodes)
    if model_type == 'seg':
        # Detection head outputs (last 4 for seg)
        # model.24/m.0/Conv, model.24/m.1/Conv, model.24/m.2/Conv
        # model.24/proto/cv3/conv/Conv (for segmentation)

        if num_convs >= 60:
            # Segmentation model
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


def rename_nodes_to_hailo_style(model, model_type: str = 'detect'):
    """
    Rename ONNX nodes from PyTorch style to Hailo-compatible YOLOv5 style.

    Args:
        model: ONNX model
        model_type: 'detect' or 'seg'

    Returns:
        Modified ONNX model with renamed nodes
    """
    # Get all Conv nodes
    conv_nodes = [n for n in model.graph.node if n.op_type == 'Conv']
    num_convs = len(conv_nodes)

    print(f"\nAnalyzing ONNX structure...")
    print(f"  Total Conv nodes: {num_convs}")
    print(f"  Model type: {model_type}")

    # Check current naming style
    first_conv = conv_nodes[0].name if conv_nodes else ''
    if first_conv.startswith('/model.'):
        print(f"  Current style: YOLOv5 official (already Hailo compatible)")
        return model, False
    elif 'conv2d' in first_conv.lower():
        print(f"  Current style: PyTorch/Ultralytics (needs conversion)")
    else:
        print(f"  Current style: Unknown ({first_conv})")

    # Build mapping
    mapping = build_yolov5_conv_mapping(num_convs, model_type)

    # Create name mapping for all references
    old_to_new = {}

    # Rename Conv nodes
    print(f"\nRenaming Conv nodes...")
    for idx, node in enumerate(conv_nodes):
        old_name = node.name

        if idx in mapping:
            new_name = mapping[idx]
        else:
            # Generate sequential name for unmapped nodes
            new_name = f'/model.backbone/conv{idx}/Conv'

        old_to_new[old_name] = new_name
        node.name = new_name

        # Also rename output
        if node.output:
            old_output = node.output[0]
            new_output = new_name.replace('/', '_').lstrip('_') + '_output'
            old_to_new[old_output] = new_output
            node.output[0] = new_output

    # Update all references in other nodes
    for node in model.graph.node:
        # Update inputs
        for i, inp in enumerate(node.input):
            if inp in old_to_new:
                node.input[i] = old_to_new[inp]

        # Update outputs
        for i, out in enumerate(node.output):
            if out in old_to_new:
                node.output[i] = old_to_new[out]

    # Update graph outputs
    for output in model.graph.output:
        if output.name in old_to_new:
            output.name = old_to_new[output.name]

    # Print renamed detection head nodes
    print(f"\nDetection head nodes (for --end-node-names):")
    for idx in sorted(mapping.keys()):
        if idx >= num_convs - 4:
            print(f"  {mapping[idx]}")

    return model, True


def get_end_node_names(model, model_type: str = 'seg'):
    """
    Extract the end node names for hailomz compile command.
    """
    conv_nodes = [n for n in model.graph.node if n.op_type == 'Conv']

    if model_type == 'seg':
        # Last 4 Conv nodes for segmentation
        end_nodes = [n.name for n in conv_nodes[-4:]]
    else:
        # Last 3 Conv nodes for detection
        end_nodes = [n.name for n in conv_nodes[-3:]]

    return end_nodes


def main():
    parser = argparse.ArgumentParser(
        description='Rename ONNX nodes for Hailo compatibility'
    )
    parser.add_argument('input', help='Input ONNX file')
    parser.add_argument('output', help='Output ONNX file')
    parser.add_argument(
        '--model-type',
        choices=['seg', 'detect'],
        default='seg',
        help='Model type (default: seg)'
    )
    parser.add_argument(
        '--analyze-only',
        action='store_true',
        help='Only analyze, do not modify'
    )

    args = parser.parse_args()

    # Load model
    print(f"Loading ONNX: {args.input}")
    model = onnx.load(args.input)

    print(f"Opset version: {model.opset_import[0].version}")

    if args.analyze_only:
        # Just show current structure
        conv_nodes = [n for n in model.graph.node if n.op_type == 'Conv']
        print(f"\nConv nodes ({len(conv_nodes)} total):")
        print("\nFirst 5:")
        for n in conv_nodes[:5]:
            print(f"  {n.name}")
        print("\nLast 5:")
        for n in conv_nodes[-5:]:
            print(f"  {n.name}")

        print(f"\nEnd node names for hailomz:")
        end_nodes = get_end_node_names(model, args.model_type)
        for n in end_nodes:
            print(f"  {n}")
        return

    # Rename nodes
    model, modified = rename_nodes_to_hailo_style(model, args.model_type)

    if not modified:
        print(f"\nNo modification needed. Model already Hailo compatible.")
        return

    # Save modified model
    print(f"\nSaving to: {args.output}")
    onnx.save(model, args.output)

    # Print compilation command
    end_nodes = get_end_node_names(model, args.model_type)

    print(f"\n" + "="*60)
    print("Compilation command:")
    print("="*60)

    model_name = 'yolov5s_seg' if args.model_type == 'seg' else 'yolov5s'

    print(f"""
hailomz compile {model_name} \\
    --ckpt {args.output} \\
    --hw-arch hailo8 \\
    --calib-path /path/to/calibration/images \\
    --classes <num_classes> \\
    --end-node-names {' '.join(end_nodes)}
""")

    print("Or use Python SDK directly:")
    print(f"""
from hailo_sdk_client import ClientRunner

runner = ClientRunner(hw_arch='hailo8')
hn, npz = runner.translate_onnx_model(
    '{args.output}',
    net_name='{Path(args.output).stem}',
    end_node_names={end_nodes}
)
""")


if __name__ == '__main__':
    main()
