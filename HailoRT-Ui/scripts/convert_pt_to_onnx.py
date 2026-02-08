#!/usr/bin/env python3
"""
Convert PyTorch model to ONNX format.

Usage:
    python convert_pt_to_onnx.py model.pt -o model.onnx --size 640 640
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def main():
    parser = argparse.ArgumentParser(
        description='Convert PyTorch model to ONNX format'
    )
    parser.add_argument('input', help='Input PyTorch model (.pt file)')
    parser.add_argument('-o', '--output', help='Output ONNX file path')
    parser.add_argument(
        '--size', nargs=2, type=int, default=[640, 640],
        metavar=('HEIGHT', 'WIDTH'),
        help='Input size (default: 640 640)'
    )
    parser.add_argument(
        '--batch', type=int, default=1,
        help='Batch size (default: 1)'
    )
    parser.add_argument(
        '--opset', type=int, default=11,
        help='ONNX opset version (default: 11)'
    )

    args = parser.parse_args()

    # Set default output path
    if not args.output:
        args.output = args.input.replace('.pt', '.onnx').replace('.pth', '.onnx')

    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Size:   {args.size[0]}x{args.size[1]}")
    print(f"Batch:  {args.batch}")
    print(f"Opset:  {args.opset}")
    print("-" * 50)

    try:
        from services.converter_service import ConverterService

        converter = ConverterService()
        converter.set_callbacks(
            progress_cb=lambda p: print(f"Progress: {p}%"),
            log_cb=lambda m: print(m)
        )

        success = converter.convert_pt_to_onnx(
            pt_path=args.input,
            onnx_path=args.output,
            input_size=tuple(args.size),
            batch_size=args.batch,
            opset_version=args.opset
        )

        if success:
            print("-" * 50)
            print(f"✓ Successfully converted to: {args.output}")
            return 0
        else:
            print("✗ Conversion failed")
            return 1

    except Exception as e:
        print(f"✗ Error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
