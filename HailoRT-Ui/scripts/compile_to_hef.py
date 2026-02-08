#!/usr/bin/env python3
"""
Compile ONNX model to Hailo HEF format.

Usage:
    python compile_to_hef.py model.onnx -o model.hef --calib ./calibration_images
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def main():
    parser = argparse.ArgumentParser(
        description='Compile ONNX model to Hailo HEF format'
    )
    parser.add_argument('input', help='Input ONNX model (.onnx file)')
    parser.add_argument('-o', '--output', help='Output HEF file path')
    parser.add_argument(
        '--calib', required=True,
        help='Calibration images directory'
    )
    parser.add_argument(
        '--target', choices=['hailo8', 'hailo8l', 'hailo15h'],
        default='hailo8',
        help='Target Hailo device (default: hailo8)'
    )
    parser.add_argument(
        '--opt-level', type=int, choices=[0, 1, 2, 3], default=2,
        help='Optimization level (default: 2)'
    )

    args = parser.parse_args()

    # Set default output path
    if not args.output:
        args.output = args.input.replace('.onnx', '.hef')

    # Validate calibration directory
    if not os.path.isdir(args.calib):
        print(f"Error: Calibration directory not found: {args.calib}")
        return 1

    print(f"Input:       {args.input}")
    print(f"Output:      {args.output}")
    print(f"Calibration: {args.calib}")
    print(f"Target:      {args.target}")
    print(f"Opt Level:   {args.opt_level}")
    print("-" * 50)

    try:
        from services.converter_service import ConverterService

        converter = ConverterService()
        converter.set_callbacks(
            progress_cb=lambda p: print(f"Progress: {p}%"),
            log_cb=lambda m: print(m)
        )

        success = converter.compile_onnx_to_hef(
            onnx_path=args.input,
            hef_path=args.output,
            calib_dir=args.calib,
            target=args.target,
            optimization_level=args.opt_level
        )

        if success:
            print("-" * 50)
            print(f"✓ Successfully compiled to: {args.output}")
            return 0
        else:
            print("✗ Compilation failed")
            return 1

    except ImportError as e:
        print(f"✗ Error: Hailo SDK not installed")
        print("Please install hailo_dataflow_compiler package")
        return 1
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
