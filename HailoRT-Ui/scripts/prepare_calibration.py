#!/usr/bin/env python3
"""
Prepare calibration dataset for Hailo model compilation.

Usage:
    python prepare_calibration.py ./images -o calibration_set.npy --size 640 640
"""

import argparse
import sys
import os
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(
        description='Prepare calibration dataset for Hailo compilation'
    )
    parser.add_argument('input_dir', help='Input directory with images')
    parser.add_argument('-o', '--output', help='Output .npy file path')
    parser.add_argument(
        '--size', nargs=2, type=int, default=[640, 640],
        metavar=('HEIGHT', 'WIDTH'),
        help='Target size (default: 640 640)'
    )
    parser.add_argument(
        '--max-images', type=int, default=500,
        help='Maximum number of images (default: 500)'
    )
    parser.add_argument(
        '--normalize', action='store_true', default=True,
        help='Normalize to 0-1 range (default: True)'
    )

    args = parser.parse_args()

    # Set default output path
    if not args.output:
        args.output = os.path.join(args.input_dir, 'calibration_set.npy')

    # Validate input directory
    if not os.path.isdir(args.input_dir):
        print(f"Error: Directory not found: {args.input_dir}")
        return 1

    print(f"Input:      {args.input_dir}")
    print(f"Output:     {args.output}")
    print(f"Size:       {args.size[0]}x{args.size[1]}")
    print(f"Max images: {args.max_images}")
    print("-" * 50)

    # Find all images
    input_path = Path(args.input_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(f'*{ext}'))
        image_files.extend(input_path.glob(f'*{ext.upper()}'))

    image_files = sorted(set(image_files))[:args.max_images]

    if not image_files:
        print(f"Error: No images found in {args.input_dir}")
        return 1

    print(f"Found {len(image_files)} images")

    # Process images
    images = []
    for i, img_path in enumerate(image_files):
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((args.size[1], args.size[0]))  # PIL uses (width, height)
            img_array = np.array(img).astype(np.float32)

            if args.normalize:
                img_array = img_array / 255.0

            images.append(img_array)

            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(image_files)} images")

        except Exception as e:
            print(f"Warning: Failed to process {img_path}: {e}")
            continue

    if not images:
        print("Error: No images could be processed")
        return 1

    # Stack into array
    calibration_data = np.stack(images, axis=0)

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.save(args.output, calibration_data)

    print("-" * 50)
    print(f"✓ Saved calibration data:")
    print(f"  Path:  {args.output}")
    print(f"  Shape: {calibration_data.shape}")
    print(f"  Dtype: {calibration_data.dtype}")
    print(f"  Size:  {calibration_data.nbytes / (1024*1024):.1f} MB")

    return 0


if __name__ == '__main__':
    sys.exit(main())
