#!/usr/bin/env python3
"""
YOLOv5 Export Wrapper for PyTorch 2.6+ compatibility.

This script patches torch.load to work with PyTorch 2.6+ which defaults
weights_only=True for security. We need weights_only=False to load YOLO models.

Usage:
    python yolov5_export_wrapper.py --weights model.pt --img-size 640 640 \
        --batch-size 1 --device cpu --include onnx --opset 11 --simplify
"""

import sys
import os

# Patch torch.load BEFORE importing anything else
def patch_torch_load():
    """Patch torch.load to use weights_only=False by default."""
    import torch

    # Check if already patched
    if hasattr(torch, '_yolo_wrapper_patched'):
        return
    torch._yolo_wrapper_patched = True

    _original_load = torch.load

    def patched_load(*args, **kwargs):
        # Force weights_only=False for YOLO model loading
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return _original_load(*args, **kwargs)

    torch.load = patched_load
    print("Patched torch.load for PyTorch 2.6+ compatibility")

    # Add yolov5 package path to allow imports
    try:
        import yolov5
        yolov5_path = os.path.dirname(yolov5.__file__)
        if yolov5_path not in sys.path:
            sys.path.insert(0, yolov5_path)
    except ImportError:
        pass

    # Add safe globals for YOLO models (multiple import paths)
    if hasattr(torch, 'serialization') and hasattr(torch.serialization, 'add_safe_globals'):
        safe_classes = []

        # Try yolov5 package path
        try:
            from yolov5.models.yolo import DetectionModel, SegmentationModel, ClassificationModel
            safe_classes.extend([DetectionModel, SegmentationModel, ClassificationModel])
            print(f"Added yolov5.models.yolo classes to safe globals")
        except ImportError:
            pass

        # Try relative path (used internally by yolov5)
        try:
            from models.yolo import DetectionModel as DM, SegmentationModel as SM
            safe_classes.extend([DM, SM])
            print(f"Added models.yolo classes to safe globals")
        except ImportError:
            pass

        # Try models.common
        try:
            from models.common import DetectMultiBackend
            safe_classes.append(DetectMultiBackend)
        except ImportError:
            pass

        if safe_classes:
            torch.serialization.add_safe_globals(safe_classes)


# Apply patch immediately
patch_torch_load()


def main():
    """Run YOLOv5 export with patched torch.load."""
    # Parse arguments - convert from our format to yolov5 format
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640])
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--include', nargs='+', default=['onnx'])
    parser.add_argument('--opset', type=int, default=11)
    parser.add_argument('--simplify', action='store_true')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--dynamic', action='store_true')

    args = parser.parse_args()

    # Try to use yolov5 package export
    try:
        from yolov5.export import run as yolov5_export

        yolov5_export(
            weights=args.weights,
            imgsz=args.img_size if len(args.img_size) == 2 else [args.img_size[0], args.img_size[0]],
            batch_size=args.batch_size,
            device=args.device,
            include=args.include,
            opset=args.opset,
            simplify=args.simplify,
            half=args.half,
            dynamic=args.dynamic
        )
        print(f"Export successful: {args.weights}")

    except Exception as e:
        print(f"Export failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
