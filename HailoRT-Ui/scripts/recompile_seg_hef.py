"""
Recompile YOLOv8-seg ONNX to HEF with proper segmentation output nodes.

Hailo DFC compilation with explicit end nodes for:
  - cv2.X: bbox DFL (64ch per scale)
  - cv3.X: class scores (nc per scale)
  - cv4.X: mask coefficients (32ch per scale)
  - proto:  prototype masks (32ch @ 160x160)
"""

import os
import sys
import glob
import numpy as np

def main():
    from hailo_sdk_client import ClientRunner

    # --- Configuration ---
    onnx_path = "/home/amap/yolov8_custom/Project_yolov8/forklift/models/onnx/best.onnx"
    output_dir = "/home/amap/yolov8_custom/Project_yolov8/forklift/models/hef"
    har_path = os.path.join(output_dir, "best_seg.har")
    hef_path = os.path.join(output_dir, "best_seg.hef")
    calib_dir = "/home/amap/yolov8_custom/Project_yolov8/forklift/train/images"
    net_name = "yolov8s_seg"

    # YOLOv8-seg end nodes (split at detection head outputs + proto)
    end_nodes = [
        "/model.22/cv2.0/cv2.0.2/Conv",
        "/model.22/cv3.0/cv3.0.2/Conv",
        "/model.22/cv4.0/cv4.0.2/Conv",
        "/model.22/cv2.1/cv2.1.2/Conv",
        "/model.22/cv3.1/cv3.1.2/Conv",
        "/model.22/cv4.1/cv4.1.2/Conv",
        "/model.22/cv2.2/cv2.2.2/Conv",
        "/model.22/cv3.2/cv3.2.2/Conv",
        "/model.22/cv4.2/cv4.2.2/Conv",
        "/model.22/proto/cv3/act/Mul",
    ]

    os.makedirs(output_dir, exist_ok=True)

    # --- Step 1: Parse ONNX ---
    print("=" * 60)
    print("[1/4] Parsing ONNX model with seg end nodes...")
    print(f"  ONNX: {onnx_path}")
    print(f"  End nodes: {len(end_nodes)}")

    runner = ClientRunner(hw_arch="hailo8")
    hn, npz = runner.translate_onnx_model(
        onnx_path,
        net_name=net_name,
        end_node_names=end_nodes,
        start_node_names=None,
    )

    # --- Step 2: Apply alls script ---
    print("\n" + "=" * 60)
    print("[2/4] Applying model optimization script...")

    # Normalization only; sigmoid is applied in post-processing code
    alls_script = 'normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])'
    print(f"  ALLS: {alls_script}")
    runner.load_model_script(alls_script)

    # --- Step 3: Optimize (Quantize) ---
    print("\n" + "=" * 60)
    print("[3/4] Quantizing model (this may take several minutes)...")

    # Prepare calibration dataset
    import cv2

    calib_images = sorted(glob.glob(os.path.join(calib_dir, "*.jpg")))
    calib_images += sorted(glob.glob(os.path.join(calib_dir, "*.png")))
    # Use up to 64 images for calibration
    calib_images = calib_images[:64]
    print(f"  Calibration images: {len(calib_images)}")

    input_shape = (640, 640)
    calib_data = []
    for img_path in calib_images:
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, input_shape)
        img = img.astype(np.float32)  # DFC expects float after normalization is applied
        calib_data.append(img)

    calib_dataset = np.array(calib_data)
    print(f"  Calibration dataset shape: {calib_dataset.shape}")

    runner.optimize(calib_dataset)
    print("  Quantization complete!")

    # Save HAR
    runner.save_har(har_path)
    print(f"  HAR saved: {har_path}")

    # --- Step 4: Compile to HEF ---
    print("\n" + "=" * 60)
    print("[4/4] Compiling to HEF...")

    hef = runner.compile()

    with open(hef_path, "wb") as f:
        f.write(hef)
    print(f"  HEF saved: {hef_path}")

    # --- Verify ---
    print("\n" + "=" * 60)
    print("Verifying compiled HEF...")
    from hailo_platform import HEF as HEFReader
    from collections import defaultdict

    hef_obj = HEFReader(hef_path)
    groups = defaultdict(list)
    for v in hef_obj.get_output_vstream_infos():
        h, w, c = v.shape[0], v.shape[1], v.shape[2]
        groups[(h, w)].append((v.name, c))

    for (h, w), layers in sorted(groups.items()):
        print(f"  Resolution {h}x{w}:")
        for name, c in layers:
            print(f"    {name}  channels={c}")

    has_proto = any(
        any(c == 32 for _, c in layers)
        for (h, w), layers in groups.items()
        if not any(c == 64 for _, c in layers)
    )
    print(f"\n  Proto output detected: {has_proto}")
    print(f"  Total outputs: {sum(len(v) for v in groups.values())}")
    print("\nDone!")


if __name__ == "__main__":
    main()
