"""
Hailo Service
Wrapper for HailoRT API with YOLOv8 raw output decoding.
"""

import re
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from utils.logger import setup_logger

logger = setup_logger(__name__)

# COCO 80 class names (default fallback)
COCO_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush',
]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def _nms(boxes, scores, iou_threshold=0.45):
    """Simple NMS implementation."""
    if len(boxes) == 0:
        return np.array([], dtype=int)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return np.array(keep)


class HailoService:
    """Service for interacting with Hailo device."""

    def __init__(self):
        self.device = None
        self.hef = None
        self.network_group = None
        self.input_vstreams = None
        self.output_vstreams = None
        self.class_names = None
        self._output_map = None
        self._quant_map = None
        self._check_hailo_available()

    def _check_hailo_available(self):
        """Check if HailoRT is available."""
        try:
            from hailo_platform import HailoRTException
            self.hailo_available = True
        except ImportError:
            self.hailo_available = False

    def connect(self) -> Optional[Any]:
        """Connect to Hailo device."""
        if not self.hailo_available:
            raise ImportError("HailoRT not installed")

        from hailo_platform import VDevice, HailoRTException

        try:
            self.device = VDevice()
            return self.device
        except HailoRTException as e:
            logger.error(f"Failed to connect to Hailo device: {e}")
            return None

    def disconnect(self):
        """Disconnect from Hailo device."""
        if self.network_group:
            self.network_group = None
        if self.device:
            self.device = None
        if hasattr(self, '_board_cache'):
            del self._board_cache
        if hasattr(self, '_power_unsupported'):
            del self._power_unsupported

    def get_device_info(self) -> Dict[str, Any]:
        """Get device information via Device.control API (HailoRT 4.23)."""
        if not self.device:
            return {}

        try:
            physical_device = self.device.get_physical_devices()[0]
            ctrl = physical_device.control

            if not hasattr(self, '_board_cache'):
                board_info = ctrl.identify()
                def _clean(val, default='N/A'):
                    s = str(val or default).replace('\x00', '').strip()
                    return s if s else default

                self._board_cache = {
                    'device_name': _clean(board_info.board_name, 'Hailo-8'),
                    'architecture': _clean(board_info.device_architecture, 'HAILO8'),
                    'serial': _clean(board_info.serial_number),
                    'firmware': _clean(board_info.firmware_version),
                }
                logger.info(
                    f"Device identified: {self._board_cache['device_name']} "
                    f"FW={self._board_cache['firmware']} "
                    f"Arch={self._board_cache['architecture']}"
                )

            temp_info = ctrl.get_chip_temperature()
            temperature = temp_info.ts0_temperature
            logger.debug(f"Temperature: ts0={temp_info.ts0_temperature:.1f}°C, ts1={temp_info.ts1_temperature:.1f}°C")

            power = 0.0
            if not hasattr(self, '_power_unsupported'):
                try:
                    power_info = ctrl.get_power_measurement()
                    power = power_info.average_value
                except Exception:
                    self._power_unsupported = True
                    logger.info("Power measurement not supported on this board")

            info = {
                **self._board_cache,
                'driver': self._get_driver_version(),
                'temperature': temperature,
                'power': power,
                'utilization': self._get_utilization()
            }
            return info

        except (IndexError, AttributeError) as e:
            logger.error(f"Device access error: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error getting device info ({type(e).__name__}): {e}")
            return {}

    def _get_driver_version(self) -> str:
        """Get HailoRT driver version."""
        try:
            import hailo_platform
            return hailo_platform.__version__
        except (AttributeError, ImportError):
            return "Unknown"

    def _get_utilization(self) -> float:
        """Get device utilization."""
        return 0.0

    def set_class_names(self, names: List[str]):
        """Set custom class names (e.g., from data.yaml)."""
        self.class_names = names
        logger.info(f"Class names set: {names}")

    def load_model(self, hef_path: str) -> bool:
        """Load HEF model and analyze output structure."""
        if not self.device:
            raise RuntimeError("Device not connected")

        from hailo_platform import HEF

        try:
            self.hef = HEF(hef_path)
            self.network_group = self.device.configure(self.hef)[0]
            self.input_vstream_info = self.network_group.get_input_vstream_infos()
            self.output_vstream_info = self.network_group.get_output_vstream_infos()

            # Build quantization map
            self._quant_map = {}
            for v in self.output_vstream_info:
                self._quant_map[v.name] = (v.quant_info.qp_zp, v.quant_info.qp_scale)

            # Analyze output structure for YOLOv8 decoding
            self._output_map = self._analyze_outputs()
            self._output_logged = False

            input_shape = self.input_vstream_info[0].shape
            logger.info(f"Model loaded: {hef_path}")
            logger.info(f"  Input: {self.input_vstream_info[0].name} shape={input_shape}")
            logger.info(f"  Outputs: {len(self.output_vstream_info)} layers")
            if self._output_map:
                model_type = "YOLOv8-seg" if self._output_map.get('is_seg') else "YOLOv8-detect"
                logger.info(f"  Detected {model_type}: {len(self._output_map['scales'])} scales, "
                           f"nc={self._output_map['num_classes']}")

            return True

        except FileNotFoundError as e:
            logger.error(f"HEF file not found: {e}")
            return False
        except (IndexError, AttributeError) as e:
            logger.error(f"Failed to configure network group: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to load model ({type(e).__name__}): {e}")
            return False

    def _analyze_outputs(self) -> Optional[Dict]:
        """Analyze output layers to detect YOLOv8 DFL structure (detect + segment).

        Groups outputs by spatial resolution, then identifies:
        - bbox DFL output: 64 channels (4 * reg_max=16), highest conv number per scale
        - class output: small channel count (nc), per scale
        - (seg) mask coefficient output: 32 channels per scale
        - (seg) proto output: 32 channels at higher resolution (input/4)
        """
        # Group outputs by spatial resolution
        groups = {}
        for v in self.output_vstream_info:
            h, w = v.shape[0], v.shape[1]
            key = (h, w)
            if key not in groups:
                groups[key] = []
            groups[key].append(v)

        # Need at least 2 different spatial resolutions for multi-scale detection
        if len(groups) < 2:
            logger.debug("Not a multi-scale model, skipping YOLOv8 analysis")
            return None

        input_shape = self.input_vstream_info[0].shape
        input_h = input_shape[0]

        # Extract conv number for sorting
        def conv_num(v):
            m = re.search(r'conv(\d+)', v.name)
            return int(m.group(1)) if m else 0

        # First pass: identify detection scale groups (those with 64-channel bbox DFL)
        detection_keys = set()
        for (h, w), vstreams in groups.items():
            if any(v.shape[2] == 64 for v in vstreams):
                detection_keys.add((h, w))

        # Check for proto output: single 32-channel output at a non-detection resolution
        proto_output = None
        for (h, w), vstreams in groups.items():
            if (h, w) in detection_keys:
                continue
            proto_candidates = [v for v in vstreams if v.shape[2] == 32]
            if len(proto_candidates) == 1:
                proto_output = proto_candidates[0]
                break

        is_seg = proto_output is not None

        scales = []
        num_classes = None

        for (h, w), vstreams in sorted(groups.items(), key=lambda x: x[0]):
            if (h, w) not in detection_keys:
                continue

            stride = input_h // h

            # Find bbox DFL: 64-channel output with highest conv number
            bbox_candidates = [v for v in vstreams if v.shape[2] == 64]
            # Find class output; exclude 32-ch mask coefficients when seg model detected
            exclude_ch = {64, 128}
            if is_seg:
                exclude_ch.add(32)
            cls_candidates = [v for v in vstreams
                              if v.shape[2] not in exclude_ch and v.shape[2] < 64]

            if not bbox_candidates or not cls_candidates:
                continue

            bbox_out = max(bbox_candidates, key=conv_num)
            cls_out = max(cls_candidates, key=conv_num)
            nc = cls_out.shape[2]

            if num_classes is None:
                num_classes = nc
            elif nc != num_classes:
                logger.warning(f"Inconsistent class count: {nc} vs {num_classes}")

            scale_info = {
                'stride': stride,
                'bbox_name': bbox_out.name,
                'cls_name': cls_out.name,
                'h': h, 'w': w,
            }

            # For seg models, find mask coefficient output (32 channels)
            if is_seg:
                mask_candidates = [v for v in vstreams
                                   if v.shape[2] == 32
                                   and v.name != bbox_out.name
                                   and v.name != cls_out.name]
                if mask_candidates:
                    scale_info['mask_name'] = max(mask_candidates, key=conv_num).name

            scales.append(scale_info)

        if not scales:
            logger.debug("Could not detect YOLOv8 output structure")
            return None

        # Sort by stride descending (32, 16, 8) for standard decoding order
        scales.sort(key=lambda s: s['stride'], reverse=True)

        result = {
            'scales': scales,
            'num_classes': num_classes,
            'reg_max': 15,  # 4 * (15+1) = 64 channels
            'is_seg': False,
        }

        if is_seg and all('mask_name' in s for s in scales):
            result['is_seg'] = True
            result['proto_name'] = proto_output.name
            result['proto_h'] = proto_output.shape[0]
            result['proto_w'] = proto_output.shape[1]
            logger.info(f"Detected YOLOv8-seg model: proto={proto_output.name} "
                        f"({proto_output.shape[0]}x{proto_output.shape[1]})")

        return result

    def unload_model(self):
        """Unload current model."""
        self.hef = None
        self.network_group = None
        self._output_map = None
        self._quant_map = None

    def infer(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run inference on frame."""
        if not self.network_group:
            raise RuntimeError("No model loaded")

        from hailo_platform import InferVStreams, InputVStreamParams, OutputVStreamParams

        try:
            input_data = self._preprocess(frame)

            # Both input and output as quantized uint8 (manual dequantization)
            input_params = InputVStreamParams.make_from_network_group(
                self.network_group, quantized=True
            )
            output_params = OutputVStreamParams.make_from_network_group(
                self.network_group, quantized=True
            )

            with self.network_group.activate():
                with InferVStreams(self.network_group, input_params, output_params) as pipeline:
                    input_dict = {self.input_vstream_info[0].name: input_data}
                    output = pipeline.infer(input_dict)

            # Log output structure once
            if not self._output_logged:
                self._output_logged = True
                for name, data in output.items():
                    arr = np.array(data)
                    logger.info(f"Output '{name}': shape={arr.shape}, dtype={arr.dtype}")

            detections = self._postprocess(output, frame.shape)
            return detections

        except (IndexError, KeyError) as e:
            logger.error(f"Input/output stream error: {e}")
            return []
        except ValueError as e:
            logger.error(f"Invalid input data: {e}")
            return []
        except Exception as e:
            logger.error(f"Inference error ({type(e).__name__}): {e}")
            return []

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for inference.

        HailoRT expects uint8 input with shape (batch, H, W, C).
        """
        import cv2

        input_shape = self.input_vstream_info[0].shape
        target_h, target_w = input_shape[0], input_shape[1]
        resized = cv2.resize(frame, (target_w, target_h))
        return np.expand_dims(resized, axis=0)

    def _dequantize(self, data: np.ndarray, name: str) -> np.ndarray:
        """Dequantize uint8 output to float32."""
        zp, scale = self._quant_map[name]
        return (data.astype(np.float32) - zp) * scale

    def _postprocess(self, output: Dict, original_shape: tuple,
                     conf_threshold: float = 0.25, iou_threshold: float = 0.45) -> List[Dict[str, Any]]:
        """Postprocess inference output.

        Supports YOLOv8 raw DFL output (auto-detected) and NMS-processed output.
        """
        # Use YOLOv8 DFL decoder if output structure was detected
        if self._output_map:
            if self._output_map.get('is_seg'):
                return self._postprocess_yolov8_seg(output, original_shape, conf_threshold, iou_threshold)
            return self._postprocess_yolov8_dfl(output, original_shape, conf_threshold, iou_threshold)

        # Fallback: try NMS-style output parsing
        return self._postprocess_nms(output, original_shape, conf_threshold)

    def _postprocess_yolov8_dfl(self, output: Dict, original_shape: tuple,
                                 conf_threshold: float, iou_threshold: float) -> List[Dict[str, Any]]:
        """Decode YOLOv8 raw DFL outputs with manual dequantization."""
        omap = self._output_map
        scales = omap['scales']
        reg_max = omap['reg_max']
        nc = omap['num_classes']
        input_shape = self.input_vstream_info[0].shape
        img_h, img_w = input_shape[0], input_shape[1]
        orig_h, orig_w = original_shape[:2]

        # Decode boxes and scores per scale
        all_boxes = []
        all_scores = []

        for scale in scales:
            stride = scale['stride']

            # Dequantize bbox DFL output
            raw_bbox = self._dequantize(np.array(output[scale['bbox_name']]), scale['bbox_name'])
            bs, h, w, c = raw_bbox.shape

            # Create grid centers
            grid_x = (np.arange(w) + 0.5) * stride
            grid_y = (np.arange(h) + 0.5) * stride
            gx, gy = np.meshgrid(grid_x, grid_y)
            ct_col = gx.flatten()
            ct_row = gy.flatten()
            center = np.stack((ct_col, ct_row, ct_col, ct_row), axis=1)

            # DFL decode: softmax over reg_max+1 bins, weighted sum
            box_dist = raw_bbox.reshape(bs, h * w, 4, reg_max + 1)
            box_dist = _softmax(box_dist)
            reg_range = np.arange(reg_max + 1, dtype=np.float32)
            box_distance = np.sum(box_dist * reg_range, axis=-1) * stride

            # Convert ltrb distances to xyxy coordinates
            box_xyxy = np.concatenate([-box_distance[:, :, :2], box_distance[:, :, 2:]], axis=-1)
            boxes = center[np.newaxis] + box_xyxy
            all_boxes.append(boxes)

            # Dequantize class scores and apply sigmoid
            raw_cls = self._dequantize(np.array(output[scale['cls_name']]), scale['cls_name'])
            cls_flat = raw_cls.reshape(bs, h * w, nc)
            cls_scores = _sigmoid(cls_flat)
            all_scores.append(cls_scores)

        # Concatenate all scales: (bs, total_proposals, 4) and (bs, total_proposals, nc)
        boxes = np.concatenate(all_boxes, axis=1)
        scores = np.concatenate(all_scores, axis=1)

        # Process each batch
        detections = []
        for b in range(boxes.shape[0]):
            for cls_id in range(nc):
                cls_scores = scores[b, :, cls_id]
                mask = cls_scores > conf_threshold
                if not np.any(mask):
                    continue

                cls_boxes = boxes[b][mask]
                cls_conf = cls_scores[mask]

                # NMS per class
                keep = _nms(cls_boxes, cls_conf, iou_threshold)
                for idx in keep:
                    x1, y1, x2, y2 = cls_boxes[idx]
                    # Scale from model input to original image
                    sx = orig_w / img_w
                    sy = orig_h / img_h
                    detections.append({
                        'bbox': [float(x1 * sx), float(y1 * sy),
                                 float(x2 * sx), float(y2 * sy)],
                        'confidence': float(cls_conf[idx]),
                        'class': self._get_class_name(cls_id),
                        'class_id': cls_id,
                    })

        # Sort by confidence
        detections.sort(key=lambda d: d['confidence'], reverse=True)
        return detections

    def _postprocess_yolov8_seg(self, output: Dict, original_shape: tuple,
                                 conf_threshold: float, iou_threshold: float) -> List[Dict[str, Any]]:
        """Decode YOLOv8-seg outputs including instance masks."""
        omap = self._output_map
        scales = omap['scales']
        reg_max = omap['reg_max']
        nc = omap['num_classes']
        input_shape = self.input_vstream_info[0].shape
        img_h, img_w = input_shape[0], input_shape[1]
        orig_h, orig_w = original_shape[:2]

        all_boxes = []
        all_scores = []
        all_mask_coeffs = []

        for scale in scales:
            stride = scale['stride']

            # Dequantize bbox DFL output
            raw_bbox = self._dequantize(np.array(output[scale['bbox_name']]), scale['bbox_name'])
            bs, h, w, c = raw_bbox.shape

            # Create grid centers
            grid_x = (np.arange(w) + 0.5) * stride
            grid_y = (np.arange(h) + 0.5) * stride
            gx, gy = np.meshgrid(grid_x, grid_y)
            ct_col = gx.flatten()
            ct_row = gy.flatten()
            center = np.stack((ct_col, ct_row, ct_col, ct_row), axis=1)

            # DFL decode
            box_dist = raw_bbox.reshape(bs, h * w, 4, reg_max + 1)
            box_dist = _softmax(box_dist)
            reg_range = np.arange(reg_max + 1, dtype=np.float32)
            box_distance = np.sum(box_dist * reg_range, axis=-1) * stride

            box_xyxy = np.concatenate([-box_distance[:, :, :2], box_distance[:, :, 2:]], axis=-1)
            boxes = center[np.newaxis] + box_xyxy
            all_boxes.append(boxes)

            # Dequantize class scores
            raw_cls = self._dequantize(np.array(output[scale['cls_name']]), scale['cls_name'])
            cls_flat = raw_cls.reshape(bs, h * w, nc)
            cls_scores = _sigmoid(cls_flat)
            all_scores.append(cls_scores)

            # Dequantize mask coefficients (32 channels)
            raw_mask = self._dequantize(np.array(output[scale['mask_name']]), scale['mask_name'])
            mask_flat = raw_mask.reshape(bs, h * w, 32)
            all_mask_coeffs.append(mask_flat)

        boxes = np.concatenate(all_boxes, axis=1)
        scores = np.concatenate(all_scores, axis=1)
        mask_coeffs = np.concatenate(all_mask_coeffs, axis=1)

        # Dequantize proto output: (bs, proto_h, proto_w, 32)
        raw_proto = self._dequantize(np.array(output[omap['proto_name']]), omap['proto_name'])
        proto_h_size = omap['proto_h']
        proto_w_size = omap['proto_w']

        detections = []
        for b in range(boxes.shape[0]):
            proto = raw_proto[b]  # (proto_h, proto_w, 32)

            for cls_id in range(nc):
                cls_scores = scores[b, :, cls_id]
                mask = cls_scores > conf_threshold
                if not np.any(mask):
                    continue

                cls_boxes = boxes[b][mask]
                cls_conf = cls_scores[mask]
                cls_coeffs = mask_coeffs[b][mask]  # (n, 32)

                keep = _nms(cls_boxes, cls_conf, iou_threshold)
                for idx in keep:
                    x1, y1, x2, y2 = cls_boxes[idx]
                    sx = orig_w / img_w
                    sy = orig_h / img_h

                    # Compute mask: proto @ coeffs -> (proto_h, proto_w)
                    coeffs = cls_coeffs[idx]  # (32,)
                    mask_pred = _sigmoid(proto @ coeffs)  # (proto_h, proto_w)

                    # Crop mask to bbox region (in model input space)
                    scale_x = proto_w_size / img_w
                    scale_y = proto_h_size / img_h
                    mx1 = max(0, int(x1 * scale_x))
                    my1 = max(0, int(y1 * scale_y))
                    mx2 = min(proto_w_size, int(x2 * scale_x))
                    my2 = min(proto_h_size, int(y2 * scale_y))

                    cropped = np.zeros_like(mask_pred)
                    cropped[my1:my2, mx1:mx2] = mask_pred[my1:my2, mx1:mx2]
                    binary_mask = (cropped > 0.5).astype(np.uint8)

                    detections.append({
                        'bbox': [float(x1 * sx), float(y1 * sy),
                                 float(x2 * sx), float(y2 * sy)],
                        'confidence': float(cls_conf[idx]),
                        'class': self._get_class_name(cls_id),
                        'class_id': cls_id,
                        'mask': binary_mask,  # (proto_h, proto_w) uint8
                    })

        detections.sort(key=lambda d: d['confidence'], reverse=True)
        return detections

    def _postprocess_nms(self, output: Dict, original_shape: tuple,
                          conf_threshold: float) -> List[Dict[str, Any]]:
        """Fallback: parse NMS-processed output (batch, num_dets, 6)."""
        detections = []
        orig_h, orig_w = original_shape[:2]

        for name, data in output.items():
            arr = np.array(data)
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim == 2 and arr.shape[1] >= 5:
                for row in arr:
                    score = float(row[4])
                    if score < conf_threshold:
                        continue
                    if arr.shape[1] >= 6:
                        y1, x1, y2, x2 = row[0], row[1], row[2], row[3]
                        class_id = int(row[5])
                    else:
                        x1, y1, x2, y2 = row[0], row[1], row[2], row[3]
                        class_id = 0
                    if max(x1, y1, x2, y2) <= 1.0:
                        x1 *= orig_w; y1 *= orig_h
                        x2 *= orig_w; y2 *= orig_h
                    detections.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': score,
                        'class': self._get_class_name(class_id),
                        'class_id': class_id,
                    })
        return detections

    def _get_class_name(self, class_id: int) -> str:
        """Get class name from ID."""
        if self.class_names and 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        if 0 <= class_id < len(COCO_NAMES):
            return COCO_NAMES[class_id]
        return f'class_{class_id}'

    def get_model_info(self) -> Dict[str, Any]:
        """Get loaded model information."""
        if not self.hef:
            return {}

        try:
            info = {
                'input_shape': self.input_vstream_info[0].shape if self.input_vstream_info else None,
                'output_shape': self.output_vstream_info[0].shape if self.output_vstream_info else None,
                'input_names': [v.name for v in self.input_vstream_info] if self.input_vstream_info else [],
                'output_names': [v.name for v in self.output_vstream_info] if self.output_vstream_info else [],
            }
            if self._output_map:
                info['num_classes'] = self._output_map['num_classes']
                info['scales'] = len(self._output_map['scales'])
            return info
        except (IndexError, AttributeError) as e:
            logger.error(f"Error accessing model stream info: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error getting model info ({type(e).__name__}): {e}")
            return {}
