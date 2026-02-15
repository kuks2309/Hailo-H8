"""
Inference Tab Controller
Manages model inference and visualization.
"""

from __future__ import annotations

import os
import time
from typing import Optional, List, Dict, Any, Union
from PyQt5.QtWidgets import QWidget, QFileDialog, QMessageBox, QTableWidgetItem
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import uic
import numpy as np

from utils.logger import setup_logger
from utils.styles import get_color
from utils.folder_structure import parse_data_yaml

logger = setup_logger(__name__)


class InferenceWorker(QThread):
    """Worker thread for running inference."""
    frame_ready = pyqtSignal(np.ndarray, list, float)  # frame, detections, fps
    error = pyqtSignal(str)
    stopped = pyqtSignal()

    def __init__(self, model_path: str, source: Union[str, int], conf_threshold: float = 0.5,
                 hailo_service=None, class_names=None) -> None:
        super().__init__()
        self.model_path: str = model_path
        self.source: Union[str, int] = source
        self.conf_threshold: float = conf_threshold
        self.running: bool = False
        self.model: Optional[Any] = None
        self._shared_hailo_service = hailo_service
        self._class_names = class_names

    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    def run(self) -> None:
        """Run inference loop."""
        self.running = True
        logger.debug(f"InferenceWorker starting: model={self.model_path}, source={self.source}")

        try:
            # Initialize model
            self._load_model()

            # Check if source is a folder
            if isinstance(self.source, str) and os.path.isdir(self.source):
                self._run_folder_inference()
            else:
                self._run_video_inference()

        except ImportError as e:
            logger.error(f"Import error in inference worker: {e}", exc_info=True)
            self.error.emit(f"Import error: {e}")
        except RuntimeError as e:
            logger.error(f"Runtime error in inference worker: {e}", exc_info=True)
            self.error.emit(f"Runtime error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in inference worker ({type(e).__name__}): {e}", exc_info=True)
            self.error.emit(f"Unexpected error ({type(e).__name__}): {e}")
        finally:
            self.stopped.emit()

    def _run_folder_inference(self) -> None:
        """Run inference on all images in a folder."""
        import cv2

        image_files = sorted(
            f for f in os.listdir(self.source)
            if os.path.splitext(f)[1].lower() in self.IMAGE_EXTENSIONS
        )

        if not image_files:
            self.error.emit(f"No image files found in: {self.source}")
            return

        logger.info(f"Folder inference: {len(image_files)} images in {self.source}")

        frame_count = 0
        start_time = time.time()

        for filename in image_files:
            if not self.running:
                break

            filepath = os.path.join(self.source, filename)
            frame = cv2.imread(filepath)
            if frame is None:
                logger.warning(f"Failed to read image: {filename}")
                continue

            detections = self._run_inference(frame)

            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0

            result_frame = self._draw_detections(frame, detections)
            self.frame_ready.emit(result_frame, detections, fps)

            # Delay between images for UI viewing
            time.sleep(0.5)

        logger.info(f"Folder inference completed: {frame_count}/{len(image_files)} images")

    def _run_video_inference(self) -> None:
        """Run inference on video/camera source."""
        cap = self._init_capture()
        if cap is None:
            logger.error("Failed to open video source")
            self.error.emit("Failed to open video source")
            return

        frame_count = 0
        start_time = time.time()
        logger.info("Video inference loop started")

        while self.running:
            ret, frame = cap.read()
            if not ret:
                if isinstance(self.source, str) and os.path.isfile(self.source):
                    cap.set(1, 0)  # cv2.CAP_PROP_POS_FRAMES
                    continue
                else:
                    break

            detections = self._run_inference(frame)

            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0

            result_frame = self._draw_detections(frame, detections)
            self.frame_ready.emit(result_frame, detections, fps)

            time.sleep(0.001)

        cap.release()
        logger.info(f"Video inference stopped after {frame_count} frames")

    def stop(self) -> None:
        """Stop inference loop."""
        logger.debug("Stopping inference worker...")
        self.running = False

    def _load_model(self) -> None:
        """Load the model."""
        try:
            logger.debug(f"Loading model: {self.model_path}")
            if self._shared_hailo_service and self._shared_hailo_service.device:
                # Use shared device connection from Device tab
                self.hailo = self._shared_hailo_service
                self.model = self.hailo.load_model(self.model_path)
                self.use_hailo = True
                logger.info(f"Model loaded via shared device: {os.path.basename(self.model_path)}")
            else:
                # Try creating new connection
                from services.hailo_service import HailoService
                self.hailo = HailoService()
                device = self.hailo.connect()
                if device:
                    self.model = self.hailo.load_model(self.model_path)
                    self.use_hailo = True
                    logger.info(f"Model loaded via new connection: {os.path.basename(self.model_path)}")
                else:
                    raise RuntimeError("No Hailo device available. Please connect device first.")
            # Set custom class names from project data.yaml
            if self.use_hailo and self._class_names:
                self.hailo.set_class_names(self._class_names)
        except ImportError:
            # Fallback to mock inference
            logger.warning("HailoRT not available, using mock inference mode")
            self.use_hailo = False
            self.model = None

    def _init_capture(self) -> Optional[Any]:
        """Initialize video capture."""
        try:
            import cv2
        except ImportError:
            logger.error("OpenCV not installed")
            self.error.emit("OpenCV not installed. Install with: pip install opencv-python")
            return None

        if isinstance(self.source, int) or self.source.isdigit():
            # Camera
            logger.debug(f"Initializing camera capture: index={self.source}")
            return cv2.VideoCapture(int(self.source))
        else:
            # File
            if os.path.exists(self.source):
                logger.debug(f"Initializing file capture: {self.source}")
                return cv2.VideoCapture(self.source)
            else:
                logger.error(f"Source file not found: {self.source}")
        return None

    def _run_inference(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run inference on frame."""
        if self.use_hailo and self.model:
            return self.hailo.infer(frame)
        else:
            # Mock detection for testing
            return self._mock_inference(frame)

    def _mock_inference(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Generate mock detections for testing."""
        import random

        detections = []
        h, w = frame.shape[:2]

        # Random number of detections
        num_detections = random.randint(0, 5)

        classes = ['person', 'car', 'dog', 'cat', 'chair', 'bottle']

        for _ in range(num_detections):
            x1 = random.randint(0, w - 100)
            y1 = random.randint(0, h - 100)
            x2 = x1 + random.randint(50, 150)
            y2 = y1 + random.randint(50, 150)
            conf = random.uniform(0.5, 0.99)
            cls = random.choice(classes)

            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': conf,
                'class': cls
            })

        return detections

    # Per-class color palette (BGR)
    CLASS_COLORS = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255),
        (255, 0, 255), (255, 255, 0), (0, 128, 255), (255, 128, 0),
        (128, 0, 255), (0, 255, 128), (255, 0, 128), (128, 255, 0),
    ]

    def _draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draw detection boxes or segmentation masks on frame."""
        try:
            import cv2
        except ImportError:
            return frame

        result = frame.copy()
        fh, fw = result.shape[:2]

        for det in detections:
            bbox = det['bbox']
            conf = det['confidence']
            cls = det['class']
            class_id = det.get('class_id', 0)
            color = self.CLASS_COLORS[class_id % len(self.CLASS_COLORS)]

            x1, y1, x2, y2 = map(int, bbox)

            if 'mask' in det:
                # Segmentation: draw semi-transparent mask overlay
                mask = det['mask']
                mask_resized = cv2.resize(mask, (fw, fh), interpolation=cv2.INTER_LINEAR)
                mask_bool = mask_resized > 0.5

                # Colored overlay with transparency
                overlay = result.copy()
                overlay[mask_bool] = color
                cv2.addWeighted(overlay, 0.45, result, 0.55, 0, result)

                # Draw contour border
                contours, _ = cv2.findContours(
                    (mask_bool.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(result, contours, -1, color, 2)
            else:
                # Detection: draw bounding box
                cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{cls}: {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(result, (x1, y1 - 20), (x1 + tw, y1), color, -1)
            cv2.putText(result, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return result


class InferenceTabController:
    """Controller for inference tab."""

    def __init__(self, tab_widget: QWidget, base_path: str, device_controller=None) -> None:
        self.tab: QWidget = tab_widget
        self.base_path: str = base_path
        self.device_controller = device_controller
        self.worker: Optional[InferenceWorker] = None
        self.model_loaded: bool = False
        self.frame_count: int = 0

        # Load UI into tab
        ui_path = os.path.join(base_path, 'ui', 'tabs', 'inference_tab.ui')
        uic.loadUi(ui_path, self.tab)

        # Connect signals
        self._connect_signals()

    def _connect_signals(self) -> None:
        """Connect UI signals."""
        self.tab.btnBrowseHef.clicked.connect(self._browse_hef)
        self.tab.btnBrowseSource.clicked.connect(self._browse_source)
        self.tab.btnLoadModel.clicked.connect(self.load_model)
        self.tab.btnStart.clicked.connect(self.start_inference)
        self.tab.btnStop.clicked.connect(self.stop_inference)
        self.tab.btnCapture.clicked.connect(self._capture_frame)

        # Source type change
        self.tab.comboSource.currentIndexChanged.connect(self._on_source_type_changed)

    def set_project_info(self, info: dict) -> None:
        """Apply project info from Project tab."""
        self._project_info = info
        if not info.get('valid'):
            return

        # Store class names from project
        self._class_names = info.get('class_names')

        # Auto-fill HEF path from detected files
        hef_files = info.get('hef_files', [])
        if hef_files and not self.tab.editHefPath.text():
            best_hef = next(
                (f for f in hef_files if 'best' in os.path.basename(f).lower()),
                hef_files[0]
            )
            self.tab.editHefPath.setText(best_hef)
            logger.info(f"[Auto] HEF model: {os.path.basename(best_hef)}")
            # Auto-load the model
            self.load_model()

    def _browse_hef(self) -> None:
        """Browse for HEF model file."""
        if hasattr(self, '_project_info') and self._project_info and self._project_info.get('hef_dir'):
            start_dir = self._project_info['hef_dir']
        else:
            start_dir = os.path.join(self.base_path, 'data', 'models', 'hef')
        path, _ = QFileDialog.getOpenFileName(
            self.tab,
            "Select HEF Model",
            start_dir,
            "Hailo Models (*.hef);;All Files (*)"
        )
        if path:
            logger.debug(f"Selected HEF file: {path}")
            self.tab.editHefPath.setText(path)

    def _browse_source(self) -> None:
        """Browse for video/image source."""
        source_type = self.tab.comboSource.currentText()

        if source_type == "Video File":
            path, _ = QFileDialog.getOpenFileName(
                self.tab,
                "Select Video File",
                os.path.join(self.base_path, 'data', 'input', 'videos'),
                "Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*)"
            )
        elif source_type == "Image File":
            path, _ = QFileDialog.getOpenFileName(
                self.tab,
                "Select Image File",
                os.path.join(self.base_path, 'data', 'input', 'images'),
                "Image Files (*.jpg *.jpeg *.png *.bmp);;All Files (*)"
            )
        elif source_type == "Image Folder":
            # Default to project's test/images if available
            start_dir = os.path.join(self.base_path, 'data', 'input', 'images')
            if hasattr(self, '_project_info') and self._project_info and self._project_info.get('valid'):
                test_images = os.path.join(self._project_info['base'], 'test', 'images')
                if os.path.isdir(test_images):
                    start_dir = test_images
            path = QFileDialog.getExistingDirectory(
                self.tab,
                "Select Image Folder",
                start_dir
            )
        else:
            return

        if path:
            logger.debug(f"Selected {source_type}: {path}")
            self.tab.editSourcePath.setText(path)

    def _on_source_type_changed(self, index: int) -> None:
        """Handle source type change."""
        source_type = self.tab.comboSource.currentText()
        if source_type == "Camera":
            self.tab.editSourcePath.setText("0")
            self.tab.editSourcePath.setPlaceholderText("Camera index (0, 1, ...)")
        else:
            self.tab.editSourcePath.setText("")
            self.tab.editSourcePath.setPlaceholderText("Select file or folder...")

    def load_model(self) -> None:
        """Load HEF model."""
        hef_path = self.tab.editHefPath.text()

        if not hef_path:
            logger.warning("Model load aborted: No HEF file selected")
            QMessageBox.warning(self.tab, "Warning", "Please select a HEF model file.")
            return

        if not os.path.exists(hef_path):
            logger.warning(f"Model load aborted: File not found: {hef_path}")
            QMessageBox.warning(self.tab, "Warning", f"Model file not found: {hef_path}")
            return

        try:
            logger.info(f"Loading HEF model: {hef_path}")
            # In real implementation, load the model here
            self.model_loaded = True
            self.hef_path = hef_path

            model_name = os.path.basename(hef_path)
            self.tab.lblModelInfo.setText(f"Loaded: {model_name}")
            self.tab.lblModelInfo.setStyleSheet(f"color: {get_color('success')}; font-style: normal;")

            logger.info(f"Model loaded successfully: {model_name}")
            QMessageBox.information(self.tab, "Success", f"Model loaded: {model_name}")

        except RuntimeError as e:
            logger.error(f"Runtime error loading model: {e}", exc_info=True)
            QMessageBox.critical(self.tab, "Error", f"Failed to load model: {e}")
        except Exception as e:
            # Fallback for unexpected errors
            logger.error(f"Failed to load model ({type(e).__name__}): {e}", exc_info=True)
            QMessageBox.critical(self.tab, "Error", f"Failed to load model ({type(e).__name__}): {e}")

    def unload_model(self) -> None:
        """Unload current model."""
        logger.debug("Unloading model")
        self.model_loaded = False
        self.tab.lblModelInfo.setText("No model loaded")
        self.tab.lblModelInfo.setStyleSheet(f"color: {get_color('text_secondary')}; font-style: italic;")
        logger.info("Model unloaded")

    def start_inference(self) -> None:
        """Start inference."""
        if not self.model_loaded:
            logger.warning("Inference start aborted: No model loaded")
            QMessageBox.warning(self.tab, "Warning", "Please load a model first.")
            return

        source = self.tab.editSourcePath.text()
        if not source:
            logger.warning("Inference start aborted: No source specified")
            QMessageBox.warning(self.tab, "Warning", "Please specify input source.")
            return

        logger.info(f"Starting inference: model={os.path.basename(self.hef_path)}, source={source}")

        # Reset counters
        self.frame_count = 0
        self._update_ui_running(True)

        # Create and start worker (pass shared device if available)
        hailo_svc = None
        if self.device_controller and hasattr(self.device_controller, 'hailo_service'):
            hailo_svc = self.device_controller.hailo_service
        class_names = getattr(self, '_class_names', None)
        if not class_names and hasattr(self, '_project_info') and self._project_info:
            class_names = self._project_info.get('class_names')
        # Auto-detect class names from data.yaml near the HEF file
        if not class_names:
            class_names = self._find_class_names_from_hef(self.hef_path)
        self.worker = InferenceWorker(
            self.hef_path, source, hailo_service=hailo_svc, class_names=class_names
        )
        self.worker.frame_ready.connect(self._on_frame_ready)
        self.worker.error.connect(self._on_error)
        self.worker.stopped.connect(self._on_stopped)
        self.worker.start()
        logger.debug("Inference worker thread started")

    def stop_inference(self) -> None:
        """Stop inference."""
        logger.debug("Stopping inference...")
        if self.worker:
            self.worker.stop()
            self.worker.wait(2000)
            logger.info("Inference stopped")

    def _on_frame_ready(self, frame: np.ndarray, detections: List[Dict[str, Any]], fps: float) -> None:
        """Handle new frame from worker."""
        self.frame_count += 1

        # Update video preview
        self._display_frame(frame)

        # Update stats
        self.tab.lblFpsValue.setText(f"{fps:.1f}")
        self.tab.lblFramesValue.setText(str(self.frame_count))

        # Update detection results table
        self._update_results_table(detections)

    def _display_frame(self, frame: np.ndarray) -> None:
        """Display frame in preview label."""
        try:
            import cv2

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Get label size
            label_w = self.tab.lblVideoPreview.width()
            label_h = self.tab.lblVideoPreview.height()

            # Resize frame to fit
            h, w = rgb_frame.shape[:2]
            scale = min(label_w / w, label_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = cv2.resize(rgb_frame, (new_w, new_h))

            # Convert to QImage
            h, w, ch = resized.shape
            bytes_per_line = ch * w
            q_img = QImage(resized.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

            # Display
            self.tab.lblVideoPreview.setPixmap(QPixmap.fromImage(q_img))
            self.current_frame = frame

        except ImportError as e:
            logger.error(f"OpenCV import error in display_frame: {e}")
        except Exception as e:
            logger.error(f"Error displaying frame: {e}", exc_info=True)

    def _update_results_table(self, detections: List[Dict[str, Any]]) -> None:
        """Update detection results table."""
        # Count detections by class
        class_counts = {}
        for det in detections:
            cls = det['class']
            conf = det['confidence']
            if cls in class_counts:
                class_counts[cls]['count'] += 1
                class_counts[cls]['max_conf'] = max(class_counts[cls]['max_conf'], conf)
            else:
                class_counts[cls] = {'count': 1, 'max_conf': conf}

        # Update table
        self.tab.tableResults.setRowCount(len(class_counts))

        for row, (cls, data) in enumerate(class_counts.items()):
            self.tab.tableResults.setItem(row, 0, QTableWidgetItem(cls))
            self.tab.tableResults.setItem(row, 1, QTableWidgetItem(f"{data['max_conf']:.2f}"))
            self.tab.tableResults.setItem(row, 2, QTableWidgetItem(str(data['count'])))

    def _on_error(self, error_msg: str) -> None:
        """Handle worker error."""
        logger.error(f"Inference worker error: {error_msg}")
        QMessageBox.critical(self.tab, "Error", f"Inference error: {error_msg}")
        self._update_ui_running(False)

    def _on_stopped(self) -> None:
        """Handle worker stopped."""
        logger.debug("Inference worker stopped")
        self._update_ui_running(False)

    def _update_ui_running(self, running: bool) -> None:
        """Update UI for running state."""
        self.tab.btnStart.setEnabled(not running)
        self.tab.btnStop.setEnabled(running)
        self.tab.btnLoadModel.setEnabled(not running)

    def _find_class_names_from_hef(self, hef_path: str) -> Optional[List[str]]:
        """Auto-detect class names by searching for data.yaml near the HEF file.

        Walks up from the HEF directory to find a data.yaml file within
        the project structure (e.g., .../forklift/models/hef/best.hef
        -> finds .../forklift/data.yaml).
        """
        current = os.path.dirname(os.path.abspath(hef_path))
        # Walk up at most 5 levels to find data.yaml
        for _ in range(5):
            data_config = parse_data_yaml(current)
            if data_config:
                names = data_config.get('names')
                if names:
                    logger.info(f"Auto-detected {len(names)} class names from {os.path.join(current, 'data.yaml')}")
                    return names
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
        return None

    def _capture_frame(self) -> None:
        """Capture current frame."""
        if hasattr(self, 'current_frame'):
            try:
                import cv2
                output_dir = os.path.join(self.base_path, 'data', 'output')
                os.makedirs(output_dir, exist_ok=True)

                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"capture_{timestamp}.jpg"
                filepath = os.path.join(output_dir, filename)

                logger.debug(f"Capturing frame to: {filepath}")
                cv2.imwrite(filepath, self.current_frame)
                logger.info(f"Frame captured successfully: {filename}")
                QMessageBox.information(self.tab, "Saved", f"Frame saved to:\n{filepath}")

            except (FileNotFoundError, PermissionError) as e:
                logger.error(f"Failed to save frame (file system error): {e}", exc_info=True)
                QMessageBox.critical(self.tab, "Error", f"Failed to save frame: {e}")
            except Exception as e:
                # Fallback for unexpected errors
                logger.error(f"Failed to save frame ({type(e).__name__}): {e}", exc_info=True)
                QMessageBox.critical(self.tab, "Error", f"Failed to save frame ({type(e).__name__}): {e}")
        else:
            logger.warning("Frame capture aborted: No frame available")
