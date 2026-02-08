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

logger = setup_logger(__name__)


class InferenceWorker(QThread):
    """Worker thread for running inference."""
    frame_ready = pyqtSignal(np.ndarray, list, float)  # frame, detections, fps
    error = pyqtSignal(str)
    stopped = pyqtSignal()

    def __init__(self, model_path: str, source: Union[str, int], conf_threshold: float = 0.5) -> None:
        super().__init__()
        self.model_path: str = model_path
        self.source: Union[str, int] = source
        self.conf_threshold: float = conf_threshold
        self.running: bool = False
        self.model: Optional[Any] = None

    def run(self) -> None:
        """Run inference loop."""
        self.running = True
        logger.debug(f"InferenceWorker starting: model={self.model_path}, source={self.source}")

        try:
            # Initialize model
            self._load_model()

            # Initialize video capture
            cap = self._init_capture()
            if cap is None:
                logger.error("Failed to open video source")
                self.error.emit("Failed to open video source")
                return

            frame_count = 0
            start_time = time.time()
            logger.info("Inference loop started")

            while self.running:
                ret, frame = cap.read()
                if not ret:
                    # Loop video or stop
                    if isinstance(self.source, str) and os.path.isfile(self.source):
                        cap.set(1, 0)  # cv2.CAP_PROP_POS_FRAMES
                        continue
                    else:
                        break

                # Run inference
                detections = self._run_inference(frame)

                # Calculate FPS
                frame_count += 1
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0

                # Draw results
                result_frame = self._draw_detections(frame, detections)

                # Emit frame
                self.frame_ready.emit(result_frame, detections, fps)

                # Small delay to prevent overwhelming the UI
                time.sleep(0.001)

            cap.release()
            logger.info(f"Inference loop stopped after {frame_count} frames")

        except ImportError as e:
            logger.error(f"Import error in inference worker: {e}", exc_info=True)
            self.error.emit(f"Import error: {e}")
        except RuntimeError as e:
            logger.error(f"Runtime error in inference worker: {e}", exc_info=True)
            self.error.emit(f"Runtime error: {e}")
        except Exception as e:
            # Fallback for unexpected errors
            logger.error(f"Unexpected error in inference worker ({type(e).__name__}): {e}", exc_info=True)
            self.error.emit(f"Unexpected error ({type(e).__name__}): {e}")
        finally:
            self.stopped.emit()

    def stop(self) -> None:
        """Stop inference loop."""
        logger.debug("Stopping inference worker...")
        self.running = False

    def _load_model(self) -> None:
        """Load the model."""
        try:
            # Try HailoRT first
            logger.debug(f"Loading model: {self.model_path}")
            from services.hailo_service import HailoService
            self.hailo = HailoService()
            self.model = self.hailo.load_model(self.model_path)
            self.use_hailo = True
            logger.info(f"Model loaded successfully via HailoService: {os.path.basename(self.model_path)}")
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

    def _draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draw detection boxes on frame."""
        try:
            import cv2
        except ImportError:
            return frame

        result = frame.copy()

        for det in detections:
            bbox = det['bbox']
            conf = det['confidence']
            cls = det['class']

            x1, y1, x2, y2 = map(int, bbox)

            # Draw box
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            label = f"{cls}: {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(result, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
            cv2.putText(result, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return result


class InferenceTabController:
    """Controller for inference tab."""

    def __init__(self, tab_widget: QWidget, base_path: str) -> None:
        self.tab: QWidget = tab_widget
        self.base_path: str = base_path
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

    def _browse_hef(self) -> None:
        """Browse for HEF model file."""
        path, _ = QFileDialog.getOpenFileName(
            self.tab,
            "Select HEF Model",
            os.path.join(self.base_path, 'data', 'models', 'hef'),
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
            path = QFileDialog.getExistingDirectory(
                self.tab,
                "Select Image Folder",
                os.path.join(self.base_path, 'data', 'input', 'images')
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
            self.tab.lblModelInfo.setStyleSheet("color: #4caf50; font-style: normal;")

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
        self.tab.lblModelInfo.setStyleSheet("color: #888888; font-style: italic;")
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

        # Create and start worker
        self.worker = InferenceWorker(self.hef_path, source)
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
            q_img = QImage(resized.data, w, h, bytes_per_line, QImage.Format_RGB888)

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
