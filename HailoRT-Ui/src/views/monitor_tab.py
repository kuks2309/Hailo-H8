"""
Monitor Tab Controller
Real-time performance monitoring.
"""

from __future__ import annotations

import os
from collections import deque
from typing import Optional, Dict, Any
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import QTimer
from PyQt5 import uic

from utils.constants import REFRESH_INTERVAL_MS
from utils.helpers import get_temperature_color, generate_mock_device_info
from utils.logger import setup_logger

logger = setup_logger(__name__)


class MonitorTabController:
    """Controller for performance monitoring tab."""

    def __init__(self, tab_widget: QWidget, base_path: str,
                 device_controller=None) -> None:
        self.tab: QWidget = tab_widget
        self.base_path: str = base_path
        self.device_controller = device_controller

        # Data history for charts
        self.fps_history: deque = deque(maxlen=60)
        self.latency_history: deque = deque(maxlen=60)

        # Statistics
        self.fps_sum: float = 0
        self.fps_count: int = 0
        self.latency_sum: float = 0
        self.latency_count: int = 0

        # Load UI into tab
        ui_path = os.path.join(base_path, 'ui', 'tabs', 'monitor_tab.ui')
        uic.loadUi(ui_path, self.tab)

        # Connect signals
        self._connect_signals()

        # Setup refresh timer
        self.refresh_timer: QTimer = QTimer()
        self.refresh_timer.timeout.connect(self._update_stats)

    def _connect_signals(self) -> None:
        """Connect UI signals."""
        self.tab.btnRefresh.clicked.connect(self._update_stats)
        self.tab.btnExport.clicked.connect(self._export_data)
        self.tab.chkAutoRefresh.stateChanged.connect(self._toggle_auto_refresh)

    def start_monitoring(self) -> None:
        """Start performance monitoring."""
        logger.debug("Starting performance monitoring")
        if self.tab.chkAutoRefresh.isChecked():
            self.refresh_timer.start(REFRESH_INTERVAL_MS)
            logger.info(f"Auto-refresh enabled (interval: {REFRESH_INTERVAL_MS}ms)")

    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        logger.debug("Stopping performance monitoring")
        self.refresh_timer.stop()
        logger.info("Performance monitoring stopped")

    def _toggle_auto_refresh(self, state: int) -> None:
        """Toggle auto refresh."""
        if state:
            logger.debug(f"Auto-refresh enabled (interval: {REFRESH_INTERVAL_MS}ms)")
            self.refresh_timer.start(REFRESH_INTERVAL_MS)
        else:
            logger.debug("Auto-refresh disabled")
            self.refresh_timer.stop()

    def update_metrics(self, fps: float, latency: float, throughput: float = 0) -> None:
        """Update metrics from external source (e.g., inference tab)."""
        logger.debug(f"Updating metrics: fps={fps:.1f}, latency={latency:.1f}ms, throughput={throughput:.1f}")

        # Update history
        self.fps_history.append(fps)
        self.latency_history.append(latency)

        # Update statistics
        self.fps_sum += fps
        self.fps_count += 1
        self.latency_sum += latency
        self.latency_count += 1

        # Update display
        self._update_display(fps, latency, throughput)

    def _update_stats(self) -> None:
        """Update statistics display using shared device controller."""
        try:
            # Use shared device controller if available and connected
            if (self.device_controller
                    and hasattr(self.device_controller, 'hailo_service')
                    and self.device_controller.hailo_service
                    and self.device_controller.is_connected):
                logger.debug("Fetching device stats from shared HailoService")
                info = self.device_controller.hailo_service.get_device_info()
                self._update_device_stats(info)
            else:
                logger.debug("No connected device, using mock stats")
                self._update_mock_stats()

        except Exception as e:
            logger.error(f"Error updating stats: {e}", exc_info=True)

    def _update_display(self, fps: float, latency: float, throughput: float) -> None:
        """Update display values."""
        # FPS
        self.tab.lblFpsValue.setText(f"{fps:.1f}")
        if self.fps_count > 0:
            avg_fps = self.fps_sum / self.fps_count
            self.tab.lblFpsAvg.setText(f"Avg: {avg_fps:.1f}")

        # Latency
        self.tab.lblLatencyValue.setText(f"{latency:.1f} ms")
        if self.latency_count > 0:
            avg_latency = self.latency_sum / self.latency_count
            self.tab.lblLatencyAvg.setText(f"Avg: {avg_latency:.1f} ms")

        # Throughput
        self.tab.lblThroughputValue.setText(f"{throughput:.1f} TOPS")

    def _update_device_stats(self, info: Dict[str, Any]) -> None:
        """Update device statistics."""
        logger.debug(f"Updating device stats: temp={info.get('temperature', 0):.1f}°C, "
                    f"util={info.get('utilization', 0):.0f}%, power={info.get('power', 0):.1f}W")

        # Temperature
        temp = info.get('temperature', 0)
        self.tab.lblTempValue.setText(f"{temp:.0f} °C")
        self._set_temp_status(temp)

        # Log warning for high temperature
        if temp > 80:
            logger.warning(f"High device temperature detected: {temp:.1f}°C")

        # NPU utilization
        util = info.get('utilization', 0)
        self.tab.progressNpu.setValue(int(util))
        self.tab.lblNpuValue.setText(f"{util:.0f}%")

        # Power
        power = info.get('power', 0)
        self.tab.progressPower.setValue(int(power))
        self.tab.lblPowerValue.setText(f"{power:.1f} W")

    def _update_mock_stats(self) -> None:
        """Update with mock statistics."""
        import random

        # Mock FPS and latency
        fps = random.uniform(100, 150)
        latency = random.uniform(6, 10)
        throughput = random.uniform(10, 20)

        self.update_metrics(fps, latency, throughput)

        # Mock device stats using centralized generator
        mock_info = generate_mock_device_info()

        temp = mock_info.get('temperature', 45)
        self.tab.lblTempValue.setText(f"{temp:.0f} °C")
        self._set_temp_status(temp)

        util = mock_info.get('utilization', 0)
        self.tab.progressNpu.setValue(int(util))
        self.tab.lblNpuValue.setText(f"{util:.0f}%")

        mem = random.uniform(100, 500)
        self.tab.progressMem.setValue(int(mem / 10))
        self.tab.lblMemValue.setText(f"{mem:.0f} MB")

        power = mock_info.get('power', 2.5)
        self.tab.progressPower.setValue(int(power))
        self.tab.lblPowerValue.setText(f"{power:.1f} W")

    def _set_temp_status(self, temp: float) -> None:
        """Set temperature status."""
        color, status = get_temperature_color(temp)
        self.tab.lblTempStatus.setText(f"Status: {status}")
        self.tab.lblTempValue.setStyleSheet(
            f"font-size: 42px; font-weight: bold; color: {color};"
        )

    def _export_data(self) -> None:
        """Export performance data to CSV."""
        from PyQt5.QtWidgets import QFileDialog, QMessageBox
        import csv
        from datetime import datetime

        logger.debug("Exporting performance data")
        path, _ = QFileDialog.getSaveFileName(
            self.tab,
            "Export Performance Data",
            os.path.join(self.base_path, 'data', 'output', 'performance.csv'),
            "CSV Files (*.csv)"
        )

        if path:
            try:
                logger.info(f"Exporting {len(self.fps_history)} data points to: {path}")
                with open(path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Timestamp', 'FPS', 'Latency (ms)'])

                    timestamp = datetime.now()
                    for fps, latency in zip(self.fps_history, self.latency_history):
                        writer.writerow([
                            timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                            f"{fps:.2f}",
                            f"{latency:.2f}"
                        ])

                logger.info(f"Performance data exported successfully: {os.path.basename(path)}")
                QMessageBox.information(
                    self.tab, "Success",
                    f"Data exported to:\n{path}"
                )

            except (FileNotFoundError, PermissionError) as e:
                logger.error(f"Failed to export data (file system error): {e}", exc_info=True)
                QMessageBox.critical(
                    self.tab, "Error",
                    f"Failed to export data: {e}"
                )
            except Exception as e:
                logger.error(f"Failed to export data: {e}", exc_info=True)
                QMessageBox.critical(
                    self.tab, "Error",
                    f"Failed to export data: {e}"
                )

    def reset_statistics(self) -> None:
        """Reset all statistics."""
        logger.debug("Resetting performance statistics")
        self.fps_history.clear()
        self.latency_history.clear()
        self.fps_sum = 0
        self.fps_count = 0
        self.latency_sum = 0
        self.latency_count = 0
        logger.info("Performance statistics reset")
