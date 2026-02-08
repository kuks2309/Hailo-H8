"""
Device Tab Controller
Manages device connection and status monitoring.
"""

from __future__ import annotations

import os
from typing import Optional, Dict, Any
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import QTimer
from PyQt5 import uic

from utils.constants import REFRESH_INTERVAL_MS
from utils.helpers import get_temperature_color, generate_mock_device_info
from utils.logger import setup_logger

logger = setup_logger(__name__)


class DeviceTabController:
    """Controller for device monitoring tab."""

    def __init__(self, tab_widget: QWidget, base_path: str) -> None:
        self.tab: QWidget = tab_widget
        self.base_path: str = base_path
        self.device: Optional[Any] = None
        self.is_connected: bool = False

        # Load UI into tab
        ui_path = os.path.join(base_path, 'ui', 'tabs', 'device_tab.ui')
        uic.loadUi(ui_path, self.tab)

        # Connect signals
        self._connect_signals()

        # Setup refresh timer
        self.refresh_timer: QTimer = QTimer()
        self.refresh_timer.timeout.connect(self._update_device_status)

    def _connect_signals(self) -> None:
        """Connect UI signals."""
        self.tab.btnConnect.clicked.connect(self.connect_device)
        self.tab.btnDisconnect.clicked.connect(self.disconnect_device)
        self.tab.btnRefresh.clicked.connect(self.refresh_status)

    def connect_device(self) -> None:
        """Connect to Hailo device."""
        logger.debug("Attempting to connect to Hailo device...")
        try:
            from services.hailo_service import HailoService
            self.hailo_service = HailoService()
            self.device = self.hailo_service.connect()

            if self.device:
                self.is_connected = True
                device_name = getattr(self.device, 'device_id', 'Unknown')
                logger.info(f"Device connected successfully: {device_name}")
                self._update_ui_connected()
                self.refresh_timer.start(REFRESH_INTERVAL_MS)
                logger.debug(f"Status refresh timer started (interval: {REFRESH_INTERVAL_MS}ms)")
                self._update_device_status()
            else:
                logger.warning("No Hailo device found")
                self._show_no_device()

        except ImportError:
            # HailoRT not installed, show mock data
            logger.warning("HailoRT not installed, running in mock mode")
            self._show_mock_device()
        except RuntimeError as e:
            logger.error(f"Device connection runtime error: {e}", exc_info=True)
            self.tab.txtDeviceInfo.setPlainText(f"Connection error: {str(e)}")
        except Exception as e:
            # Final fallback for unexpected errors (e.g., HailoRT-specific exceptions)
            logger.error(f"Failed to connect to device ({type(e).__name__}): {e}", exc_info=True)
            self.tab.txtDeviceInfo.setPlainText(f"Connection error: {str(e)}")

    def disconnect_device(self) -> None:
        """Disconnect from Hailo device."""
        logger.debug("Disconnecting from Hailo device...")
        self.refresh_timer.stop()
        self.is_connected = False
        self.device = None
        logger.info("Device disconnected successfully")
        self._update_ui_disconnected()

    def refresh_status(self) -> None:
        """Refresh device status."""
        logger.debug("Manual refresh status requested")
        if self.is_connected:
            self._update_device_status()
        else:
            logger.debug("Refresh skipped - no device connected")

    def _update_device_status(self) -> None:
        """Update device status displays."""
        if not self.is_connected:
            return

        try:
            if hasattr(self, 'hailo_service'):
                logger.debug("Fetching device info from HailoService...")
                info = self.hailo_service.get_device_info()
                self._display_device_info(info)

                # Log warnings for high temperature
                temp = info.get('temperature', 0)
                if temp > 80:
                    logger.warning(f"Device temperature high: {temp:.1f}°C")
            else:
                logger.debug("Displaying mock status data")
                self._show_mock_status()
        except AttributeError as e:
            logger.error(f"Device attribute error: {e}", exc_info=True)
            self.tab.txtDeviceInfo.append(f"Error updating status: {e}")
        except Exception as e:
            # Final fallback for PyQt signals and HailoRT-specific exceptions
            logger.error(f"Error updating device status ({type(e).__name__}): {e}", exc_info=True)
            self.tab.txtDeviceInfo.append(f"Error updating status: {e}")

    def _display_device_info(self, info: Dict[str, Any]) -> None:
        """Display device information."""
        logger.debug(f"Displaying device info: temp={info.get('temperature', 0):.1f}°C, "
                    f"util={info.get('utilization', 0):.0f}%, power={info.get('power', 0):.1f}W")

        # Update labels
        self.tab.lblDeviceValue.setText(info.get('device_name', 'Unknown'))
        self.tab.lblSerialValue.setText(info.get('serial', 'N/A'))
        self.tab.lblFirmwareValue.setText(info.get('firmware', 'N/A'))
        self.tab.lblDriverValue.setText(info.get('driver', 'N/A'))

        # Update temperature
        temp = info.get('temperature', 0)
        self.tab.lblTempValue.setText(f"{temp:.1f} °C")
        self.tab.progressTemp.setValue(int(temp))
        self._set_temp_color(temp)

        # Update power
        power = info.get('power', 0)
        self.tab.lblPowerValue.setText(f"{power:.1f} W")
        self.tab.progressPower.setValue(int(power * 10))  # Scale for display

        # Update utilization
        util = info.get('utilization', 0)
        self.tab.lblUtilValue.setText(f"{util:.0f} %")
        self.tab.progressUtil.setValue(int(util))

        # Update info text
        info_text = (
            f"Device: {info.get('device_name', 'N/A')}\n"
            f"Architecture: {info.get('architecture', 'N/A')}\n"
            f"Serial Number: {info.get('serial', 'N/A')}\n"
            f"Firmware Version: {info.get('firmware', 'N/A')}\n"
            f"Driver Version: {info.get('driver', 'N/A')}\n"
            f"\n--- Real-time Status ---\n"
            f"Temperature: {temp:.1f} °C\n"
            f"Power: {power:.2f} W\n"
            f"Utilization: {util:.0f}%\n"
        )
        self.tab.txtDeviceInfo.setPlainText(info_text)

    def _set_temp_color(self, temp: float) -> None:
        """Set temperature label color based on value."""
        color, _ = get_temperature_color(temp)
        self.tab.lblTempValue.setStyleSheet(
            f"font-size: 36px; font-weight: bold; color: {color};"
        )

    def _update_ui_connected(self) -> None:
        """Update UI for connected state."""
        self.tab.lblStatusValue.setText("● Connected")
        self.tab.lblStatusValue.setStyleSheet("color: #4caf50; font-weight: bold;")
        self.tab.btnConnect.setEnabled(False)
        self.tab.btnDisconnect.setEnabled(True)

    def _update_ui_disconnected(self) -> None:
        """Update UI for disconnected state."""
        self.tab.lblStatusValue.setText("● Disconnected")
        self.tab.lblStatusValue.setStyleSheet("color: #f44336; font-weight: bold;")
        self.tab.btnConnect.setEnabled(True)
        self.tab.btnDisconnect.setEnabled(False)

        # Reset values
        self.tab.lblDeviceValue.setText("-")
        self.tab.lblSerialValue.setText("-")
        self.tab.lblFirmwareValue.setText("-")
        self.tab.lblDriverValue.setText("-")
        self.tab.lblTempValue.setText("-- °C")
        self.tab.lblPowerValue.setText("-- W")
        self.tab.lblUtilValue.setText("-- %")
        self.tab.progressTemp.setValue(0)
        self.tab.progressPower.setValue(0)
        self.tab.progressUtil.setValue(0)
        self.tab.txtDeviceInfo.setPlainText("No device connected.")

    def _show_no_device(self) -> None:
        """Show no device found message."""
        logger.warning("No Hailo device found on system")
        self.tab.txtDeviceInfo.setPlainText(
            "No Hailo device found.\n\n"
            "Please ensure:\n"
            "1. Hailo-8 device is connected\n"
            "2. PCIe driver is installed\n"
            "3. HailoRT is properly installed"
        )
        self._update_ui_disconnected()

    def _show_mock_device(self) -> None:
        """Show mock device data for testing."""
        logger.info("Running in mock mode with simulated device data")
        self.is_connected = True
        self._update_ui_connected()

        mock_info = generate_mock_device_info()
        self._display_device_info(mock_info)
        self.tab.txtDeviceInfo.append("\n[Mock Mode - HailoRT not installed]")

    def _show_mock_status(self) -> None:
        """Show mock status updates."""
        mock_info = generate_mock_device_info()
        self._display_device_info(mock_info)
