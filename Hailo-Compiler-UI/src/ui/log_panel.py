"""
Log Panel widget for Hailo-Compiler-UI.
Displays conversion logs and progress.
"""

from datetime import datetime
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QTextEdit, QProgressBar, QPushButton,
    QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QTextCharFormat, QColor, QTextCursor

from .styles import get_color


class LogPanel(QWidget):
    """
    Log panel widget with text output and progress bar.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        """Setup the UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Log group box
        group = QGroupBox("Log")
        group_layout = QVBoxLayout(group)

        # Log text area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(150)
        group_layout.addWidget(self.log_text)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        group_layout.addWidget(self.progress_bar)

        # Button row
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.export_btn = QPushButton("Export Log")
        self.export_btn.clicked.connect(self._export_log)
        button_layout.addWidget(self.export_btn)

        self.clear_btn = QPushButton("Clear Log")
        self.clear_btn.clicked.connect(self.clear)
        button_layout.addWidget(self.clear_btn)

        group_layout.addLayout(button_layout)

        layout.addWidget(group)

    def log(self, message: str, level: str = 'info'):
        """
        Add a log message.

        Args:
            message: Log message text
            level: Log level ('info', 'success', 'warning', 'error')
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {message}"

        # Set text color based on level
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.End)

        fmt = QTextCharFormat()
        if level == 'success':
            fmt.setForeground(QColor(get_color('success')))
        elif level == 'warning':
            fmt.setForeground(QColor(get_color('warning')))
        elif level == 'error':
            fmt.setForeground(QColor(get_color('error')))
        else:
            fmt.setForeground(QColor(get_color('text')))

        cursor.insertText(formatted + "\n", fmt)
        self.log_text.setTextCursor(cursor)
        self.log_text.ensureCursorVisible()

    def log_info(self, message: str):
        """Log an info message."""
        self.log(message, 'info')

    def log_success(self, message: str):
        """Log a success message."""
        self.log(message, 'success')

    def log_warning(self, message: str):
        """Log a warning message."""
        self.log(message, 'warning')

    def log_error(self, message: str):
        """Log an error message."""
        self.log(message, 'error')

    def set_progress(self, value: int):
        """
        Set progress bar value.

        Args:
            value: Progress percentage (0-100)
        """
        self.progress_bar.setValue(value)

        # Reset error state when progress starts
        if value > 0 and value < 100:
            self.progress_bar.setProperty('error', False)
            self.progress_bar.style().unpolish(self.progress_bar)
            self.progress_bar.style().polish(self.progress_bar)

    def set_error_state(self, error: bool = True):
        """
        Set progress bar to error state.

        Args:
            error: Whether to show error state
        """
        self.progress_bar.setProperty('error', error)
        self.progress_bar.style().unpolish(self.progress_bar)
        self.progress_bar.style().polish(self.progress_bar)

    def reset_progress(self):
        """Reset progress bar to 0."""
        self.progress_bar.setValue(0)
        self.set_error_state(False)

    def clear(self):
        """Clear log and reset progress."""
        self.log_text.clear()
        self.reset_progress()

    def get_log_text(self) -> str:
        """Get all log text."""
        return self.log_text.toPlainText()

    def _export_log(self):
        """Export log content to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"hailo_compile_log_{timestamp}.txt"

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Log",
            default_name,
            "Text Files (*.txt);;All Files (*)"
        )

        if filepath:
            try:
                with open(filepath, 'w') as f:
                    f.write(self.log_text.toPlainText())
                self.log(f"Log exported to: {filepath}", 'success')
            except Exception as e:
                QMessageBox.warning(self, "Export Failed", f"Could not export log: {e}")
