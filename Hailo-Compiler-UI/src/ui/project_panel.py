"""
Project Panel widget for Hailo-Compiler-UI.
Handles project folder selection and structure validation.
"""

import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton,
    QFileDialog, QMessageBox, QFrame
)
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QFont

from ..utils.folder_structure import (
    validate_folder_structure,
    get_missing_folders,
    create_folder_structure,
    get_project_paths,
    find_calibration_folder,
    REQUIRED_FOLDERS,
    CALIBRATION_FOLDERS
)


class ProjectPanel(QWidget):
    """
    Panel for project folder selection and validation.
    """

    project_changed = pyqtSignal(dict)  # Emits project paths when folder changes

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project_path = ""
        self._setup_ui()

    def _setup_ui(self):
        """Setup the UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        group = QGroupBox("Project Folder")
        main_layout = QVBoxLayout(group)

        # Folder selection row
        folder_layout = QHBoxLayout()

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Select project folder...")
        self.path_edit.setReadOnly(True)
        folder_layout.addWidget(self.path_edit)

        self.browse_btn = QPushButton("Browse")
        self.browse_btn.setProperty('browse', True)
        self.browse_btn.clicked.connect(self._browse_folder)
        self.browse_btn.setFixedWidth(80)
        folder_layout.addWidget(self.browse_btn)

        main_layout.addLayout(folder_layout)

        # Status frame (hidden by default)
        self.status_frame = QFrame()
        status_layout = QVBoxLayout(self.status_frame)
        status_layout.setContentsMargins(8, 8, 8, 8)

        # Status label
        self.status_label = QLabel()
        self.status_label.setFont(QFont("Consolas", 9))
        status_layout.addWidget(self.status_label)

        # Folder status grid
        self.folder_grid = QGridLayout()
        self.folder_grid.setSpacing(4)
        status_layout.addLayout(self.folder_grid)

        # Create missing folders button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.create_btn = QPushButton("Create Missing Folders")
        self.create_btn.setProperty('accent', True)
        self.create_btn.clicked.connect(self._create_missing_folders)
        self.create_btn.setVisible(False)
        btn_layout.addWidget(self.create_btn)

        status_layout.addLayout(btn_layout)

        self.status_frame.setVisible(False)
        main_layout.addWidget(self.status_frame)

        layout.addWidget(group)

    def _browse_folder(self):
        """Browse for project folder."""
        path = QFileDialog.getExistingDirectory(
            self, "Select Project Folder",
            os.path.expanduser("~")
        )
        if path:
            self._set_project_path(path)

    def _set_project_path(self, path: str):
        """Set the project path and validate structure."""
        self.project_path = path
        self.path_edit.setText(path)

        # Validate folder structure
        all_ok, results = validate_folder_structure(path)

        # Show status frame
        self.status_frame.setVisible(True)

        # Clear existing folder status labels
        while self.folder_grid.count():
            item = self.folder_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Add folder status labels
        row = 0
        col = 0
        max_cols = 3

        for folder_info in REQUIRED_FOLDERS:
            exists = results.get(folder_info.path, False)

            if exists:
                icon = "✓"
                color = "#4CAF50"
            else:
                icon = "✗" if folder_info.required else "○"
                color = "#F44336" if folder_info.required else "#FFC107"

            label = QLabel(f"{icon} {folder_info.path}")
            label.setStyleSheet(f"color: {color}; font-family: Consolas; font-size: 9pt;")
            label.setToolTip(folder_info.description)

            self.folder_grid.addWidget(label, row, col)

            col += 1
            if col >= max_cols:
                col = 0
                row += 1

        # Add calibration source status (special handling)
        calib_exists = results.get('calibration_source', False)
        calib_path, _ = find_calibration_folder(path)
        calib_folder = os.path.basename(os.path.dirname(calib_path)) + "/images"

        if calib_exists:
            icon = "✓"
            color = "#4CAF50"
            calib_text = f"{icon} calibration ({calib_folder})"
            tooltip = f"Calibration images found in: {calib_path}"
        else:
            icon = "✗"
            color = "#F44336"
            calib_text = f"{icon} calibration (missing)"
            tooltip = "Need images in: train/images, valid/images, or test/images"

        calib_label = QLabel(calib_text)
        calib_label.setStyleSheet(f"color: {color}; font-family: Consolas; font-size: 9pt;")
        calib_label.setToolTip(tooltip)
        self.folder_grid.addWidget(calib_label, row, col)
        col += 1
        if col >= max_cols:
            col = 0
            row += 1

        # Update status label and button
        if all_ok:
            self.status_label.setText("✓ Folder structure is valid")
            self.status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
            self.create_btn.setVisible(False)

            # Emit project paths
            paths = get_project_paths(path)
            self.project_changed.emit(paths)
        else:
            missing = get_missing_folders(path, required_only=True)
            self.status_label.setText(f"✗ Missing {len(missing)} required folder(s)")
            self.status_label.setStyleSheet("color: #F44336; font-weight: bold;")
            self.create_btn.setVisible(True)

    def _create_missing_folders(self):
        """Create missing folders."""
        if not self.project_path:
            return

        missing = get_missing_folders(self.project_path)

        if not missing:
            QMessageBox.information(self, "Info", "All folders already exist.")
            return

        # Confirm creation
        folder_list = "\n".join([f"  • {f.path}" for f in missing])
        reply = QMessageBox.question(
            self, "Create Folders",
            f"The following folders will be created:\n\n{folder_list}\n\nContinue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )

        if reply != QMessageBox.Yes:
            return

        # Create folders
        success, created = create_folder_structure(self.project_path, missing)

        if success:
            QMessageBox.information(
                self, "Success",
                f"Created {len(created)} folder(s):\n\n" + "\n".join([f"  • {f}" for f in created])
            )
            # Refresh validation
            self._set_project_path(self.project_path)
        else:
            QMessageBox.critical(
                self, "Error",
                f"Failed to create folders:\n\n{created[0]}"
            )

    def get_project_path(self) -> str:
        """Get the current project path."""
        return self.project_path

    def get_project_paths(self) -> dict:
        """Get all project paths."""
        if self.project_path:
            return get_project_paths(self.project_path)
        return {}
