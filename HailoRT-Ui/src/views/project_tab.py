"""
Project Tab Controller
Manages dataset/project folder selection and auto-path configuration.
Emits project_changed signal to other tabs.
"""

from __future__ import annotations

import json
import os
from typing import Optional, Dict, List
from PyQt5.QtWidgets import QWidget, QFileDialog, QTreeWidgetItem
from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtGui import QColor, QFont
from PyQt5 import uic

from utils.logger import setup_logger
from utils.folder_structure import detect_project_structure

logger = setup_logger(__name__)

MAX_RECENT_FOLDERS = 5


class ProjectTabSignals(QObject):
    """Signals for project tab - separated for use with non-QObject controller."""
    project_changed = pyqtSignal(dict)


class ProjectTabController:
    """Controller for project/dataset folder management tab."""

    def __init__(self, tab_widget: QWidget, base_path: str) -> None:
        self.tab: QWidget = tab_widget
        self.base_path: str = base_path
        self.signals = ProjectTabSignals()
        self._project_info: Optional[Dict] = None
        self._recent_file = os.path.join(base_path, 'recent_folders.json')

        # Load UI into tab
        ui_path = os.path.join(base_path, 'ui', 'tabs', 'project_tab.ui')
        uic.loadUi(ui_path, self.tab)

        # Load recent folders into combobox
        self._load_recent_folders()

        # Connect signals
        self._connect_signals()

    @property
    def project_changed(self) -> pyqtSignal:
        """Convenience accessor for the project_changed signal."""
        return self.signals.project_changed

    @property
    def project_info(self) -> Optional[Dict]:
        """Current project info."""
        return self._project_info

    def _connect_signals(self) -> None:
        """Connect UI signals."""
        self.tab.btnBrowseFolder.clicked.connect(self._browse_folder)
        self.tab.btnApplyToTabs.clicked.connect(self._apply_to_tabs)
        self.tab.comboFolderPath.activated.connect(self._on_combo_selected)

    # ---- Recent Folders ----

    def _load_recent_folders(self) -> None:
        """Load recent folders from JSON file into combobox."""
        folders = self._read_recent_list()
        combo = self.tab.comboFolderPath
        combo.clear()
        for folder in folders:
            combo.addItem(folder)

    def _read_recent_list(self) -> List[str]:
        """Read the recent folders list from disk."""
        if not os.path.exists(self._recent_file):
            return []
        try:
            with open(self._recent_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                return [p for p in data if isinstance(p, str)][:MAX_RECENT_FOLDERS]
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to read recent folders: {e}")
        return []

    def _save_recent_folder(self, path: str) -> None:
        """Add a path to the recent folders list and persist."""
        folders = self._read_recent_list()

        # Remove if already exists, then prepend
        if path in folders:
            folders.remove(path)
        folders.insert(0, path)
        folders = folders[:MAX_RECENT_FOLDERS]

        try:
            with open(self._recent_file, 'w', encoding='utf-8') as f:
                json.dump(folders, f, ensure_ascii=False, indent=2)
        except OSError as e:
            logger.warning(f"Failed to save recent folders: {e}")

        # Update combobox
        combo = self.tab.comboFolderPath
        combo.clear()
        for folder in folders:
            combo.addItem(folder)
        combo.setCurrentIndex(0)

    def _on_combo_selected(self, index: int) -> None:
        """Handle selection from the recent folders combobox."""
        path = self.tab.comboFolderPath.currentText()
        if path and os.path.isdir(path):
            self._save_recent_folder(path)
            self._detect_and_display(path)

    # ---- Browse ----

    def _browse_folder(self) -> None:
        """Browse for dataset/project folder."""
        current = self.tab.comboFolderPath.currentText()
        start_dir = current if current and os.path.isdir(current) else os.path.expanduser('~')

        path = QFileDialog.getExistingDirectory(
            self.tab,
            "Select Dataset / Project Folder",
            start_dir
        )
        if path:
            self._save_recent_folder(path)
            self._detect_and_display(path)

    # ---- Detection & Display ----

    def _detect_and_display(self, folder_path: str) -> None:
        """Detect project structure and update the UI."""
        info = detect_project_structure(folder_path)
        self._project_info = info

        # Always show the full tree regardless of detection result
        self._update_tree(folder_path, info)

        if not info.get('valid'):
            self.tab.lblFolderStatus.setText(
                "No dataset structure detected in this folder."
            )
            self.tab.lblFolderStatus.setStyleSheet(
                "color: #c62828; font-style: italic; padding-left: 4px;"
            )
            self.tab.btnApplyToTabs.setEnabled(False)
            self._clear_info_labels()
            return

        # Update status
        self.tab.lblFolderStatus.setText(f"Detected: {info['summary']}")
        self.tab.lblFolderStatus.setStyleSheet(
            "color: #2e7d32; font-weight: bold; padding-left: 4px;"
        )
        self.tab.btnApplyToTabs.setEnabled(True)

        # Update detected info panel
        self._update_info_panel(info)

        # Update auto-path display
        self._update_path_display(info)

        logger.info(f"Project detected: {folder_path} -> {info['summary']}")

        # Auto-apply to other tabs immediately
        self.signals.project_changed.emit(info)
        self.tab.btnApplyToTabs.setText("Applied!")
        self.tab.btnApplyToTabs.setStyleSheet(
            "background-color: #2e7d32; color: white; font-size: 14px; "
            "font-weight: bold; padding: 10px 30px; border-radius: 5px;"
        )
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(2000, self._reset_apply_button)

    def _update_info_panel(self, info: Dict) -> None:
        """Update the detected information panel."""
        # Classes
        nc = info.get('num_classes')
        names = info.get('class_names')
        if nc:
            class_text = f"{nc} classes"
            if names and len(names) <= 10:
                class_text += f": {', '.join(str(n) for n in names)}"
            elif names:
                class_text += f": {', '.join(str(n) for n in names[:10])}..."
            self.tab.lblClasses.setText(class_text)
            self.tab.lblClasses.setStyleSheet("color: #1565c0;")
        else:
            self.tab.lblClasses.setText("-")
            self.tab.lblClasses.setStyleSheet("color: #999999;")

        # Calibration
        if info.get('calib_found'):
            count = info.get('calib_count', 0)
            calib_name = os.path.basename(os.path.dirname(info['calib_dir']))
            self.tab.lblCalibInfo.setText(f"{calib_name}/images ({count} images)")
            self.tab.lblCalibInfo.setStyleSheet("color: #2e7d32;")
        else:
            self.tab.lblCalibInfo.setText("Not found")
            self.tab.lblCalibInfo.setStyleSheet("color: #c62828;")

        # PT Models
        pt_files = info.get('pt_files', [])
        if pt_files:
            names_str = ', '.join(os.path.basename(f) for f in pt_files)
            self.tab.lblPtInfo.setText(f"{len(pt_files)} files: {names_str}")
            self.tab.lblPtInfo.setStyleSheet("color: #4a148c;")
        else:
            self.tab.lblPtInfo.setText("Not found")
            self.tab.lblPtInfo.setStyleSheet("color: #999999;")

        # ONNX Models
        onnx_files = info.get('onnx_files', [])
        if onnx_files and hasattr(self.tab, 'lblOnnxInfo'):
            names_str = ', '.join(os.path.basename(f) for f in onnx_files)
            self.tab.lblOnnxInfo.setText(f"{len(onnx_files)} files: {names_str}")
            self.tab.lblOnnxInfo.setStyleSheet("color: #1565c0;")
        elif hasattr(self.tab, 'lblOnnxInfo'):
            self.tab.lblOnnxInfo.setText("Not found")
            self.tab.lblOnnxInfo.setStyleSheet("color: #999999;")

        # HEF Models
        hef_files = info.get('hef_files', [])
        if hef_files and hasattr(self.tab, 'lblHefInfo'):
            names_str = ', '.join(os.path.basename(f) for f in hef_files)
            self.tab.lblHefInfo.setText(f"{len(hef_files)} files: {names_str}")
            self.tab.lblHefInfo.setStyleSheet("color: #e65100;")
        elif hasattr(self.tab, 'lblHefInfo'):
            self.tab.lblHefInfo.setText("Not found")
            self.tab.lblHefInfo.setStyleSheet("color: #999999;")

        # data.yaml
        if info.get('data_yaml'):
            self.tab.lblDataYamlInfo.setText("Found")
            self.tab.lblDataYamlInfo.setStyleSheet("color: #2e7d32;")
        else:
            self.tab.lblDataYamlInfo.setText("Not found")
            self.tab.lblDataYamlInfo.setStyleSheet("color: #999999;")

    # ---- Full Directory Tree ----

    def _update_tree(self, folder_path: str, info: Dict) -> None:
        """Build a full directory tree from the actual filesystem."""
        tree = self.tab.treeStructure
        tree.clear()

        root_name = os.path.basename(folder_path) or folder_path
        root_item = QTreeWidgetItem([root_name, ''])
        root_item.setExpanded(True)
        bold_font = QFont()
        bold_font.setBold(True)
        root_item.setFont(0, bold_font)
        tree.addTopLevelItem(root_item)

        # Known dataset directories for highlighting
        known_dirs = {
            'train', 'valid', 'test', 'models', 'configs', 'logs',
            'images', 'labels', 'pt', 'onnx', 'har', 'hef',
            'inference_output', 'scripts',
        }
        known_files = {'data.yaml'}

        self._populate_tree_recursive(
            folder_path, root_item, info, known_dirs, known_files, depth=0
        )

        tree.resizeColumnToContents(0)

    def _populate_tree_recursive(
        self,
        dir_path: str,
        parent_item: QTreeWidgetItem,
        info: Dict,
        known_dirs: set,
        known_files: set,
        depth: int,
    ) -> None:
        """Recursively populate tree items from actual filesystem."""
        try:
            entries = sorted(os.listdir(dir_path))
        except OSError:
            return

        # Separate dirs and files, dirs first
        dirs = []
        files = []
        for entry in entries:
            full = os.path.join(dir_path, entry)
            if os.path.isdir(full):
                # Skip hidden dirs and __pycache__
                if entry.startswith('.') or entry == '__pycache__':
                    continue
                dirs.append(entry)
            else:
                if entry.startswith('.'):
                    continue
                files.append(entry)

        # Add directories
        for d in dirs:
            full = os.path.join(dir_path, d)
            display = d + '/'

            # Count contents for status
            try:
                count = len([e for e in os.listdir(full)
                             if not e.startswith('.')])
            except OSError:
                count = 0

            status = f"{count} items" if count > 0 else "empty"

            item = QTreeWidgetItem([display, status])

            if d.lower() in known_dirs:
                item.setForeground(0, QColor('#1565c0'))
                item.setForeground(1, QColor('#666666'))
            else:
                item.setForeground(0, QColor('#333333'))
                item.setForeground(1, QColor('#999999'))

            parent_item.addChild(item)

            # Expand top 2 levels, collapse deeper
            if depth < 2:
                item.setExpanded(True)

            # Recurse (limit depth to avoid huge trees)
            if depth < 4:
                self._populate_tree_recursive(
                    full, item, info, known_dirs, known_files, depth + 1
                )

        # Add files
        for f in files:
            full = os.path.join(dir_path, f)

            # File size for status
            try:
                size = os.path.getsize(full)
                if size >= 1_073_741_824:
                    status = f"{size / 1_073_741_824:.1f} GB"
                elif size >= 1_048_576:
                    status = f"{size / 1_048_576:.1f} MB"
                elif size >= 1024:
                    status = f"{size / 1024:.1f} KB"
                else:
                    status = f"{size} B"
            except OSError:
                status = ''

            item = QTreeWidgetItem([f, status])

            if f.lower() in known_files:
                item.setForeground(0, QColor('#2e7d32'))
                item.setForeground(1, QColor('#666666'))
            elif f.endswith(('.pt', '.pth')):
                item.setForeground(0, QColor('#4a148c'))
                item.setForeground(1, QColor('#666666'))
            elif f.endswith('.onnx'):
                item.setForeground(0, QColor('#1565c0'))
                item.setForeground(1, QColor('#666666'))
            elif f.endswith(('.har', '.hef')):
                item.setForeground(0, QColor('#e65100'))
                item.setForeground(1, QColor('#666666'))
            elif f.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                item.setForeground(0, QColor('#888888'))
                item.setForeground(1, QColor('#aaaaaa'))
            else:
                item.setForeground(0, QColor('#555555'))
                item.setForeground(1, QColor('#999999'))

            parent_item.addChild(item)

    # ---- Auto Paths ----

    def _update_path_display(self, info: Dict) -> None:
        """Update the auto-configured paths display."""
        base = info.get('base', '')

        if info.get('calib_found'):
            # Show relative path from base
            rel = os.path.relpath(info['calib_dir'], base)
            self.tab.lblAutoCalibPath.setText(rel)
            self.tab.lblAutoCalibPath.setStyleSheet("color: #2e7d32;")
        else:
            self.tab.lblAutoCalibPath.setText("-")
            self.tab.lblAutoCalibPath.setStyleSheet("color: #999999;")

        onnx_dir = info.get('onnx_dir', '')
        if onnx_dir:
            rel = os.path.relpath(onnx_dir, base)
            self.tab.lblAutoOnnxPath.setText(rel)
            self.tab.lblAutoOnnxPath.setStyleSheet("color: #1565c0;")
        else:
            self.tab.lblAutoOnnxPath.setText("-")
            self.tab.lblAutoOnnxPath.setStyleSheet("color: #999999;")

        hef_dir = info.get('hef_dir', '')
        if hef_dir:
            rel = os.path.relpath(hef_dir, base)
            self.tab.lblAutoHefPath.setText(rel)
            self.tab.lblAutoHefPath.setStyleSheet("color: #e65100;")
        else:
            self.tab.lblAutoHefPath.setText("-")
            self.tab.lblAutoHefPath.setStyleSheet("color: #999999;")

    def _clear_info_labels(self) -> None:
        """Clear info and path labels (tree is kept)."""
        labels = [self.tab.lblClasses, self.tab.lblCalibInfo,
                  self.tab.lblPtInfo, self.tab.lblDataYamlInfo,
                  self.tab.lblAutoCalibPath, self.tab.lblAutoOnnxPath,
                  self.tab.lblAutoHefPath]
        # Optional labels (may not exist in UI yet)
        for name in ('lblOnnxInfo', 'lblHefInfo'):
            if hasattr(self.tab, name):
                labels.append(getattr(self.tab, name))
        for lbl in labels:
            lbl.setText("-")
            lbl.setStyleSheet("color: #999999;")

    # ---- Apply ----

    def _apply_to_tabs(self) -> None:
        """Emit project info to other tabs."""
        if self._project_info and self._project_info.get('valid'):
            self.signals.project_changed.emit(self._project_info)
            self.tab.btnApplyToTabs.setText("Applied!")
            self.tab.btnApplyToTabs.setStyleSheet(
                "background-color: #2e7d32; color: white; font-size: 14px; "
                "font-weight: bold; padding: 10px 30px; border-radius: 5px;"
            )
            # Reset button text after 2 seconds
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(2000, self._reset_apply_button)

    def _reset_apply_button(self) -> None:
        """Reset the Apply button to default state."""
        self.tab.btnApplyToTabs.setText("Apply to All Tabs")
        self.tab.btnApplyToTabs.setStyleSheet(
            "background-color: #1565c0; color: white; font-size: 14px; "
            "font-weight: bold; padding: 10px 30px; border-radius: 5px;"
        )
