"""
Style sheets for Hailo-Compiler-UI.
Modern minimal LIGHT theme with blue and orange accents.
"""

# Light Color palette
COLORS = {
    'background': '#f0f0f0',
    'surface': '#ffffff',
    'surface_light': '#f8f8f8',
    'border': '#cccccc',
    'border_light': '#dddddd',
    'text': '#000000',  # Pure black text
    'text_secondary': '#333333',  # Dark gray secondary text
    'primary': '#1565c0',  # Darker blue
    'primary_light': '#1976d2',
    'primary_dark': '#0d47a1',
    'accent': '#e65100',  # Darker orange
    'accent_light': '#ff6f00',
    'success': '#2e7d32',  # Darker green
    'warning': '#ef6c00',  # Darker orange
    'error': '#c62828',  # Darker red
}

LIGHT_THEME = f"""
/* Main Window */
QMainWindow {{
    background-color: {COLORS['background']};
}}

QWidget {{
    background-color: {COLORS['background']};
    color: {COLORS['text']};
    font-family: 'Segoe UI', 'Arial', sans-serif;
    font-size: 12pt;
}}

/* Group Box (Cards) */
QGroupBox {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    margin-top: 20px;
    padding: 18px;
    padding-top: 28px;
    font-weight: bold;
    font-size: 13pt;
}}

QGroupBox::title {{
    color: {COLORS['primary']};
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 16px;
    top: 4px;
    padding: 0 8px;
    background-color: {COLORS['surface']};
}}

/* Buttons */
QPushButton {{
    background-color: {COLORS['primary']};
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 12px 24px;
    font-weight: bold;
    font-size: 12pt;
    min-width: 110px;
}}

QPushButton:hover {{
    background-color: {COLORS['primary_light']};
}}

QPushButton:pressed {{
    background-color: {COLORS['primary_dark']};
}}

QPushButton:disabled {{
    background-color: {COLORS['border']};
    color: {COLORS['text_secondary']};
}}

/* Accent Button */
QPushButton[accent="true"] {{
    background-color: {COLORS['accent']};
}}

QPushButton[accent="true"]:hover {{
    background-color: {COLORS['accent_light']};
}}

/* Browse Button */
QPushButton[browse="true"] {{
    min-width: 80px;
    padding: 8px 16px;
    background-color: {COLORS['surface']};
    color: {COLORS['primary']};
    border: 1px solid {COLORS['primary']};
}}

QPushButton[browse="true"]:hover {{
    background-color: {COLORS['primary']};
    color: #ffffff;
}}

/* Line Edit */
QLineEdit {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 5px;
    padding: 10px 14px;
    font-size: 12pt;
    color: {COLORS['text']};
    selection-background-color: {COLORS['primary']};
}}

QLineEdit:focus {{
    border-color: {COLORS['primary']};
    border-width: 2px;
}}

QLineEdit:disabled {{
    background-color: {COLORS['surface_light']};
    color: {COLORS['text_secondary']};
}}

/* Spin Box */
QSpinBox {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 5px;
    padding: 8px 12px;
    font-size: 12pt;
    color: {COLORS['text']};
}}

QSpinBox:focus {{
    border-color: {COLORS['primary']};
}}

QSpinBox::up-button, QSpinBox::down-button {{
    background-color: {COLORS['surface_light']};
    border: none;
    width: 20px;
}}

QSpinBox::up-button:hover, QSpinBox::down-button:hover {{
    background-color: {COLORS['primary_light']};
}}

/* Combo Box */
QComboBox {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 5px;
    padding: 10px 14px;
    font-size: 12pt;
    color: {COLORS['text']};
    min-width: 120px;
}}

QComboBox:focus {{
    border-color: {COLORS['primary']};
}}

QComboBox::drop-down {{
    border: none;
    width: 30px;
}}

QComboBox::down-arrow {{
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 5px solid {COLORS['text_secondary']};
    margin-right: 10px;
}}

QComboBox QAbstractItemView {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    selection-background-color: {COLORS['primary']};
    selection-color: #ffffff;
    color: {COLORS['text']};
}}

/* Progress Bar */
QProgressBar {{
    background-color: {COLORS['border_light']};
    border: none;
    border-radius: 4px;
    text-align: center;
    color: {COLORS['text']};
    height: 24px;
}}

QProgressBar::chunk {{
    background-color: {COLORS['primary']};
    border-radius: 4px;
}}

/* Progress Bar Error State */
QProgressBar[error="true"]::chunk {{
    background-color: {COLORS['error']};
}}

/* Text Edit (Log Panel) */
QTextEdit {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    padding: 10px;
    color: {COLORS['text']};
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 11pt;
}}

QTextEdit:focus {{
    border-color: {COLORS['primary']};
}}

/* Scroll Bar */
QScrollBar:vertical {{
    background-color: {COLORS['surface_light']};
    width: 12px;
    border-radius: 6px;
}}

QScrollBar::handle:vertical {{
    background-color: {COLORS['border']};
    border-radius: 6px;
    min-height: 30px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {COLORS['text_secondary']};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

QScrollBar:horizontal {{
    background-color: {COLORS['surface_light']};
    height: 12px;
    border-radius: 6px;
}}

QScrollBar::handle:horizontal {{
    background-color: {COLORS['border']};
    border-radius: 6px;
    min-width: 30px;
}}

/* Labels */
QLabel {{
    color: {COLORS['text']};
    background-color: transparent;
}}

QLabel[heading="true"] {{
    font-size: 18pt;
    font-weight: bold;
    color: {COLORS['primary']};
}}

QLabel[status="success"] {{
    color: {COLORS['success']};
}}

QLabel[status="warning"] {{
    color: {COLORS['warning']};
}}

QLabel[status="error"] {{
    color: {COLORS['error']};
}}

/* Status Bar */
QStatusBar {{
    background-color: {COLORS['surface']};
    border-top: 1px solid {COLORS['border']};
    color: {COLORS['text_secondary']};
}}

/* Menu Bar */
QMenuBar {{
    background-color: {COLORS['surface']};
    color: {COLORS['text']};
    border-bottom: 1px solid {COLORS['border']};
}}

QMenuBar::item:selected {{
    background-color: {COLORS['primary']};
    color: #ffffff;
}}

QMenu {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    color: {COLORS['text']};
}}

QMenu::item:selected {{
    background-color: {COLORS['primary']};
    color: #ffffff;
}}

/* Tool Tip */
QToolTip {{
    background-color: {COLORS['text']};
    border: none;
    color: #ffffff;
    padding: 4px 8px;
    border-radius: 4px;
}}

/* Message Box */
QMessageBox {{
    background-color: {COLORS['surface']};
}}

QMessageBox QLabel {{
    color: {COLORS['text']};
}}

/* Frame */
QFrame {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
}}
"""


def get_theme() -> str:
    """Get the current theme stylesheet."""
    return LIGHT_THEME


def get_color(name: str) -> str:
    """Get a color by name."""
    return COLORS.get(name, '#212121')
