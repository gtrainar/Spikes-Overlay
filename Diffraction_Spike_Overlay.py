##############################################
# Diffraction Spike Overlay Tool for Siril
#
# (c) G. Trainar 2026
# SPDX-License-Identifier: MIT License
##############################################

# Version 1.1.0

"""
Diffraction Spike Overlay Tool for Siril

Purpose:
  Add diffraction spikes to the current image loaded in Siril.
  Provides a live preview and allows saving the modified image under a new name and a mask with alpha channel (PNG).
"""

import sys
import time
import os
import logging
import base64
import shlex
from typing import List, Tuple, Optional, Dict

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.draw import disk, polygon
from PIL import Image

# PyQt imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel,
    QSlider, QComboBox, QPushButton,
    QFormLayout, QStatusBar, QMessageBox,
    QFileDialog, QSizePolicy,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QFrame, QColorDialog, QDialog, QTextEdit, QProgressBar, 
    QGroupBox, QScrollArea
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QPointF
from PyQt6.QtGui import QPixmap, QImage, QColor, QPainter, QIcon, QPen

# Siril imports
try:
    import sirilpy as s
    from sirilpy import NoImageError, SirilError
    SIRIL = s.SirilInterface()
    if not SIRIL.connect():
        logging.error("Failed to connect to Siril. Is Siril running?")
        sys.exit(1)
except Exception as e:
    logging.exception(f"Error initializing Siril interface: {e}")
    sys.exit(1)

# Constants
VERSION = "1.1.0"

DEFAULT_CONFIG = {
    'FWHM_MIN': 20,
    'FWHM_MAX': 300,
    'FWHM_MAX_LIMIT': 1000,
    'SPIKE_THICKNESS': 4,
    'SPIKE_COUNT': 4,          # 4, 6 or 8
    'BLUR_SIGMA': 0.1,         # 0.5 σ
    'ROTATION_ANGLE': 0,
    'SPIKE_LENGTH_PCT': 80,
    'TRANSPARENCY': 0, 
}

CCHANGELOG = """
Version 1.1.0
    - GUI Update
    - Live Preview of the spiked-stars mask

Version 1.0.0 (Initial Release) - Feb 2026
*Features:
    -Diffraction Spike Generation: Add synthetic diffraction spikes to astronomical images
    -Real-time Preview: Live preview of spike effects with adjustable parameters
    -Star Detection: Automatic detection of stars in the image for spike placement
    -Customizable Parameters:
    -FWHM filtering (min/max)
    -Spike length and thickness
    -Number of spikes (4, 6, or 8)
    -Spike pattern shapes (Radial, Cross, X)
    -Rotation angle
    -Blur sigma for realistic effects
    -Color Customization: Choose spike color with a color picker
    -Image Formats: Support for FITS, TIFF, PNG, and JPEG formats

*Compatibility:
    -Siril Integration: Full compatibility with Siril's image processing pipeline
    -32-bit TIFF Support: Native handling of 32-bit floating point images

*Usage Notes:
    -The mask PNG can be opened in GIMP and used as a layer with transparency
    -The composed TIFF contains the final image with spikes applied
    -Both files are saved in the same directory as the original

*Known Limitations:
    -Requires Siril to be running with an image loaded
    -Works on RGB stretched images only
    -Large images can take a long time to process or cause the script to hang when it starts
"""

DARK_STYLESHEET = """
QWidget { background-color: #2b2b2b; color: #e0e0e0; font-size: 13pt; }
QToolTip { background-color: #333333; color: #ffffff; border: 1px solid #88aaff; }
QGroupBox {
    border: 1px solid #444444;
    margin-top: 10px;
    font-weight: bold;
    border-radius: 4px;
    padding-top: 14px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5x;
    color: #88aaff;
}
QLabel { color: #cccccc; }
QRadioButton, QCheckBox { color: #cccccc; spacing: 5px; }
QRadioButton::indicator, QCheckBox::indicator {
    width: 14px; height: 14px;
    border: 1px solid #666666;
    background: #3c3c3c;
    border-radius: 7px;
}
QCheckBox::indicator { border-radius: 3px; }
QRadioButton::indicator:checked {
    background: qradialgradient(cx:0.5, cy:0.5, radius: 0.4,
        fx:0.5, fy:0.5, stop:0 #ffffff, stop:1 #285299);
    border: 1px solid #88aaff;
}
QCheckBox::indicator:checked {
    background-color: #285299;
    border: 1px solid #88aaff;
}
QSlider::groove:horizontal {
    background: #444444;
    height: 6px;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background-color: #aaaaaa;
    width: 14px; height: 14px;
    margin: -4px 0;
    border-radius: 7px;
    border: 1px solid #555555;
}
QSlider::handle:horizontal:hover { background-color: #ffffff; }
QPushButton {
    background-color: #444444;
    color: #dddddd;
    border: 1px solid #666666;
    border-radius: 4px;
    padding: 8px 16px;
    font-weight: bold;
}
QPushButton:hover { background-color: #555555; border-color: #777777; }
QPushButton#ProcessButton {
    background-color: #285299;
    border: 1px solid #1e3f7a;
    font-size: 12pt;
    padding: 12px;
}
QPushButton#PrepButton:hover { background-color: #355ea1; }
"""

class ImageProcessingError(Exception):
    """Custom exception for image processing errors"""
    pass

# Helper functions
def wait_for_lock(siril: s.SirilInterface, retries: int = 5, delay: float = 0.5) -> None:
    """Yield a lock on Siril's processing thread."""
    for _ in range(retries):
        try:
            return siril.image_lock()
        except s.ProcessingThreadBusyError:
            time.sleep(delay)
    raise s.ProcessingThreadBusyError(
        "Siril processing thread is still busy after retries"
    )

def array_to_qimage(arr: np.ndarray) -> QImage:
    """Return a QImage from a numpy array."""
    if arr.ndim == 2:  # grayscale
        return QImage(arr.tobytes(), arr.shape[1], arr.shape[0],
                      QImage.Format.Format_Grayscale8)
    else:  # RGB / RGBA
        fmt = QImage.Format.Format_RGBA8888 if arr.shape[2] == 4 else QImage.Format.Format_RGB888
        return QImage(arr.tobytes(), arr.shape[1], arr.shape[0],
                      arr.strides[0], fmt)

def calculate_spike_heights(fwhm: float, spike_length_pct: int, spike_count: int) -> List[float]:
    """
    Calculate heights for each spike based on configuration.

    For 8 spikes: alternate long/short (short = 0.7 × long).
    For other counts: all spikes have the same height.
    """
    base_height = fwhm * spike_length_pct / 100.0
    if spike_count == 8:
        return [base_height if i % 2 == 0 else base_height * 0.7
                for i in range(8)]
    return [base_height] * spike_count

def compute_spike_angles(spike_count: int, pattern: str, rotation_angle: float) -> List[float]:
    """Calculate spike angles based on pattern and rotation."""
    base = 2 * np.pi / spike_count
    angles = [i * base for i in range(spike_count)]

    if pattern == "Cross":
        angles = [a + np.pi/2 for a in angles]
    elif pattern == "X":
        angles = [a + np.pi/4 for a in angles] + \
                 [a + 3*np.pi/4 for a in angles]

    return [(a + np.radians(rotation_angle)) % (2 * np.pi) for a in angles]

def draw_spike_triangle(mask: np.ndarray, x: float, y: float,
                        angle: float, height: float, base: float) -> None:
    """Draw a single spike triangle on the mask."""
    tip_x = x + height * np.cos(angle)
    tip_y = y + height * np.sin(angle)

    perp1 = angle + np.pi/2
    perp2 = angle - np.pi/2

    left_x = x + (base/2) * np.cos(perp1)
    left_y = y + (base/2) * np.sin(perp1)
    right_x = x + (base/2) * np.cos(perp2)
    right_y = y + (base/2) * np.sin(perp2)

    rr, cc = polygon([tip_y, left_y, right_y],
                     [tip_x, left_x, right_x], mask.shape)
    valid = (rr >= 0) & (rr < mask.shape[0]) & \
            (cc >= 0) & (cc < mask.shape[1])
    rr, cc = rr[valid], cc[valid]
    mask[rr, cc] = 1.0

# Worker threads
class ImageLoader(QThread):
    """Background thread to load image from Siril."""
    finished = pyqtSignal(object)  # emits the fetched ffit

    def run(self):
        try:
            with wait_for_lock(SIRIL):
                ffit = SIRIL.get_image()
            self.finished.emit(ffit)
        except Exception as e:
            logging.exception(f"Error loading image from Siril: {e}")
            self.finished.emit(None)

class PreviewWorker(QThread):
    """Background thread to generate preview with spikes."""
    finished = pyqtSignal(QPixmap, int)

    def __init__(self, image: np.ndarray, params: Dict, parent=None):
        super().__init__(parent)
        self.image = image.copy()
        self.params = params

    def run(self):
        try:
            mask = np.zeros((self.image.shape[0], self.image.shape[1]), dtype=np.float32)
            angles = compute_spike_angles(
                self.params["spike_count"],
                self.params["pattern"],
                self.params["rotation_angle"]
            )

            star_count = 0
            for x, y, fwhm in self.params["stars_list"]:
                if not (self.params["fwhm_min"] <= fwhm <= self.params["fwhm_max"]):
                    continue
                star_count += 1

                height = fwhm * self.params["spike_length"] / 100.0
                base = max(self.params["spike_thickness"], fwhm / 10.0)

                base_factor = max(self.params["spike_length"] / 100.0, 0.2)
                base *= base_factor

                heights = calculate_spike_heights(fwhm, self.params["spike_length"], len(angles))

                for a, h in zip(angles, heights):
                    draw_spike_triangle(mask, x, y, a, h, base)

                # Apply blur around the star
                y0 = max(0, int(y) - 20)
                y1 = min(self.image.shape[0], int(y) + 21)
                x0 = max(0, int(x) - 20)
                x1 = min(self.image.shape[1], int(x) + 21)
                self.image[y0:y1, x0:x1] = gaussian_filter(
                    self.image[y0:y1, x0:x1], sigma=self.params["blur_sigma"]
                )

            # ----- Darken the base image according to transparency -----
            alpha = self.params.get("transparency", 0.0)
            if alpha > 0:
                # Multiply by (1 – alpha) to darken everything
                self.image *= (1.0 - alpha)

            # Convert image to float if necessary
            if self.image.dtype.kind != 'f':
                self.image = self.image.astype(np.float32)

            # Add colored spikes
            color_norm = np.array(self.params["spike_color"], dtype=np.float32) / 255.0
            self.image[..., 0] += mask * color_norm[0]
            self.image[..., 1] += mask * color_norm[1]
            self.image[..., 2] += mask * color_norm[2]

            # Clip to valid range
            np.clip(self.image, 0, 1, out=self.image)

            # Create pixmap
            img = np.flipud(self.image)
            if img.dtype.kind != 'u':
                img = np.clip(img, 0, 1)
                img = (img * 255).astype(np.uint8)

            qimg = array_to_qimage(img)
            pixmap = QPixmap.fromImage(qimg)

            # Scale to view size
            target_size = self.parent().view.size()
            scaled_pixmap = pixmap.scaled(
                target_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

            self.finished.emit(scaled_pixmap, star_count)

        except Exception as e:
            logging.error(f"Error generating preview: {e}")
            self.finished.emit(QPixmap(), 0)

# UI Components
class HelpDialog(QDialog):
    """Scrollable help dialog."""
    def __init__(self, title: str, text: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)

        txt = QTextEdit()
        txt.setReadOnly(True)
        txt.setPlainText(text)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)

        lay = QVBoxLayout()
        lay.addWidget(txt)
        lay.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignRight)
        self.setLayout(lay)

        # Start with a reasonable size; user can still resize
        self.resize(800, 600)

class SpikeWindow(QMainWindow):
    """Main application window for diffraction spike generation."""
    def __init__(self, preloaded_ffit=None):
        super().__init__()
        self.setWindowTitle(f"Diffraction Spikes Generator {VERSION}")
        self.setStyleSheet(DARK_STYLESHEET)
        self.resize(1200, 800)
        self.showing_original = False 
        self._prev_transform = None 

        # Image data
        self.image_orig = None
        self.header = None
        self.current_ffit = preloaded_ffit
        self.filename = ""
        self.stars_list = []

        # Spike appearance - RGB only
        self.spike_color = (255, 255, 255)  # Default white RGB

        # Preview timer
        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self.update_preview)

        # Resize debounce timer
        self.resize_timer = QTimer(self)
        self.resize_timer.setSingleShot(True)
        self.resize_timer.timeout.connect(self.update_preview)

        # UI
        self.init_ui()

        # Progress bar overlay (hidden until we start loading)
        self.loading_widget = QWidget(self)
        self.loading_widget.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.loading_widget.setStyleSheet("background-color: rgba(255, 255, 255, 200);")
        self.loading_layout = QVBoxLayout(self.loading_widget)
        self.loading_bar = QProgressBar()
        self.loading_bar.setRange(0, 0)  # indeterminate
        self.loading_label = QLabel("Loading image…")
        self.loading_layout.addWidget(self.loading_label, alignment=Qt.AlignmentFlag.AlignCenter)
        self.loading_layout.addWidget(self.loading_bar, alignment=Qt.AlignmentFlag.AlignCenter)
        self.loading_widget.hide()

        # Start loading the image in a background thread
        if preloaded_ffit is None:
            self._start_image_loader()
        else:
            self._set_image_from_ffit(preloaded_ffit)

    def init_ui(self):
        """Initialize the user interface."""
        # -----------------------------------------------------------------
        #  MAIN SPLITTER
        # -----------------------------------------------------------------
        from PyQt6.QtWidgets import QSplitter, QSpacerItem

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.setCentralWidget(splitter)
        splitter.setContentsMargins(10, 0, 0, 0)

        # -----------------------------------------------------------------
        #  LEFT PANE – ALL CONTROLS IN ONE FIXED WIDGET
        # -----------------------------------------------------------------
        left_widget = QWidget()
        left_widget.setFixedWidth(320)                       # fixed width
        left_widget.setSizePolicy(QSizePolicy.Policy.Fixed,
                                  QSizePolicy.Policy.Expanding)

        # The left pane will contain only one widget – the whole controls block
        controls_container = QWidget()
        controls_layout = QVBoxLayout(controls_container)
        controls_layout.setContentsMargins(0, 0, 0, 0)

        # -----------------------------------------------------------------
        #  CONTROLS – SLIDERS & GROUP BOXES
        # -----------------------------------------------------------------

        # Helper to create sliders (unchanged)
        def make_slider(label: str, min_: int, max_: int, init_: int,
                        unit: str, display_scale: float = 1.0) -> Tuple[QWidget, QSlider]:
            sld = QSlider(Qt.Orientation.Horizontal)
            sld.setRange(min_, max_)
            sld.setValue(init_)

            val_lbl = QLabel(str(init_))
            val_lbl.setParent(controls_container)   # use the controls container as parent

            def update_val(val):
                lbl_text = f"{val / display_scale:.1f}"
                val_lbl.setText(lbl_text)
            sld.valueChanged.connect(update_val)
            sld.valueChanged.connect(self.schedule_preview_update)

            w = QWidget()
            lay = QHBoxLayout(w)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(sld)
            lay.addWidget(val_lbl)
            lay.addWidget(QLabel(unit))
            return w, sld

        # Slider widgets
        self.fwhm_min_lbl, self.fwhm_min_slider = make_slider(
            "FWHM Min", 5, 40, DEFAULT_CONFIG['FWHM_MIN'], "px")
        self.fwhm_max_lbl, self.fwhm_max_slider = make_slider(
            "FWHM Max", 5, DEFAULT_CONFIG['FWHM_MAX_LIMIT'], DEFAULT_CONFIG['FWHM_MAX'], "px")
        self.length_lbl, self.spike_length_slider = make_slider(
            "Spike Length", 0, 100, DEFAULT_CONFIG['SPIKE_LENGTH_PCT'], "%")
        self.thickness_lbl, self.spike_thickness_slider = make_slider(
            "Spike Thickness", 1, 10, DEFAULT_CONFIG['SPIKE_THICKNESS'], "px")
        self.blur_lbl, self.blur_sigma_slider = make_slider(
            "Blur Sigma", 0, 30, int(DEFAULT_CONFIG['BLUR_SIGMA'] * 10), "σ", display_scale=10.0)
        self.rotation_lbl, self.rotation_slider = make_slider(
            "Rotation Angle", 0, 360, DEFAULT_CONFIG['ROTATION_ANGLE'], "°")

        # --- Transparency slider ---------------------------------------
        self.transparency_lbl, self.transparency_slider = make_slider(
            "Transparency", 0, 100, DEFAULT_CONFIG['TRANSPARENCY'], "%",
            display_scale=1.0)

        # Colour display box
        self.color_display = QLabel()
        self.color_display.setFixedSize(20, 20)
        self.color_display.setStyleSheet(
            "background-color: rgb(255,255,255); border: 1px solid black;"
        )

        # Color-picker button
        choose_btn = QPushButton("Choose…")
        choose_btn.clicked.connect(self.choose_color)

        color_row_widget = QWidget()
        row_lay = QHBoxLayout(color_row_widget)
        row_lay.setContentsMargins(0, 0, 0, 0)   # no extra spacing
        row_lay.addWidget(self.color_display)
        row_lay.addWidget(choose_btn)

        # Combo boxes
        self.spike_count_combo = QComboBox()
        self.spike_count_combo.addItems(["4", "6", "8"])
        self.spike_count_combo.setCurrentText(str(DEFAULT_CONFIG['SPIKE_COUNT']))
        self.spike_count_combo.currentIndexChanged.connect(self.schedule_preview_update)

        self.pattern_combo = QComboBox()
        self.pattern_combo.addItems(["Radial", "Cross", "X"])
        self.pattern_combo.setCurrentText("Radial")
        self.pattern_combo.currentIndexChanged.connect(self.schedule_preview_update)

        # Star count label
        self.star_count_label = QLabel("0")

        # Buttons
        default_btn = QPushButton("Default")
        default_btn.clicked.connect(self.reset_to_defaults)

        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save_spiked_image)

        help_btn = QPushButton("Help")
        help_btn.clicked.connect(self.show_help)

        btn_row_layout = QHBoxLayout()
        btn_row_layout.addWidget(default_btn)
        btn_row_layout.addWidget(save_btn)
        btn_row_layout.addWidget(help_btn)

        # -----------------------------------------------------------------
        #  ADD GROUP BOXES TO CONTROLS LAYOUT
        # -----------------------------------------------------------------

        # Star Selection group
        star_group = QGroupBox("Stars Selection")
        star_layout = QFormLayout()
        star_layout.addRow("FWHM Min:", self.fwhm_min_lbl)
        star_layout.addRow("FWHM Max:", self.fwhm_max_lbl)
        star_layout.addRow("Detected Stars:", self.star_count_label)
        star_group.setLayout(star_layout)
        star_group.setMinimumWidth(280)
        star_group.setFixedHeight(star_group.sizeHint().height())

        # Spike Setting group
        spike_group = QGroupBox("Spike Settings")
        spike_layout = QFormLayout()
        spike_layout.addRow("Spike Length:", self.length_lbl)
        spike_layout.addRow("Spike Thickness:", self.thickness_lbl)
        spike_layout.addRow("Number of Spikes:", self.spike_count_combo)
        spike_layout.addRow("Blur Sigma:", self.blur_lbl)
        spike_layout.addRow("Rotation Angle:", self.rotation_lbl)
        spike_layout.addRow("Pattern Shape:", self.pattern_combo)
        spike_layout.addRow("Spike Color:", color_row_widget)
        spike_group.setLayout(spike_layout)
        spike_group.setMinimumWidth(280)
        spike_group.setFixedHeight(spike_group.sizeHint().height())

        # Mask Setting group
        mask_group = QGroupBox("Mask Setting")
        mask_layout = QFormLayout()
        mask_layout.addRow("Transparency:", self.transparency_lbl)
        mask_group.setLayout(mask_layout)
        mask_group.setMinimumWidth(280)
        mask_group.setFixedHeight(mask_group.sizeHint().height())

        # Add the groups and buttons to the controls layout
        controls_layout.addWidget(star_group)
        controls_layout.addWidget(spike_group)
        controls_layout.addWidget(mask_group)

        # Fixed spacer between the last group and the buttons
        controls_layout.addSpacerItem(QSpacerItem(0, 10,
                                                QSizePolicy.Policy.Minimum,
                                                QSizePolicy.Policy.Fixed))

        controls_layout.addLayout(btn_row_layout)

        # Lock the whole control block to its size‑hint
        controls_container.setFixedSize(320,
                                       controls_container.sizeHint().height())

        # Place the control block inside the left pane
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        left_layout.addSpacerItem(QSpacerItem(0, 50,
                                            QSizePolicy.Policy.Minimum,
                                            QSizePolicy.Policy.Fixed))
        left_layout.addWidget(controls_container)

        splitter.addWidget(left_widget)          # add the left pane first

        # -----------------------------------------------------------------
        #  RIGHT PANE – PREVIEW
        # -----------------------------------------------------------------
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)

        # --- Top button bar ------------------------------------------
        btn_bar = QHBoxLayout()
        zoom_in_btn = QPushButton("Zoom In")
        zoom_out_btn = QPushButton("Zoom Out")
        zoom_1to1_btn = QPushButton("1:1")
        fit_view_btn = QPushButton("Fit")

        # Fixed size for the zoom buttons
        for btn in (zoom_in_btn, zoom_out_btn, zoom_1to1_btn, fit_view_btn):
            btn.setFixedSize(100, 30)

        zoom_in_btn.clicked.connect(self.zoom_in)
        zoom_out_btn.clicked.connect(self.zoom_out)
        zoom_1to1_btn.clicked.connect(self.zoom_1to1)
        fit_view_btn.clicked.connect(self.fit_view)

        btn_bar.addWidget(zoom_in_btn)
        btn_bar.addWidget(zoom_out_btn)
        btn_bar.addWidget(zoom_1to1_btn)
        btn_bar.addWidget(fit_view_btn)
        btn_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)   # keep centered
        preview_layout.addLayout(btn_bar)

        # --- Graphics view -------------------------------------------
        self.view = QGraphicsView()
        self.scene = QGraphicsScene(self.view)
        self.view.setScene(self.scene)

        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        preview_layout.addWidget(self.view)

        self._prev_transform = self.view.transform()

        # Overlay label (kept as a child of the view's viewport)
        self.overlay_label = QLabel("Original", parent=self.view.viewport())
        self.overlay_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.overlay_label.setStyleSheet(
            "background-color: gray; border-radius: 8px;"
            "color: white; font-weight: bold; font-size: 20pt;"
        )
        self.overlay_label.setFixedSize(100, 35)
        self.overlay_label.hide()

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        splitter.addWidget(preview_widget)      # add the right pane
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)


    def zoom_in(self):
        """Zoom in by 20%."""
        self.view.scale(1.2, 1.2)

    def zoom_out(self):
        """Zoom out by 20%."""
        self.view.scale(1 / 1.2, 1 / 1.2)

    def zoom_1to1(self):
        """Reset to 100% (no transform)."""
        self.view.resetTransform()

    def fit_view(self):
        """Fit the image inside the view while keeping aspect ratio."""
        if not self.scene.items():
            return
        rect = self.scene.itemsBoundingRect()
        self.view.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)

    def choose_color(self):
        """Open a QColorDialog, store the chosen color and update preview."""
        color = QColorDialog.getColor(QColor(*self.spike_color), self, "Select Spike Colour")
        if color.isValid():
            rgb = color.getRgb()[:3]
            self.spike_color = tuple(rgb)
            # Update the display box
            self.color_display.setStyleSheet(
                f"background-color: rgb({rgb[0]},{rgb[1]},{rgb[2]}); border: 1px solid black;"
            )
            self.schedule_preview_update()

    def _set_image_from_ffit(self, ffit):
        """Set the image from FITS data with proper scaling for display."""
        img = ffit.data

        # Handle different image formats
        if img.ndim == 3 and img.shape[0] in (1, 3, 4):
            # Move channel axis to last position
            img = np.moveaxis(img, 0, -1)

        # Convert to displayable format
        if img.dtype.kind == 'f':  # Floating point data
            # Normalize to 0-1 range if needed
            img_min = np.min(img)
            img_max = np.max(img)

            # If image is very dark, apply auto-scaling
            if img_max - img_min < 0.1:
                # Use percentiles to find better range
                lower = np.percentile(img, 1)
                upper = np.percentile(img, 99.5)
                img = (img - lower) / (upper - lower + 1e-6)
            else:
                img = (img - img_min) / (img_max - img_min + 1e-6)

            # Clip to valid range
            img = np.clip(img, 0, 1)

        elif img.dtype.kind == 'u':  # Unsigned integer
            if img.dtype == np.uint16:
                img = img / 65535.0
            elif img.dtype == np.uint8:
                img = img / 255.0
        elif img.dtype.kind == 'i':  # Signed integer
            if img.dtype in (np.int16, np.int32):
                img = (img - np.min(img)) / (np.max(img) - np.min(img))

        # Convert to RGB if grayscale
        if len(img.shape) == 2:
            img = np.stack([img]*3, axis=-1)

        # Ensure we have 3 channels
        if img.shape[2] == 4:  # RGBA -> RGB
            img = img[..., :3]

        self.image_orig = img
        self.header = getattr(ffit, "header", None)
        self.current_ffit = ffit
        
        try:
            self.filename = SIRIL.get_image_filename()
        except NoImageError:
            SIRIL.log("No image loaded – cannot get filename",s.LogColor.RED)
            self.filename = ""

        self.stars_list = self.get_stars_from_siril()
        self.update_preview()

    def _start_image_loader(self):
        """Launch the worker thread that fetches the image from Siril."""
        self.loading_widget.show()
        self.loader_thread = ImageLoader(parent=self)
        self.loader_thread.finished.connect(self._on_image_loaded)
        self.loader_thread.start()

    def _on_image_loaded(self, ffit):
        """Called when the background loader finishes."""
        self.loading_widget.hide()
        if ffit is None:
            QMessageBox.critical(self, "Error",
                                 "Could not fetch image from Siril.")
            return
        self._set_image_from_ffit(ffit)

    def schedule_preview_update(self):
        self.preview_timer.start(200)

    def update_preview(self):
        """Update the preview with current settings."""
        if self.image_orig is None:
            return

        # Stop old worker
        if hasattr(self, "preview_thread") and self.preview_thread.isRunning():
            self.preview_thread.terminate()
            self.preview_thread.wait()

        params = self._get_preview_params()
        self.preview_thread = PreviewWorker(self.image_orig.copy(), params, parent=self)
        self.preview_thread.finished.connect(self._set_preview_pixmap)
        self.preview_thread.start()

    def _set_preview_pixmap(self, pixmap: QPixmap, star_count: int):
        """Display the preview inside the QGraphicsView."""
        # Preserve current zoom/rotation
        previous_transform = self.view.transform()

        # Clear the scene before adding the new image
        self.scene.clear()
        item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(item)
        self.scene.setSceneRect(item.boundingRect())

        # Restore the previous zoom/rotation instead of resetting
        self.view.setTransform(previous_transform)

        self.star_count_label.setText(f"{star_count}")

    def get_stars_from_siril(self) -> List[Tuple[float, float, float]]:
        """
        Retrieve the list of stars from Siril.

        If no stars are found or an error occurs, a message is logged
        to the Siril console and an empty list is returned.
        """
        try:
            # Prefer channel‑specific star detection; fall back otherwise
            stars = SIRIL.get_image_stars(channel=1)
        except Exception:
            try:
                stars = SIRIL.get_image_stars()
            except Exception as e:
                logging.error(f"Failed to retrieve stars: {e}")
                logging.error("No stars detected – spike generation skipped.")
                return []

        result = []
        for star in stars:
            x = getattr(star, "xpos", getattr(star, "x", None))
            y = getattr(star, "ypos", getattr(star, "y", None))
            if x is None or y is None:
                continue
            fwhm_x = getattr(star, "fwhmx", getattr(star, "fwhm_x", None))
            fwhm_y = getattr(star, "fwhmy", getattr(star, "fwhm_y", None))
            fwhm = getattr(star, "fwhm", None)
            if fwhm_x is not None and fwhm_y is not None:
                val = (float(fwhm_x) + float(fwhm_y)) / 2.0
            elif fwhm is not None:
                val = float(fwhm)
            else:
                val = 5.0
            result.append((float(x), float(y), val))

        if self.image_orig is not None:
            h = self.image_orig.shape[0]
            self.stars_list = [(x, h - 1 - y, fwhm) for (x, y, fwhm) in result]
        else:
            self.stars_list = result

        if not self.stars_list:
            logging.error("No stars detected – spike generation will be skipped.")
        
        return self.stars_list

    def reset_to_defaults(self):
        """Reset all controls to default values."""
        self.fwhm_min_slider.setValue(DEFAULT_CONFIG['FWHM_MIN'])
        # Reset max slider to default maximum
        self.fwhm_max_slider.setRange(1, DEFAULT_CONFIG['FWHM_MAX'])
        self.fwhm_max_slider.setValue(DEFAULT_CONFIG['FWHM_MAX'])

        self.spike_length_slider.setValue(DEFAULT_CONFIG['SPIKE_LENGTH_PCT'])
        self.spike_thickness_slider.setValue(DEFAULT_CONFIG['SPIKE_THICKNESS'])
        self.spike_count_combo.setCurrentText(str(DEFAULT_CONFIG['SPIKE_COUNT']))
        self.blur_sigma_slider.setValue(int(DEFAULT_CONFIG['BLUR_SIGMA'] * 10))
        self.rotation_slider.setValue(DEFAULT_CONFIG['ROTATION_ANGLE'])
        self.pattern_combo.setCurrentText("Radial")
        self.transparency_slider.setValue(DEFAULT_CONFIG['TRANSPARENCY'])

        # Reset color to default white
        self.spike_color = (255, 255, 255)
        self.color_display.setStyleSheet(
            "background-color: rgb(255,255,255); border: 1px solid black;"
        )

        if self.current_ffit is not None:
            fwhm_min = self.fwhm_min_slider.value()
            fwhm_max = self.fwhm_max_slider.value()
            self.fwhm_max_slider.blockSignals(True)
            self._set_image_from_ffit(self.current_ffit)
            self.fwhm_max_slider.blockSignals(False)
            self.fwhm_min_slider.setValue(fwhm_min)

        self.update_preview()

    def _get_preview_params(self) -> Dict:
        """Get current parameters for preview generation."""
        return {
            "fwhm_min": self.fwhm_min_slider.value(),
            "fwhm_max": self.fwhm_max_slider.value(),
            "spike_length": self.spike_length_slider.value(),
            "spike_thickness": self.spike_thickness_slider.value(),
            "spike_count": int(self.spike_count_combo.currentText()),
            "blur_sigma": self.blur_sigma_slider.value() / 10.0,
            "rotation_angle": self.rotation_slider.value(),
            "pattern": self.pattern_combo.currentText(),
            "stars_list": self.stars_list,
            "spike_color": self.spike_color,
            "transparency": self.transparency_slider.value() / 100.0,   # new entry
        }

    def _apply_spikes(self, image: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Apply diffraction spikes to the provided image.

        Returns:
            Tuple of (processed_image, star_count)
        """
        params = self._get_preview_params()

        if not params["stars_list"]:
            # Nothing to do – log and return
            SIRIL.log("No stars detected; skipping spike generation.", s.LogColor.RED)
            return image, 0

        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        angles = compute_spike_angles(
            params["spike_count"],
            params["pattern"],
            params["rotation_angle"]
        )

        star_count = 0
        for x, y, fwhm in params["stars_list"]:
            if not (params["fwhm_min"] <= fwhm <= params["fwhm_max"]):
                continue
            star_count += 1

            height = fwhm * params["spike_length"] / 100.0
            base = max(params["spike_thickness"], fwhm / 10.0)

            base_factor = max(params["spike_length"] / 100.0, 0.2)
            base *= base_factor

            heights = calculate_spike_heights(fwhm, params["spike_length"], len(angles))

            for a, h in zip(angles, heights):
                draw_spike_triangle(mask, x, y, a, h, base)

            # Apply blur to the mask
            y0 = max(0, int(y) - 20)
            y1 = min(image.shape[0], int(y) + 21)
            x0 = max(0, int(x) - 20)
            x1 = min(image.shape[1], int(x) + 21)
            image[y0:y1, x0:x1] = gaussian_filter(
                image[y0:y1, x0:x1], sigma=params["blur_sigma"]
            )

        if image.dtype.kind != 'f':
            image = image.astype(np.float32)

        color_norm = np.array(params["spike_color"], dtype=np.float32) / 255.0
        image[..., 0] += mask * color_norm[0]
        image[..., 1] += mask * color_norm[1]
        image[..., 2] += mask * color_norm[2]

        np.clip(image, 0,
                1 if image.dtype.kind == 'f' else 65535,
                out=image)

        return image, star_count

    def save_spiked_image(self):
        """Save the spiked image with separate mask layer for GIMP and export composed file."""
        if self.image_orig is None:
            return

        # Get the original image dimensions
        height, width = self.image_orig.shape[:2]

        # Create a blank mask image (same size as original)
        spikes_mask = np.zeros((height, width), dtype=np.float32)

        # Apply spikes to the mask (without color)
        params = self._get_preview_params()
        angles = compute_spike_angles(
            params["spike_count"],
            params["pattern"],
            params["rotation_angle"]
        )

        star_count = 0
        for x, y, fwhm in params["stars_list"]:
            if not (params["fwhm_min"] <= fwhm <= params["fwhm_max"]):
                continue
            star_count += 1

            height_val = fwhm * params["spike_length"] / 100.0
            base = max(params["spike_thickness"], fwhm / 10.0)
            base_factor = max(params["spike_length"] / 100.0, 0.2)
            base *= base_factor

            heights = calculate_spike_heights(fwhm, params["spike_length"], len(angles))

            for a, h in zip(angles, heights):
                draw_spike_triangle(spikes_mask, x, y, a, h, base)

            # Apply blur to the mask
            y0 = max(0, int(y) - 20)
            y1 = min(height, int(y) + 21)
            x0 = max(0, int(x) - 20)
            x1 = min(width, int(x) + 21)
            spikes_mask[y0:y1, x0:x1] = gaussian_filter(
                spikes_mask[y0:y1, x0:x1], sigma=params["blur_sigma"]
            )

        # Normalize the mask to 0-1 range
        spikes_mask = np.clip(spikes_mask, 0, 1)

        # Create the composed image (original + spikes)
        processed = self.image_orig.copy()
        if processed.dtype.kind != 'f':
            processed = processed.astype(np.float32)

        # Add colored spikes to the composed image
        color_norm = np.array(params["spike_color"], dtype=np.float32) / 255.0
        processed[..., 0] += spikes_mask * color_norm[0]
        processed[..., 1] += spikes_mask * color_norm[1]
        processed[..., 2] += spikes_mask * color_norm[2]

        np.clip(processed, 0, 1, out=processed)
        processed = np.flipud(processed)  # Vertical mirror for composed image

        # Prepare the pure spikes mask (vertically mirrored, white spikes)
        spikes_mask = np.flipud(spikes_mask)  # Flip the mask vertically
        spikes_mask_8bit = (spikes_mask * 255).astype(np.uint8)  # Convert to 0-255

        # Determine output filenames
        if self.filename:
            base, _ = os.path.splitext(self.filename)
            composed_file = f"{base}_spikes.tif"
            mask_file = f"{base}_spikes_mask.png"  # Pure alpha mask
        else:
            composed_file = "spiked.tif"
            mask_file = "spikes_mask.png"

        try:
            # Save the composed image (32‑bit TIFF)
            import imageio
            img_to_write = processed.astype(np.float32)
            imageio.imwrite(composed_file, img_to_write)

            # Save the pure spikes mask as PNG with transparency
            transparent_mask = np.zeros((height, width, 4), dtype=np.uint8)

            # Set the RGB channels to white (255,255,255) where spikes are
            transparent_mask[..., :3] = 255 * np.stack([spikes_mask_8bit > 0]*3, axis=-1)

            # Set the alpha channel to our spikes mask (white = 255)
            transparent_mask[..., 3] = spikes_mask_8bit

            # Save as PNG with transparency
            mask_img = Image.fromarray(transparent_mask, 'RGBA')
            mask_img.save(mask_file)

            # Load the composed file into Siril
            try:
                escaped_composed_filename = shlex.quote(composed_file)
                SIRIL.cmd(f'load {escaped_composed_filename}')
                self.log_current_parameters()
            except Exception as e:
                logging.warning(f"Could not load {composed_file} into Siril: {e}")

            # Show success message with both files
            QMessageBox.information(
                self, "Export Complete",
                f"Successfully saved:\n"
                f"- Composed image: {composed_file}\n"
                f"- Pure spikes mask (transparent background): {mask_file}"
            )

        except Exception as e:
            logging.error(f"Failed to save spiked image: {e}")
            QMessageBox.critical(self, "Export Failed", f"Error saving files:\n{e}")

    def keyPressEvent(self, event):

        if event.isAutoRepeat():
            return

        if event.key() == Qt.Key.Key_Space:
            self._prev_transform = self.view.transform()

            if not self.showing_original:
                self.show_original_image()
                self.overlay_label.show()
            else:
                self.update_preview()
                self.overlay_label.hide()

            self.showing_original = not self.showing_original
        else:
            super().keyPressEvent(event)

    def show_original_image(self):
        """Show the original image without spikes, preserving zoom."""
        if self.image_orig is None:
            return
        img = self.image_orig.copy()
        if img.dtype.kind != 'u':
            if img.dtype.kind == 'f':
                img = np.clip(img, 0, 1)
                img = (img * 255).astype(np.uint8)
            else:
                img = (img / 256).astype(np.uint8)
        img = np.flipud(img)
        qimg = array_to_qimage(img)

        pixmap = QPixmap.fromImage(qimg).scaled(
            self.view.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.scene.clear()
        item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(item)
        self.scene.setSceneRect(item.boundingRect())

        self.view.setTransform(self._prev_transform)

    def resizeEvent(self, event):
        """Debounce preview update when window size changes."""
        if self.size() != getattr(self, "_prev_size", None):
            self.resize_timer.start(100)
        self._prev_size = self.size()
        super().resizeEvent(event)

    def show_help(self):
        help_widget = self.help_tab()
        dlg = QDialog(self)
        dlg.setWindowTitle("Help")
        layout = QVBoxLayout(dlg)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True) 
        scroll.setWidget(help_widget)
        layout.addWidget(scroll)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        dlg.resize(750, 600)
        dlg.exec()

    def help_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        help_text = """
        <h1 style="color: #88aaff;">Diffraction‑Spike Overlay Tool</h1>

        <p>This tool adds synthetic diffraction spikes to the current image loaded in <strong>Siril</strong>.  
        It provides a live preview, star‑based placement, and the ability to save both the spiked image
        and an alpha‑masked PNG that can be opened in GIMP or other editors.</p>


        <h2 style="color: #88aaff;">Parameters (Controls)</h2>
        <table>
        <tr><th>Control</th><th>Description</th></tr>
        <tr><td><code>FWHM Min / Max</code></td><td>Filters stars by their FWHM (px). Only stars within this range are spiked.</td></tr>
        <tr><td><code>Detected Stars</code></td><td>Number of stars that will receive spikes.</td></tr>
        <tr><td><code>Spike Length (%)</code></td><td>Length of each spike relative to the star’s FWHM.</td></tr>
        <tr><td><code>Spike Thickness (px)</code></td><td>Width of the spike base.</td></tr>
        <tr><td><code>Number of Spikes</code></td><td>4, 6 or 8 radial spikes per star.</td></tr>
        <tr><td><code>Blur Sigma</code></td><td>Gaussian blur applied to each spike (σ).</td></tr>
        <tr><td><code>Rotation Angle (°)</code></td><td>Rotate the whole pattern.</td></tr>
        <tr><td><code>Pattern Shape</code></td><td>Radial, Cross or X.</td></tr>
        <tr><td><code>Spike Color</code></td><td>RGB picker; shown in the small box.</td></tr>
        <tr><td><code>Transparency (%)</code></td><td>Opacity of the star mask overlay in the preview.</td></tr>
    
        <ul>
        </table>

        <h2 style="color: #88aaff;">Workflow</h2>
        <ol>
        <li><strong>Load an image in Siril.</strong> The script automatically converts it to a 32‑bit TIFF if needed.</li>
        <li><strong>Adjust controls.</strong> The preview updates in real time after each change.</li>
        <li><strong>Verify <code>Detected Stars</code>.</strong> Ensure the list matches the stars you want spiked.</li>
        <li><strong>Use zoom/fit buttons</strong> on the preview pane for a detailed view.</li>
        <li><strong>Press <code>Save</code></strong> to export the spiked image and mask.</li>
        <li><strong>Open the mask in GIMP</strong> (or similar) as a layer with transparency for further editing.</li>
        </ol>

        <h2 style="color: #88aaff;">Notes & Limitations</h2>
        <ul>
        <li>The tool works only on RGB stretched images; grayscale is unsupported.</li>
        <li>Large images may take several seconds to process in the preview; full‑resolution saving is faster because it skips live rendering.</li>
        <li>Star detection uses Siril’s <code>get_image_stars()</code>; if no stars are found the spike generation is skipped.</li>
        </ul>

        <h2 style="color: #88aaff;">Keyboard Shortcuts</h2>
        <p><code>Space</code>toggles between the original image and the spiked preview.</p>

        <h2 style="color: #88aaff;">Troubleshooting</h2>
        <ul>
        <li><strong>“No image loaded” error:</strong> Make sure an image is open in Siril before launching the tool.</li>
        <li><strong>“Unsupported image format” error:</strong> The current image must be a 32‑bit RGB TIFF; the script will attempt conversion automatically.</li>
        <li>Check Siril’s console for detailed error logs if the tool crashes or hangs.</li>
        </ul>
        

        <h4 style="color: #88aaff;">Credits</h4>
        <p>Developed for SIRIL.<br>
        (c) G. Trainar (2026)</p>
        """
        help_label = QLabel(help_text)
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: #cccccc; font-size: 12pt;")
        layout.addWidget(help_label)
        return widget

    def log_current_parameters(self):
        """Log current parameters to Siril console."""
        # Get parameter values
        fwhm_min = self.fwhm_min_slider.value()
        fwhm_max = self.fwhm_max_slider.value()
        length = self.spike_length_slider.value()
        thickness = self.spike_thickness_slider.value()
        spike_count = int(self.spike_count_combo.currentText())
        blur_sigma = self.blur_sigma_slider.value() / 10.0
        rotation = self.rotation_slider.value()
        pattern = self.pattern_combo.currentText()
        spike_color = self.spike_color
        star_count = self.star_count_label.text()

        # Log parameters to Siril console
        SIRIL.log(f"#################################################################")
        SIRIL.log(f"The composed file and the mask with alpha channel")
        SIRIL.log(f"have been successfully saved with the following settings:")
        SIRIL.log(f"FWHM Min: {fwhm_min}px")
        SIRIL.log(f"FWHM Max: {fwhm_max}px")
        SIRIL.log(f"Spike Length: {length}%")
        SIRIL.log(f"Spike Thickness: {thickness}px")
        SIRIL.log(f"Number of Spikes: {spike_count}")
        SIRIL.log(f"Blur Sigma: {blur_sigma:.1f}σ")
        SIRIL.log(f"Rotation Angle: {rotation}°")
        SIRIL.log(f"Pattern Shape: {pattern}")
        SIRIL.log(f"Spike Color: RGB{spike_color}")
        SIRIL.log(f"Detected Stars: {star_count}")
        SIRIL.log(f"###################################################################")

def is_tiff(filename: str) -> bool:
    """Check if a filename has a TIFF extension."""
    return os.path.splitext(filename)[1].lower() in {".tif", ".tiff"}

def main():
    """Main application entry point."""
    app = QApplication(sys.argv)

    try:
        # Get the filename of the image currently loaded in Siril
        current_fname = SIRIL.get_image_filename()
    except NoImageError:
        QMessageBox.critical(
            None,
            "No Image Loaded",
            "Please open an image in Siril before running this tool."
        )
        return 1

    # If the current image is not a TIFF, convert it to a 32‑bit TIFF
    if not is_tiff(current_fname):
        try:
            # Build the new filename: <basename>.tif in same directory
            dir_name, base = os.path.split(current_fname)
            new_base = f"{os.path.splitext(base)[0]}.tif"
            new_path = os.path.join(dir_name, new_base)

            # Tell Siril to write a 32‑bit TIFF
            SIRIL.cmd(f'savetif32 "{new_path}" -astro')

            # Load the newly created file back into Siril
            escaped_composed_filename = shlex.quote(new_path)
            SIRIL.cmd(f'load {escaped_composed_filename}')

            # Now the current image is the TIFF; update the filename
            current_fname = new_path
        except Exception as e:
            QMessageBox.critical(
                None,
                "Conversion Failed",
                f"Could not convert the image to a 32‑bit TIFF:\n{e}"
            )
            return 1

    # Fetch the image (now guaranteed to be a TIFF)
    try:
        with wait_for_lock(SIRIL):
            ffit = SIRIL.get_image()
            if ffit is None:
                raise NoImageError
    except NoImageError:
        QMessageBox.critical(
            None,
            "No Image Loaded",
            "Please open an image in Siril before running this tool."
        )
        return 1
    except s.ProcessingThreadBusyError:
        time.sleep(0.5)
        return 1

    # Check if image is RGB or grayscale
    try:
        img_data = ffit.data
        if len(img_data.shape) < 3 or (len(img_data.shape) == 3 and img_data.shape[2] in [1, 2]):
            QMessageBox.critical(
                None,
                "Unsupported Image Format",
                "This tool requires a color (RGB) image. Grayscale or single‑channel images are not supported.\n\n"
                "Please load a color image and try again."
            )
            return 1
    except Exception as e:
        QMessageBox.critical(
            None,
            "Image Processing Error",
            f"Could not verify image format: {e}"
        )
        return 1

    # Create and show the main window
    win = SpikeWindow(preloaded_ffit=ffit)
    win.show()
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
