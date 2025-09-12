# ui/lut_editor.py

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QLineEdit,
    QComboBox, QPushButton, QStackedWidget, QFileDialog
)
from PySide6.QtCore import Signal, Slot
import numpy as np

from config import LutParameters
from utils import lut_manager

# Matplotlib imports
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class LutEditorWidget(QWidget):
    """A widget for editing and visualizing LUT parameters."""
    lut_params_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.lut_params = LutParameters()
        self.init_ui()
        self._connect_signals()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # --- Plot ---
        self.plot_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        main_layout.addWidget(self.plot_canvas)

        # --- Controls ---
        controls_group = QGroupBox("LUT Controls")
        controls_layout = QVBoxLayout(controls_group)

        # Source
        source_layout = QHBoxLayout()
        source_layout.addWidget(QLabel("Source:"))
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Generated", "File"])
        source_layout.addWidget(self.source_combo)
        controls_layout.addLayout(source_layout)

        self.controls_stack = QStackedWidget()
        controls_layout.addWidget(self.controls_stack)

        # Generated controls
        generated_widget = QWidget()
        generated_layout = QVBoxLayout(generated_widget)
        self.type_combo = QComboBox()
        self.type_combo.addItems(["linear", "gamma"]) # Simplified for now
        generated_layout.addWidget(self.type_combo)
        self.gamma_edit = QLineEdit()
        generated_layout.addWidget(self.gamma_edit)
        self.controls_stack.addWidget(generated_widget)

        # File controls
        file_widget = QWidget()
        file_layout = QHBoxLayout(file_widget)
        self.lut_file_edit = QLineEdit()
        file_layout.addWidget(self.lut_file_edit)
        self.browse_lut_button = QPushButton("Browse...")
        file_layout.addWidget(self.browse_lut_button)
        self.controls_stack.addWidget(file_widget)

        main_layout.addWidget(controls_group)

    def _connect_signals(self):
        self.source_combo.currentIndexChanged.connect(self.controls_stack.setCurrentIndex)
        self.source_combo.currentIndexChanged.connect(self.update_params)
        self.type_combo.currentTextChanged.connect(self.update_params)
        self.gamma_edit.editingFinished.connect(self.update_params)
        self.browse_lut_button.clicked.connect(self._browse_for_lut)

    def set_lut_params(self, params: LutParameters):
        self.lut_params = params
        self.source_combo.setCurrentText(self.lut_params.lut_source.capitalize())
        self.type_combo.setCurrentText(self.lut_params.lut_generation_type)
        self.gamma_edit.setText(str(self.lut_params.gamma_value))
        self.lut_file_edit.setText(self.lut_params.fixed_lut_path)
        self.plot_current_lut()

    @Slot()
    def update_params(self):
        self.lut_params.lut_source = self.source_combo.currentText().lower()
        self.lut_params.lut_generation_type = self.type_combo.currentText()
        try:
            self.lut_params.gamma_value = float(self.gamma_edit.text())
        except ValueError:
            self.lut_params.gamma_value = 1.0 # Default
        self.lut_params.fixed_lut_path = self.lut_file_edit.text()

        self.plot_current_lut()
        self.lut_params_changed.emit()

    def plot_current_lut(self):
        lut = self._get_lut_from_params()
        self.plot_canvas.axes.cla()
        self.plot_canvas.axes.plot(np.arange(256), lut)
        self.plot_canvas.axes.set_title("LUT Curve")
        self.plot_canvas.axes.set_xlabel("Input")
        self.plot_canvas.axes.set_ylabel("Output")
        self.plot_canvas.axes.grid(True)
        self.plot_canvas.draw()

    def _get_lut_from_params(self):
        if self.lut_params.lut_source == "file":
            try:
                return lut_manager.load_lut(self.lut_params.fixed_lut_path)
            except FileNotFoundError:
                return lut_manager.get_default_z_lut()
        else: # generated
            if self.lut_params.lut_generation_type == "gamma":
                return lut_manager.generate_gamma_lut(self.lut_params.gamma_value, 0, 255, 0, 255)
            else: # linear
                return lut_manager.generate_linear_lut(0, 255, 0, 255)

    @Slot()
    def _browse_for_lut(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select LUT File", "", "JSON Files (*.json)")
        if filepath:
            self.lut_file_edit.setText(filepath)
            self.update_params()
