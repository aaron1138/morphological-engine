import sys
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGridLayout, QLabel, QLineEdit, QPushButton,
    QFileDialog, QSpinBox, QComboBox, QGroupBox, QApplication
)
from PySide6.QtCore import Qt

class RawGLSettingsPanel(QWidget):
    """
    A widget for configuring a parallel RawGL processing job on a DaskGrid.
    """
    def __init__(self):
        super().__init__()

        main_layout = QVBoxLayout(self)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # --- Job Configuration ---
        job_group = QGroupBox("Processing Job Configuration")
        job_layout = QGridLayout(job_group)
        job_layout.setColumnStretch(1, 1)

        # Shader
        job_layout.addWidget(QLabel("Compute Shader:"), 0, 0)
        self.shader_path_edit = QLineEdit()
        browse_shader_btn = QPushButton("Browse...")
        browse_shader_btn.clicked.connect(self._browse_shader)
        job_layout.addWidget(self.shader_path_edit, 0, 1)
        job_layout.addWidget(browse_shader_btn, 0, 2)

        # Output Directory
        job_layout.addWidget(QLabel("Output Directory:"), 1, 0)
        self.output_dir_edit = QLineEdit()
        browse_output_btn = QPushButton("Browse...")
        browse_output_btn.clicked.connect(self._browse_output_dir)
        job_layout.addWidget(self.output_dir_edit, 1, 1)
        job_layout.addWidget(browse_output_btn, 1, 2)

        # Plane Selection
        job_layout.addWidget(QLabel("Processing Plane:"), 2, 0)
        self.plane_combo = QComboBox()
        self.plane_combo.addItems(["XY", "XZ", "YZ"])
        job_layout.addWidget(self.plane_combo, 2, 1, 1, 2)

        # Slice Range
        job_layout.addWidget(QLabel("Slice Range:"), 3, 0)
        range_layout = QHBoxLayout()
        self.start_slice_spin = QSpinBox()
        self.start_slice_spin.setRange(0, 99999)
        self.end_slice_spin = QSpinBox()
        self.end_slice_spin.setRange(0, 99999)
        range_layout.addWidget(QLabel("Start:"))
        range_layout.addWidget(self.start_slice_spin)
        range_layout.addWidget(QLabel("End:"))
        range_layout.addWidget(self.end_slice_spin)
        job_layout.addLayout(range_layout, 3, 1, 1, 2)

        main_layout.addWidget(job_group)

        # --- RawGL Output Format ---
        format_group = QGroupBox("RawGL Output Image Format")
        format_layout = QGridLayout(format_group)

        format_layout.addWidget(QLabel("Format:"), 0, 0)
        self.format_combo = QComboBox()
        self.format_combo.addItems(['r8', 'rg8', 'rgba8', 'r16', 'rgba16', 'r32f', 'rgba32f'])
        self.format_combo.setCurrentText('r8')
        format_layout.addWidget(self.format_combo, 0, 1)

        format_layout.addWidget(QLabel("Channels:"), 1, 0)
        self.channels_spin = QSpinBox()
        self.channels_spin.setRange(1, 4)
        self.channels_spin.setValue(1)
        format_layout.addWidget(self.channels_spin, 1, 1)

        format_layout.addWidget(QLabel("Bits:"), 2, 0)
        self.bits_spin = QSpinBox()
        self.bits_spin.setRange(8, 32)
        self.bits_spin.setValue(8)
        format_layout.addWidget(self.bits_spin, 2, 1)

        main_layout.addWidget(format_group)
        main_layout.addStretch()

    def _browse_shader(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Compute Shader", "", "GLSL Files (*.glsl *.comp)")
        if path:
            self.shader_path_edit.setText(path)

    def _browse_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.output_dir_edit.setText(path)

    def get_processing_config(self) -> dict:
        """Returns a dictionary with the processing job configuration."""
        shader_pass = {
            'pass_comp': self.shader_path_edit.text(),
            # Input and output are now handled by the controller loop
            'out_format': self.format_combo.currentText(),
            'out_channels': self.channels_spin.value(),
            'out_bits': self.bits_spin.value(),
        }

        job_config = {
            'plane': self.plane_combo.currentText(),
            'slice_range': (self.start_slice_spin.value(), self.end_slice_spin.value()),
            'output_dir': self.output_dir_edit.text(),
            'shader_pass': shader_pass
        }
        return job_config

# Example usage for testing
if __name__ == '__main__':
    app = QApplication(sys.argv)
    panel = RawGLSettingsPanel()
    panel.show()
    sys.exit(app.exec())
