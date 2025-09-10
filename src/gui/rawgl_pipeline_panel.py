import sys
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QListWidget, QPushButton, QStackedWidget,
    QFrame, QGridLayout, QLabel, QLineEdit, QFileDialog, QSpinBox, QComboBox,
    QGroupBox, QApplication
)
from PySide6.QtCore import Qt, Signal

class _StepSettingsWidget(QWidget):
    """A widget for editing the parameters of a single pipeline step."""
    settings_changed = Signal()

    def __init__(self, step_data: dict):
        super().__init__()
        self.data = step_data

        layout = QGridLayout(self)
        layout.setColumnStretch(1, 1)

        # Shader Path
        layout.addWidget(QLabel("Compute Shader:"), 0, 0)
        self.shader_path_edit = QLineEdit(self.data.get('pass_comp', ''))
        self.shader_path_edit.textChanged.connect(self._on_settings_changed)
        browse_shader_btn = QPushButton("Browse...")
        browse_shader_btn.clicked.connect(self._browse_shader)
        layout.addWidget(self.shader_path_edit, 0, 1)
        layout.addWidget(browse_shader_btn, 0, 2)

        # Input Texture
        layout.addWidget(QLabel("Input Texture:"), 1, 0)
        # This is simplified. A real app would have a dropdown to select previous outputs.
        self.input_path_edit = QLineEdit(self.data.get('in', {}).get('Texture0', ''))
        self.input_path_edit.textChanged.connect(self._on_settings_changed)
        layout.addWidget(self.input_path_edit, 1, 1, 1, 2)

        # Output Path
        layout.addWidget(QLabel("Output Path:"), 2, 0)
        self.output_path_edit = QLineEdit(self.data.get('out', {}).get('OutColor', ''))
        self.output_path_edit.setPlaceholderText("Leave blank for temp file, or specify final path")
        self.output_path_edit.textChanged.connect(self._on_settings_changed)
        layout.addWidget(self.output_path_edit, 2, 1, 1, 2)

        # Output Format Group
        format_group = QGroupBox("Output Image Format")
        format_layout = QGridLayout(format_group)

        format_layout.addWidget(QLabel("Format:"), 0, 0)
        self.format_combo = QComboBox()
        self.format_combo.addItems(['r8', 'rg8', 'rgba8', 'r16', 'rgba16', 'r32f', 'rgba32f'])
        self.format_combo.setCurrentText(self.data.get('out_format', 'r8'))
        self.format_combo.currentTextChanged.connect(self._on_settings_changed)
        format_layout.addWidget(self.format_combo, 0, 1)

        format_layout.addWidget(QLabel("Channels:"), 1, 0)
        self.channels_spin = QSpinBox()
        self.channels_spin.setRange(1, 4)
        self.channels_spin.setValue(self.data.get('out_channels', 1))
        self.channels_spin.valueChanged.connect(self._on_settings_changed)
        format_layout.addWidget(self.channels_spin, 1, 1)

        format_layout.addWidget(QLabel("Bits:"), 2, 0)
        self.bits_spin = QSpinBox()
        self.bits_spin.setRange(8, 32)
        self.bits_spin.setSpecialValueText("Auto")
        self.bits_spin.setValue(self.data.get('out_bits', 8))
        self.bits_spin.valueChanged.connect(self._on_settings_changed)
        format_layout.addWidget(self.bits_spin, 2, 1)

        layout.addWidget(format_group, 3, 0, 1, 3)
        layout.addStretch()

    def _browse_shader(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Compute Shader", "", "GLSL Files (*.glsl *.comp)")
        if path:
            self.shader_path_edit.setText(path)

    def _on_settings_changed(self):
        """Update the internal data dictionary when a UI element changes."""
        self.data['pass_comp'] = self.shader_path_edit.text()
        self.data['in'] = {'Texture0': self.input_path_edit.text()}
        # Handle temp vs final output
        out_path = self.output_path_edit.text()
        self.data['out'] = {'OutColor': 'TEMP' if not out_path else out_path}
        self.data['out_format'] = self.format_combo.currentText()
        self.data['out_channels'] = self.channels_spin.value()
        self.data['out_bits'] = self.bits_spin.value()
        self.settings_changed.emit()

class RawGLPipelinePanel(QWidget):
    """
    A widget for building and configuring a RawGL processing pipeline.
    """
    def __init__(self):
        super().__init__()
        self.pipeline = []

        main_layout = QHBoxLayout(self)

        # --- Left Side (List and Controls) ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(250)

        self.steps_list = QListWidget()
        self.steps_list.currentItemChanged.connect(self._on_step_selected)

        controls_layout = QHBoxLayout()
        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self._add_step)
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self._remove_step)
        up_btn = QPushButton("Up")
        up_btn.clicked.connect(self._move_step_up)
        down_btn = QPushButton("Down")
        down_btn.clicked.connect(self._move_step_down)

        controls_layout.addWidget(add_btn)
        controls_layout.addWidget(remove_btn)
        controls_layout.addWidget(up_btn)
        controls_layout.addWidget(down_btn)

        left_layout.addWidget(self.steps_list)
        left_layout.addLayout(controls_layout)
        main_layout.addWidget(left_panel)

        # --- Right Side (Settings Stack) ---
        self.settings_stack = QStackedWidget()
        main_layout.addWidget(self.settings_stack)

    def _add_step(self):
        step_num = len(self.pipeline) + 1
        step_name = f"Step {step_num}: New Shader Pass"

        # Create default data for the new step
        new_step_data = {
            'pass_comp': '',
            'in': {'Texture0': ''},
            'out': {'OutColor': 'TEMP'},
            'out_format': 'r8',
            'out_channels': 1,
            'out_bits': 8
        }
        self.pipeline.append(new_step_data)

        # Create UI elements
        self.steps_list.addItem(step_name)
        settings_widget = _StepSettingsWidget(new_step_data)
        settings_widget.settings_changed.connect(self._update_step_name)
        self.settings_stack.addWidget(settings_widget)
        self.steps_list.setCurrentRow(len(self.pipeline) - 1)

    def _remove_step(self):
        current_row = self.steps_list.currentRow()
        if current_row < 0: return

        # Remove from data list, list widget, and stack
        self.pipeline.pop(current_row)
        item = self.steps_list.takeItem(current_row)
        widget = self.settings_stack.widget(current_row)
        self.settings_stack.removeWidget(widget)
        del item
        del widget

    def _move_step_up(self):
        current_row = self.steps_list.currentRow()
        if current_row <= 0: return

        # Swap in data list
        self.pipeline[current_row], self.pipeline[current_row - 1] = self.pipeline[current_row - 1], self.pipeline[current_row]

        # Re-sync UI
        self._sync_ui_to_pipeline()
        self.steps_list.setCurrentRow(current_row - 1)

    def _move_step_down(self):
        current_row = self.steps_list.currentRow()
        if current_row < 0 or current_row >= len(self.pipeline) - 1: return

        self.pipeline[current_row], self.pipeline[current_row + 1] = self.pipeline[current_row + 1], self.pipeline[current_row]

        self._sync_ui_to_pipeline()
        self.steps_list.setCurrentRow(current_row + 1)

    def _on_step_selected(self, current_item):
        if current_item is None:
            self.settings_stack.setCurrentIndex(-1)
        else:
            current_row = self.steps_list.row(current_item)
            self.settings_stack.setCurrentIndex(current_row)

    def _update_step_name(self):
        """Updates the list widget item text based on the shader name."""
        current_row = self.steps_list.currentRow()
        if current_row < 0: return

        widget = self.settings_stack.widget(current_row)
        shader_path = Path(widget.data.get('pass_comp', ''))
        step_name = f"Step {current_row + 1}: {shader_path.name if shader_path.name else 'New Shader Pass'}"
        self.steps_list.item(current_row).setText(step_name)

    def _sync_ui_to_pipeline(self):
        """Rebuilds the entire UI state from the pipeline data list."""
        # Clear UI
        while self.settings_stack.count() > 0:
            widget = self.settings_stack.widget(0)
            self.settings_stack.removeWidget(widget)
            del widget
        self.steps_list.clear()

        # Rebuild UI from data
        for i, step_data in enumerate(self.pipeline):
            shader_path = Path(step_data.get('pass_comp', ''))
            step_name = f"Step {i + 1}: {shader_path.name if shader_path.name else 'New Shader Pass'}"
            self.steps_list.addItem(step_name)

            settings_widget = _StepSettingsWidget(step_data)
            settings_widget.settings_changed.connect(self._update_step_name)
            self.settings_stack.addWidget(settings_widget)

    def get_pipeline(self) -> list:
        """Returns the current pipeline definition."""
        return self.pipeline

# Example usage for testing
if __name__ == '__main__':
    app = QApplication(sys.argv)
    panel = RawGLPipelinePanel()
    panel.show()
    sys.exit(app.exec())
