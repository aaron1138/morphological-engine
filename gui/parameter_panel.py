# -*- coding: utf-8 -*-
"""
Module: parameter_panel.py
Author: Gemini
Description: A PyQt6 widget for dynamically configuring the processing pipeline.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QComboBox,
    QPushButton, QListWidget, QListWidgetItem, QSpinBox, QLineEdit,
    QFileDialog
)
from PyQt6.QtCore import Qt, pyqtSignal
from typing import Dict, Any, List

class ParameterPanel(QFrame):
    """
    The main panel for building and managing the processing pipeline steps
    and configuring the RawGL executable.
    """
    config_changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)

        main_layout = QVBoxLayout(self)

        # --- RawGL Settings ---
        self._create_rawgl_settings_group(main_layout)

        # --- Divider ---
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shape.Sunken)
        main_layout.addWidget(line)

        # --- Title ---
        title_label = QLabel("Processing Pipeline (Shaders)")
        title_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        main_layout.addWidget(title_label)

        # --- List of Operations ---
        self.ops_list_widget = QListWidget()
        self.ops_list_widget.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.ops_list_widget.setStyleSheet("QListWidget::item { border-bottom: 1px solid #ccc; }")
        main_layout.addWidget(self.ops_list_widget)

        # --- Add Operation Controls ---
        add_op_layout = QHBoxLayout()
        self.op_combo = QComboBox()
        # TODO: This should be populated by scanning the /shaders directory
        self.op_combo.addItems([
            "passthrough", "invert_color" # Example shaders
        ])

        self.add_op_button = QPushButton("Add Shader Step")
        self.add_op_button.clicked.connect(self.add_operation)

        add_op_layout.addWidget(self.op_combo, 1)
        add_op_layout.addWidget(self.add_op_button)
        main_layout.addLayout(add_op_layout)

        # --- Remove Operation Button ---
        self.remove_op_button = QPushButton("Remove Selected Step")
        self.remove_op_button.clicked.connect(self.remove_operation)
        main_layout.addWidget(self.remove_op_button)

    def _create_rawgl_settings_group(self, parent_layout):
        """Creates the UI group for RawGL settings."""
        rawgl_frame = QFrame()
        rawgl_layout = QVBoxLayout(rawgl_frame)
        rawgl_layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel("RawGL Configuration")
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        rawgl_layout.addWidget(title)

        # Path setting
        path_layout = QHBoxLayout()
        self.rawgl_path_edit = QLineEdit()
        self.rawgl_path_edit.setPlaceholderText("Path to rawgl.exe")
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self._browse_for_rawgl)
        path_layout.addWidget(QLabel("Executable:"))
        path_layout.addWidget(self.rawgl_path_edit)
        path_layout.addWidget(browse_button)
        rawgl_layout.addLayout(path_layout)

        # Output format settings
        format_layout = QHBoxLayout()
        self.channels_spinbox = QSpinBox()
        self.channels_spinbox.setRange(1, 4)
        self.channels_spinbox.setValue(1)
        self.bits_spinbox = QComboBox() # Using a combo box as valid values are specific
        self.bits_spinbox.addItems(["8", "16"])

        format_layout.addWidget(QLabel("Output Channels:"))
        format_layout.addWidget(self.channels_spinbox)
        format_layout.addStretch()
        format_layout.addWidget(QLabel("Output Bits:"))
        format_layout.addWidget(self.bits_spinbox)
        rawgl_layout.addLayout(format_layout)

        parent_layout.addWidget(rawgl_frame)

    def _browse_for_rawgl(self):
        """Opens a file dialog to find rawgl.exe."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Find RawGL Executable", "", "Executables (*.exe);;All Files (*)")
        if file_path:
            self.rawgl_path_edit.setText(file_path)

    def get_rawgl_config(self) -> Dict[str, Any]:
        """Returns the RawGL settings from the UI."""
        return {
            "rawgl_path": self.rawgl_path_edit.text(),
            "output_channels": self.channels_spinbox.value(),
            "output_bits": int(self.bits_spinbox.currentText())
        }

    def set_rawgl_config(self, config: Dict[str, Any]):
        """Sets the RawGL settings in the UI from a config dict."""
        self.rawgl_path_edit.setText(config.get("rawgl_path", ""))
        self.channels_spinbox.setValue(config.get("output_channels", 1))
        self.bits_spinbox.setCurrentText(str(config.get("output_bits", 8)))

    def add_operation(self):
        """Adds a new operation to the pipeline list."""
        op_name = self.op_combo.currentText()
        # Using a simple QLabel for shader steps as they have no params for now
        op_widget = QLabel(f"Shader: <b>{op_name}</b>")

        list_item = QListWidgetItem(self.ops_list_widget)
        list_item.setSizeHint(op_widget.sizeHint())
        # Store the op_name in the item itself for retrieval
        list_item.setData(Qt.ItemDataRole.UserRole, op_name)

        self.ops_list_widget.addItem(list_item)
        # self.ops_list_widget.setItemWidget(list_item, op_widget) # This is not needed for a simple label
        self.config_changed.emit()

    def remove_operation(self):
        """Removes the currently selected operation from the list."""
        current_row = self.ops_list_widget.currentRow()
        if current_row >= 0:
            self.ops_list_widget.takeItem(current_row)
            self.config_changed.emit()

    def get_pipeline_config(self) -> Dict[str, Any]:
        """
        Builds and returns the processing pipeline configuration dictionary.
        """
        steps = []
        for i in range(self.ops_list_widget.count()):
            item = self.ops_list_widget.item(i)
            # Retrieve the op_name from the item's data
            op_name = item.data(Qt.ItemDataRole.UserRole)
            if op_name:
                # For now, steps are just the shader name
                steps.append({"operation": op_name})

        return {"steps": steps}

    def set_pipeline_config(self, config: Dict[str, Any]):
        """
        Populates the UI from a loaded pipeline configuration dictionary.
        """
        self.ops_list_widget.clear()
        if "steps" in config:
            for step in config["steps"]:
                op_name = step.get("operation")
                if op_name:
                    # Re-use the add_operation logic, but need to set the combo first
                    # This is a simplified version
                    list_item = QListWidgetItem(f"Shader: {op_name}")
                    list_item.setData(Qt.ItemDataRole.UserRole, op_name)
                    self.ops_list_widget.addItem(list_item)
        self.config_changed.emit()
