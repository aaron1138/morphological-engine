# -*- coding: utf-8 -*-
"""
Module: parameter_panel.py
Author: Jules (Refactored for GPU Shaders)
Description: A PyQt6 widget for dynamically configuring the GPU processing pipeline
             by selecting and ordering GLSL compute shaders.
"""

import os
from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QComboBox,
    QPushButton, QListWidget, QListWidgetItem
)
from PyQt6.QtCore import pyqtSignal
from typing import Dict, Any, List

class ShaderStepWidget(QWidget):
    """
    A widget representing a single shader step in the processing pipeline.
    For now, it simply displays the shader's name. Future versions could
    parse shader uniforms and generate controls for them.
    """
    def __init__(self, shader_name: str):
        super().__init__()
        self.shader_name = shader_name

        layout = QHBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)

        # Display the shader name, formatted nicely
        display_name = shader_name.replace('_', ' ').title()
        self.label = QLabel(f"<b>{display_name}</b>")
        layout.addWidget(self.label)

        # Placeholder for future auto-generated uniform controls
        self.param_widgets = {}

        self.setLayout(layout)

    def get_step_config(self) -> Dict[str, Any]:
        """Returns the configuration for this shader step."""
        # In the future, this would gather values from self.param_widgets
        uniforms = {}
        return {"shader_name": self.shader_name, "uniforms": uniforms}

class ParameterPanel(QFrame):
    """
    The main panel for building and managing the GPU processing pipeline.
    """
    config_changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)

        main_layout = QVBoxLayout(self)

        title_label = QLabel("GPU Processing Pipeline")
        title_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        main_layout.addWidget(title_label)

        self.ops_list_widget = QListWidget()
        self.ops_list_widget.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.ops_list_widget.setStyleSheet("QListWidget::item { border-bottom: 1px solid #ccc; }")
        main_layout.addWidget(self.ops_list_widget)

        add_op_layout = QHBoxLayout()
        self.shader_combo = QComboBox()
        self._populate_shader_combo()

        self.add_step_button = QPushButton("Add Shader Step")
        self.add_step_button.clicked.connect(self.add_shader_step)

        add_op_layout.addWidget(self.shader_combo, 1)
        add_op_layout.addWidget(self.add_step_button)
        main_layout.addLayout(add_op_layout)

        self.remove_step_button = QPushButton("Remove Selected Step")
        self.remove_step_button.clicked.connect(self.remove_shader_step)
        main_layout.addWidget(self.remove_step_button)

    def _populate_shader_combo(self):
        """Scans the 'shaders' directory and populates the combo box."""
        shader_dir = Path("./shaders")
        if not shader_dir.exists():
            print("Warning: 'shaders' directory not found.")
            return

        # Find all files ending in .comp (compute shader)
        shader_files = [f for f in os.listdir(shader_dir) if f.endswith(".comp")]
        shader_names = [Path(f).stem for f in shader_files] # Get filename without extension

        self.shader_combo.addItems(shader_names)

    def add_shader_step(self):
        """Adds a new shader step to the pipeline list."""
        shader_name = self.shader_combo.currentText()
        if not shader_name:
            return

        shader_widget = ShaderStepWidget(shader_name)

        list_item = QListWidgetItem(self.ops_list_widget)
        list_item.setSizeHint(shader_widget.sizeHint())

        self.ops_list_widget.addItem(list_item)
        self.ops_list_widget.setItemWidget(list_item, shader_widget)
        self.config_changed.emit()

    def remove_shader_step(self):
        """Removes the currently selected shader step from the list."""
        current_row = self.ops_list_widget.currentRow()
        if current_row >= 0:
            self.ops_list_widget.takeItem(current_row)
            self.config_changed.emit()

    def get_config(self) -> Dict[str, Any]:
        """Builds the GPU pipeline configuration from the UI widgets."""
        steps = []
        for i in range(self.ops_list_widget.count()):
            item = self.ops_list_widget.item(i)
            shader_widget = self.ops_list_widget.itemWidget(item)
            if isinstance(shader_widget, ShaderStepWidget):
                steps.append(shader_widget.get_step_config())

        return {"steps": steps}

    def set_config(self, config: Dict[str, Any]):
        """Populates the UI from a loaded GPU pipeline configuration."""
        self.ops_list_widget.clear()
        if "steps" in config:
            for step in config["steps"]:
                shader_name = step.get("shader_name")
                if shader_name:
                    # Future: Pass uniform values from 'step' to the widget
                    shader_widget = ShaderStepWidget(shader_name)

                    list_item = QListWidgetItem(self.ops_list_widget)
                    list_item.setSizeHint(shader_widget.sizeHint())
                    self.ops_list_widget.addItem(list_item)
                    self.ops_list_widget.setItemWidget(list_item, shader_widget)
        self.config_changed.emit()
