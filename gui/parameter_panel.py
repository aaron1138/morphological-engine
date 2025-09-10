# -*- coding: utf-8 -*-
"""
Module: parameter_panel.py
Author: Jules (Refactored for Dynamic Pipelines)
Description: A PyQt6 widget for dynamically configuring the processing pipeline.
             It can switch between a ModernGL shader pipeline and a RawGL
             external tool configuration.
"""

import os
from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QComboBox,
    QPushButton, QListWidget, QListWidgetItem, QStackedWidget, QLineEdit,
    QFileDialog
)
from PyQt6.QtCore import pyqtSignal
from typing import Dict, Any, List

# --- ModernGL Step Widget ---
class ShaderStepWidget(QWidget):
    """A widget representing a single ModernGL shader step."""
    def __init__(self, shader_name: str):
        super().__init__()
        self.shader_name = shader_name
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        display_name = shader_name.replace('_', ' ').title()
        self.label = QLabel(f"<b>{display_name}</b>")
        layout.addWidget(self.label)
        self.setLayout(layout)

    def get_step_config(self) -> Dict[str, Any]:
        return {"shader_name": self.shader_name, "uniforms": {}}

# --- Main Panel ---
class ParameterPanel(QFrame):
    """The main panel for building and managing processing pipelines."""
    config_changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)

        main_layout = QVBoxLayout(self)

        title_label = QLabel("Pipeline Configuration")
        title_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        main_layout.addWidget(title_label)

        # Stacked widget to hold the different UI layouts
        self.stacked_widget = QStackedWidget()
        main_layout.addWidget(self.stacked_widget)

        # Create the two UI pages
        self.moderngl_page = self._create_moderngl_page()
        self.rawgl_page = self._create_rawgl_page()

        self.stacked_widget.addWidget(self.moderngl_page)
        self.stacked_widget.addWidget(self.rawgl_page)

    def set_pipeline_mode(self, mode_text: str):
        """Switches the visible UI based on the selected pipeline mode."""
        if "ModernGL" in mode_text:
            self.stacked_widget.setCurrentWidget(self.moderngl_page)
        elif "RawGL" in mode_text:
            self.stacked_widget.setCurrentWidget(self.rawgl_page)
        self.config_changed.emit()

    # --- ModernGL Page Builder ---
    def _create_moderngl_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0,0,0,0)

        self.ops_list_widget = QListWidget()
        self.ops_list_widget.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        layout.addWidget(self.ops_list_widget)

        add_op_layout = QHBoxLayout()
        self.shader_combo = QComboBox()
        self._populate_shader_combo()

        add_button = QPushButton("Add Shader Step")
        add_button.clicked.connect(self._add_moderngl_step)

        add_op_layout.addWidget(self.shader_combo, 1)
        add_op_layout.addWidget(add_button)
        layout.addLayout(add_op_layout)

        remove_button = QPushButton("Remove Selected Step")
        remove_button.clicked.connect(self._remove_moderngl_step)
        layout.addWidget(remove_button)

        return page

    # --- RawGL Page Builder ---
    def _create_rawgl_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0,0,0,0)

        shader_layout = QHBoxLayout()
        shader_label = QLabel("Shader File:")
        self.rawgl_shader_path_edit = QLineEdit()
        self.rawgl_shader_path_edit.setReadOnly(True)
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self._browse_for_rawgl_shader)

        shader_layout.addWidget(shader_label)
        shader_layout.addWidget(self.rawgl_shader_path_edit, 1)
        shader_layout.addWidget(browse_button)
        layout.addLayout(shader_layout)

        # Read-only info about the output format
        info_label = QLabel("Output format is fixed to 8-bit, single-channel PNG.")
        info_label.setStyleSheet("font-style: italic; color: #888;")
        layout.addWidget(info_label)

        layout.addStretch() # Pushes content to the top
        return page

    # --- Event Handlers ---
    def _populate_shader_combo(self):
        shader_dir = Path("./shaders")
        if not shader_dir.exists(): return
        shader_files = [f for f in os.listdir(shader_dir) if f.endswith(".comp")]
        shader_names = [Path(f).stem for f in shader_files]
        self.shader_combo.addItems(shader_names)

    def _add_moderngl_step(self):
        shader_name = self.shader_combo.currentText()
        if not shader_name: return
        shader_widget = ShaderStepWidget(shader_name)
        list_item = QListWidgetItem(self.ops_list_widget)
        list_item.setSizeHint(shader_widget.sizeHint())
        self.ops_list_widget.addItem(list_item)
        self.ops_list_widget.setItemWidget(list_item, shader_widget)
        self.config_changed.emit()

    def _remove_moderngl_step(self):
        current_row = self.ops_list_widget.currentRow()
        if current_row >= 0:
            self.ops_list_widget.takeItem(current_row)
            self.config_changed.emit()

    def _browse_for_rawgl_shader(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select RawGL Shader", "", "GLSL Files (*.glsl *.comp *.frag *.vert);;All Files (*)")
        if file_path:
            self.rawgl_shader_path_edit.setText(file_path)
            self.config_changed.emit()

    # --- Configuration Methods ---
    def get_config(self) -> Dict[str, Any]:
        """Builds the configuration from the currently active UI page."""
        if self.stacked_widget.currentWidget() == self.moderngl_page:
            steps = []
            for i in range(self.ops_list_widget.count()):
                item = self.ops_list_widget.item(i)
                shader_widget = self.ops_list_widget.itemWidget(item)
                if isinstance(shader_widget, ShaderStepWidget):
                    steps.append(shader_widget.get_step_config())
            return {"steps": steps}
        elif self.stacked_widget.currentWidget() == self.rawgl_page:
            return {"shader_path": self.rawgl_shader_path_edit.text()}
        return {}

    def set_config(self, config: Dict[str, Any]):
        """Populates the UI from a loaded configuration."""
        # This method would need to be aware of the config format to know
        # which page to show and how to populate it.
        # For now, we'll just clear the ModernGL side.
        self.ops_list_widget.clear()
        self.rawgl_shader_path_edit.clear()

        if "steps" in config: # Assumes ModernGL config
            self.set_pipeline_mode("ModernGL")
            for step in config["steps"]:
                shader_name = step.get("shader_name")
                if shader_name:
                    shader_widget = ShaderStepWidget(shader_name)
                    list_item = QListWidgetItem(self.ops_list_widget)
                    list_item.setSizeHint(shader_widget.sizeHint())
                    self.ops_list_widget.addItem(list_item)
                    self.ops_list_widget.setItemWidget(list_item, shader_widget)
        elif "shader_path" in config: # Assumes RawGL config
            self.set_pipeline_mode("RawGL")
            self.rawgl_shader_path_edit.setText(config["shader_path"])

        self.config_changed.emit()
