# -*- coding: utf-8 -*-
"""
Module: parameter_panel.py
Author: Jules (Refactored for Dask/RawGL)
Description: A PyQt6 widget for configuring the RawGL processing pipeline.
"""

from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel,
    QPushButton, QLineEdit, QFileDialog
)
from PyQt6.QtCore import pyqtSignal
from typing import Dict, Any

class ParameterPanel(QFrame):
    """
    The main panel for configuring the RawGL processing pipeline.
    """
    config_changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)

        main_layout = QVBoxLayout(self)

        title_label = QLabel("RawGL Pipeline Configuration")
        title_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        main_layout.addWidget(title_label)

        # --- RawGL Shader Selection ---
        shader_layout = QHBoxLayout()
        shader_label = QLabel("Shader File:")
        self.rawgl_shader_path_edit = QLineEdit()
        self.rawgl_shader_path_edit.setReadOnly(True)
        self.rawgl_shader_path_edit.textChanged.connect(self.config_changed.emit)

        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self._browse_for_rawgl_shader)

        shader_layout.addWidget(shader_label)
        shader_layout.addWidget(self.rawgl_shader_path_edit, 1)
        shader_layout.addWidget(browse_button)
        main_layout.addLayout(shader_layout)

        # Read-only info about the output format
        info_label = QLabel("Output format is fixed to 8-bit, single-channel PNG.")
        info_label.setStyleSheet("font-style: italic; color: #888;")
        main_layout.addWidget(info_label)

        main_layout.addStretch() # Pushes content to the top

    def _browse_for_rawgl_shader(self):
        """Opens a file dialog to select a shader file for RawGL."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select RawGL Shader", "",
            "GLSL Files (*.glsl *.comp *.frag *.vert);;All Files (*)"
        )
        if file_path:
            self.rawgl_shader_path_edit.setText(file_path)

    def get_config(self) -> Dict[str, Any]:
        """Builds the RawGL pipeline configuration from the UI."""
        shader_path = self.rawgl_shader_path_edit.text()
        if not shader_path:
            return {} # Return empty config if no shader is selected

        return {"shader_path": shader_path}

    def set_config(self, config: Dict[str, Any]):
        """Populates the UI from a loaded RawGL configuration."""
        self.rawgl_shader_path_edit.clear()
        if "shader_path" in config:
            self.rawgl_shader_path_edit.setText(config["shader_path"])
        self.config_changed.emit()
