# -*- coding: utf-8 -*-
"""
Module: rawgl_panel.py
Author: Jules
Description: A UI panel for configuring a single RawGL shader pass.
"""

from PyQt6.QtWidgets import (
    QWidget, QFrame, QVBoxLayout, QFormLayout, QLineEdit, QPushButton,
    QFileDialog, QLabel
)

class RawGLPanel(QFrame):
    """
    A widget that contains all the controls for configuring a single RawGL pass.
    """
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)

        self.main_layout = QVBoxLayout(self)

        title_label = QLabel("RawGL Pass Configuration")
        title_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        self.main_layout.addWidget(title_label)

        # Form layout for the settings
        self.form_layout = QFormLayout()

        # --- Executable Path ---
        self.exec_path_edit = QLineEdit()
        self.exec_path_edit.setPlaceholderText("Path to rawgl.exe")
        browse_exec_button = QPushButton("Browse...")
        browse_exec_button.clicked.connect(lambda: self._browse_file(self.exec_path_edit, "Select RawGL Executable"))
        exec_layout = _create_browse_layout(self.exec_path_edit, browse_exec_button)
        self.form_layout.addRow("RawGL Executable:", exec_layout)

        # --- Input Texture ---
        self.in_texture_edit = QLineEdit()
        self.in_texture_edit.setPlaceholderText("Path to input texture (e.g., PNG)")
        browse_in_button = QPushButton("Browse...")
        browse_in_button.clicked.connect(lambda: self._browse_file(self.in_texture_edit, "Select Input Texture"))
        in_layout = _create_browse_layout(self.in_texture_edit, browse_in_button)
        self.form_layout.addRow("Input Texture:", in_layout)

        # --- Shader File ---
        self.shader_path_edit = QLineEdit()
        self.shader_path_edit.setPlaceholderText("Path to .glsl or .vertfrag shader")
        browse_shader_button = QPushButton("Browse...")
        browse_shader_button.clicked.connect(lambda: self._browse_file(self.shader_path_edit, "Select Shader File"))
        shader_layout = _create_browse_layout(self.shader_path_edit, browse_shader_button)
        self.form_layout.addRow("Shader File:", shader_layout)

        # --- Output Path ---
        self.out_path_edit = QLineEdit()
        self.out_path_edit.setPlaceholderText("Path to save output PNG")
        browse_out_button = QPushButton("Browse...")
        browse_out_button.clicked.connect(lambda: self._browse_save_file(self.out_path_edit, "Select Output File"))
        out_layout = _create_browse_layout(self.out_path_edit, browse_out_button)
        self.form_layout.addRow("Output File:", out_layout)

        # --- Pass Size ---
        self.pass_size_edit = QLineEdit("512 512")
        self.form_layout.addRow("Output Size (W H):", self.pass_size_edit)

        self.main_layout.addLayout(self.form_layout)

    def _browse_file(self, line_edit, dialog_title):
        file_path, _ = QFileDialog.getOpenFileName(self, dialog_title, "")
        if file_path:
            line_edit.setText(file_path)

    def _browse_save_file(self, line_edit, dialog_title):
        file_path, _ = QFileDialog.getSaveFileName(self, dialog_title, "", "PNG Files (*.png)")
        if file_path:
            line_edit.setText(file_path)

    def get_config(self):
        """
        Returns a dictionary with the configuration for the RawGL pass.
        """
        if not self.exec_path_edit.text():
            raise ValueError("RawGL Executable path must be set.")

        # Basic config for a single pass
        config = {
            "pass_vertfrag": self.shader_path_edit.text(),
            "pass_size": self.pass_size_edit.text(),
            "in": f"Texture0 {self.in_texture_edit.text()}",
            "out": f"OutColor {self.out_path_edit.text()}",
            # Hardcoded values for 8-bit single-channel greyscale PNG
            "out_channels": 1,
            "out_bits": 8,
            "out_format": "r8"
        }
        return self.exec_path_edit.text(), config

    def set_config(self, config):
        """
        Populates the UI fields from a configuration dictionary.
        """
        self.shader_path_edit.setText(config.get("pass_vertfrag", ""))
        self.pass_size_edit.setText(config.get("pass_size", "512 512"))

        # Handle complex "in" and "out" fields
        in_value = config.get("in", "")
        if in_value.startswith("Texture0 "):
            self.in_texture_edit.setText(in_value.split(" ", 1)[1])
        else:
            self.in_texture_edit.setText(in_value)

        out_value = config.get("out", "")
        if out_value.startswith("OutColor "):
            self.out_path_edit.setText(out_value.split(" ", 1)[1])
        else:
            self.out_path_edit.setText(out_value)

# Helper function to create a layout with a line edit and a browse button
def _create_browse_layout(line_edit, button):
    from PyQt6.QtWidgets import QHBoxLayout, QWidget
    layout = QHBoxLayout()
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(line_edit)
    layout.addWidget(button)
    widget = QWidget()
    widget.setLayout(layout)
    return widget
