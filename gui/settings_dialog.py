# -*- coding: utf-8 -*-
"""
Module: settings_dialog.py
Author: Jules
Description: A dialog for configuring application-wide settings, such as GPU selection.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QComboBox,
    QPushButton, QLabel, QDialogButtonBox
)
from typing import List, Dict, Any

# Assuming these modules are available for import
from utils.app_settings import AppSettings
from core.gpu_processor import GpuProcessor

class GpuSettingsDialog(QDialog):
    """
    A dialog to allow the user to select the GPU for processing.
    """
    def __init__(self, settings_manager: AppSettings, parent=None):
        super().__init__(parent)
        self.settings_manager = settings_manager

        self.setWindowTitle("GPU Settings")
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)

        # --- GPU Selection ---
        gpu_layout = QHBoxLayout()
        gpu_label = QLabel("Processing GPU:")
        self.gpu_combo = QComboBox()

        gpu_layout.addWidget(gpu_label)
        gpu_layout.addWidget(self.gpu_combo, 1)
        layout.addLayout(gpu_layout)

        # --- RawGL Executable Path ---
        rawgl_layout = QHBoxLayout()
        rawgl_label = QLabel("RawGL Executable:")
        self.rawgl_path_edit = QLineEdit()
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self._browse_for_rawgl)

        rawgl_layout.addWidget(rawgl_label)
        rawgl_layout.addWidget(self.rawgl_path_edit)
        rawgl_layout.addWidget(browse_button)
        layout.addLayout(rawgl_layout)

        self.populate_gpu_list()
        self.load_settings()

        # --- Dialog Buttons ---
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _browse_for_rawgl(self):
        """Opens a file dialog to find the rawgl.exe executable."""
        from PyQt6.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select RawGL Executable", "", "Executables (*.exe);;All Files (*)"
        )
        if file_path:
            self.rawgl_path_edit.setText(file_path)

    def populate_gpu_list(self):
        """Fills the combo box with available GPUs."""
        self.gpu_combo.clear()
        self.gpu_combo.addItem("Automatic Selection", -1)
        available_gpus = GpuProcessor.list_gpus()
        for gpu in available_gpus:
            self.gpu_combo.addItem(gpu['name'], gpu['id'])

    def load_settings(self):
        """Populates the dialog's widgets with current settings."""
        # Load RawGL path
        rawgl_path = self.settings_manager.get("rawgl_executable_path", "")
        self.rawgl_path_edit.setText(rawgl_path)

        # Load GPU selection
        current_gpu_id = self.settings_manager.get("gpu_device_id", -1)
        index_to_set = self.gpu_combo.findData(current_gpu_id)
        if index_to_set != -1:
            self.gpu_combo.setCurrentIndex(index_to_set)

    def accept(self):
        """Saves all settings when OK is clicked."""
        # Save GPU ID
        selected_gpu_id = self.gpu_combo.currentData()
        self.settings_manager.settings["gpu_device_id"] = selected_gpu_id

        # Save RawGL Path
        rawgl_path = self.rawgl_path_edit.text()
        self.settings_manager.settings["rawgl_executable_path"] = rawgl_path

        # Save all settings at once
        self.settings_manager.save()
        print(f"Settings saved. GPU ID: {selected_gpu_id}, RawGL Path: {rawgl_path}")

        super().accept()
