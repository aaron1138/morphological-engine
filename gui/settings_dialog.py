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

        self.populate_gpu_list()

        # --- Dialog Buttons ---
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def populate_gpu_list(self):
        """
        Fills the combo box with available GPUs and sets the current selection.
        """
        self.gpu_combo.clear()

        # Add an "Auto" option first
        self.gpu_combo.addItem("Automatic Selection", -1)

        # Get GPUs from GpuProcessor
        # Note: This requires `moderngl-window` to be installed.
        available_gpus = GpuProcessor.list_gpus()

        for gpu in available_gpus:
            self.gpu_combo.addItem(gpu['name'], gpu['id'])

        # Set the current selection based on saved settings
        current_gpu_id = self.settings_manager.get("gpu_device_id", -1)
        # Find the index in the combo box that corresponds to the saved ID
        index_to_set = self.gpu_combo.findData(current_gpu_id)
        if index_to_set != -1:
            self.gpu_combo.setCurrentIndex(index_to_set)

    def accept(self):
        """Saves the selected GPU ID when OK is clicked."""
        selected_id = self.gpu_combo.currentData()
        self.settings_manager.set("gpu_device_id", selected_id)
        print(f"GPU preference saved. Device ID: {selected_id}")
        super().accept()
