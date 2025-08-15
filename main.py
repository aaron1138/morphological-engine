# -*- coding: utf-8 -*-
"""
Module: main.py
Author: Gemini
Description: The main entry point for the mSLA Morphological Engine application.
             Initializes and displays the main GUI window.
"""

import sys
import cv2
import json
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QListWidget, QPushButton, QFrame, QLabel, QStatusBar, QFileDialog,
    QListWidgetItem, QProgressBar, QMessageBox, QLineEdit, QCheckBox
)
from PyQt6.QtGui import QAction, QIcon

# --- Core Engine Imports ---
from core.slice_loader import SliceLoader

# --- GUI Widget Imports ---
from gui.parameter_panel import ParameterPanel
from gui.slice_viewer import SliceViewer
from gui.processing_thread import ProcessingThread

# --- Utility Imports ---
from utils import config_manager

class MainWindow(QMainWindow):
    """
    The main application window, which houses all GUI components.
    """
    def __init__(self):
        super().__init__()

        self.setWindowTitle("mSLA Morphological Engine")
        self.setGeometry(100, 100, 1280, 720)
        
        self.slice_loader: SliceLoader | None = None
        self.processing_thread: ProcessingThread | None = None

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        self.left_panel_layout = QVBoxLayout()
        self.left_panel_widget = QWidget()
        self.left_panel_widget.setLayout(self.left_panel_layout)
        self.left_panel_widget.setMaximumWidth(450)
        self.right_panel_layout = QVBoxLayout()
        self.right_panel_widget = QWidget()
        self.right_panel_widget.setLayout(self.right_panel_layout)
        self.main_layout.addWidget(self.left_panel_widget, 1)
        self.main_layout.addWidget(self.right_panel_widget, 3)

        self._create_menu_bar()
        self._create_status_bar()
        self._create_file_management_panel()
        self._create_parameter_panel()
        self._create_output_panel() # New panel for output settings
        self._create_slice_viewer_panel()

        self.show()

    def _create_menu_bar(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        open_action = QAction("&Open Slice Directory...", self)
        open_action.triggered.connect(self.open_directory)
        file_menu.addAction(open_action)
        file_menu.addSeparator()
        save_config_action = QAction("&Save Configuration...", self)
        save_config_action.triggered.connect(self.save_config)
        file_menu.addAction(save_config_action)
        load_config_action = QAction("&Load Configuration...", self)
        load_config_action.triggered.connect(self.load_config)
        file_menu.addAction(load_config_action)
        file_menu.addSeparator()
        exit_action = QAction("&Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def _create_status_bar(self):
        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.status_bar.showMessage("Ready. Please open a slice directory.")

    def _create_file_management_panel(self):
        file_label = QLabel("Slice Files")
        self.file_list_widget = QListWidget()
        self.file_list_widget.currentItemChanged.connect(self.display_selected_slice)
        self.left_panel_layout.addWidget(file_label)
        self.left_panel_layout.addWidget(self.file_list_widget)

    def _create_parameter_panel(self):
        self.param_panel = ParameterPanel()
        self.left_panel_layout.addWidget(self.param_panel)

    def _create_output_panel(self):
        """Creates the panel for output settings."""
        output_frame = QFrame()
        output_frame.setFrameShape(QFrame.Shape.StyledPanel)
        output_layout = QVBoxLayout(output_frame)
        
        title_label = QLabel("Output Settings")
        title_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        output_layout.addWidget(title_label)

        # Output directory selection
        dir_layout = QHBoxLayout()
        self.output_dir_line_edit = QLineEdit()
        self.output_dir_line_edit.setPlaceholderText("Select Output Directory...")
        self.output_dir_line_edit.setReadOnly(True)
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.select_output_directory)
        dir_layout.addWidget(self.output_dir_line_edit)
        dir_layout.addWidget(browse_button)
        output_layout.addLayout(dir_layout)

        # Debug checkbox
        self.debug_checkbox = QCheckBox("Save intermediate debug steps")
        output_layout.addWidget(self.debug_checkbox)

        self.left_panel_layout.addWidget(output_frame)

        # Add the main action button here, after all other controls
        self.run_button = QPushButton("Run Processing")
        self.run_button.setFixedHeight(40)
        self.run_button.setStyleSheet("font-size: 14pt; font-weight: bold;")
        self.run_button.clicked.connect(self.run_processing)
        self.left_panel_layout.addWidget(self.run_button)

    def _create_slice_viewer_panel(self):
        self.slice_viewer = SliceViewer()
        self.right_panel_layout.addWidget(self.slice_viewer)
        
    def open_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Open Slice Directory", ".")
        if not dir_path: return
        try:
            self.status_bar.showMessage(f"Scanning directory: {dir_path}...")
            self.slice_loader = SliceLoader(dir_path)
            self.file_list_widget.clear()
            self.slice_viewer.set_image(None)
            slice_files = self.slice_loader.get_slice_list()
            if not slice_files:
                self.status_bar.showMessage("No valid slice files found.")
                return
            for slice_path in slice_files: self.file_list_widget.addItem(slice_path.name)
            self.status_bar.showMessage(f"Successfully loaded {len(slice_files)} slice files.")
            if self.file_list_widget.count() > 0: self.file_list_widget.setCurrentRow(0)
        except Exception as e:
            self.status_bar.showMessage(f"Error: {e}")
            self.slice_loader = None

    def display_selected_slice(self, current: QListWidgetItem, previous: QListWidgetItem):
        if current is None or self.slice_loader is None: return
        try:
            image_array = cv2.imread(str(self.slice_loader.directory / current.text()), cv2.IMREAD_GRAYSCALE)
            self.slice_viewer.set_image(image_array)
        except Exception as e:
            self.status_bar.showMessage(f"Error displaying slice: {e}")

    def select_output_directory(self):
        """Opens a dialog to select the output directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory", ".")
        if dir_path:
            self.output_dir_line_edit.setText(dir_path)

    def save_config(self):
        config = self.param_panel.get_config()
        if not config["steps"]:
            QMessageBox.warning(self, "Warning", "Pipeline is empty. Nothing to save.")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Configuration", "", "JSON Files (*.json)")
        if not file_path: return
        try:
            config_manager.save_configuration(config, file_path)
            self.status_bar.showMessage(f"Configuration saved to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save configuration:\n{e}")

    def load_config(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Configuration", "", "JSON Files (*.json)")
        if not file_path: return
        try:
            config = config_manager.load_configuration(file_path)
            self.param_panel.set_config(config)
            self.status_bar.showMessage(f"Configuration loaded from {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load configuration:\n{e}")

    def run_processing(self):
        # --- Validation ---
        if self.slice_loader is None or len(self.slice_loader) == 0:
            QMessageBox.warning(self, "Warning", "Please load a slice directory first.")
            return
        output_path = self.output_dir_line_edit.text()
        if not output_path:
            QMessageBox.warning(self, "Warning", "Please select an output directory.")
            return
        config = self.param_panel.get_config()
        if not config["steps"]:
            QMessageBox.warning(self, "Warning", "The processing pipeline is empty. Please add at least one step.")
            return

        self.set_ui_enabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        
        save_debug = self.debug_checkbox.isChecked()
        
        self.processing_thread = ProcessingThread(
            slice_loader=self.slice_loader, 
            config=config, 
            output_path=output_path,
            save_debug=save_debug
        )
        self.processing_thread.progress_update.connect(self.update_progress)
        self.processing_thread.finished.connect(self.on_processing_finished)
        self.processing_thread.error.connect(self.on_processing_error)
        self.processing_thread.start()

    def update_progress(self, value, total):
        self.status_bar.showMessage(f"Processing window {value} of {total}...")
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(value)

    def on_processing_finished(self):
        self.status_bar.showMessage("Processing complete. Files saved to output directory.")
        self.set_ui_enabled(True)
        self.progress_bar.hide()
        QMessageBox.information(self, "Success", "Processing finished successfully.")

    def on_processing_error(self, message: str):
        self.status_bar.showMessage(f"An error occurred: {message}")
        self.set_ui_enabled(True)
        self.progress_bar.hide()
        QMessageBox.critical(self, "Processing Error", f"An error occurred during processing:\n\n{message}")

    def set_ui_enabled(self, enabled: bool):
        self.left_panel_widget.setEnabled(enabled)
        self.menuBar().setEnabled(enabled)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    main_win = MainWindow()
    sys.exit(app.exec())
