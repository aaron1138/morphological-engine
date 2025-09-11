# -*- coding: utf-8 -*-
"""
Module: rawgl_controller.py
Author: Jules
Description: A PyQt6 widget for configuring and running RawGL processing pipelines.
"""

import subprocess
import os
import tempfile
import numpy as np
from PIL import Image
from dask.distributed import Client, LocalCluster
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QPushButton,
    QLineEdit, QFileDialog, QTextEdit, QMessageBox, QSpinBox, QCheckBox
)
from PyQt6.QtCore import QThread, pyqtSignal
from core.dask_handler import load_image_stack_as_dask_array

def run_rawgl_on_slice(slice_data, output_path, rawgl_path, shader_path):
    """
    This function is executed by a Dask worker. It saves a numpy array slice
    to a temporary file and runs RawGL on it.
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_in:
        input_filepath = tmp_in.name
        Image.fromarray(slice_data).save(input_filepath)

    try:
        command = [
            rawgl_path,
            '--pass_vertfrag', shader_path,
            '--in', 'Texture0', input_filepath,
            '--out', 'OutColor', output_path,
            '--out_format', 'r8',
            '--out_channels', '1',
            '--out_bits', '8'
        ]
        # Use subprocess.run as we want to wait for completion
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return f"Successfully processed {output_path}\n{result.stdout}"
    finally:
        # Clean up the temporary input file
        os.remove(input_filepath)

class DaskRawGLThread(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            self.progress.emit(f"Initializing Dask cluster with {self.params['workers']} workers...")
            with LocalCluster(n_workers=self.params['workers'], threads_per_worker=1) as cluster, Client(cluster) as client:
                self.progress.emit(f"Dask dashboard link: {client.dashboard_link}")

                dask_array = load_image_stack_as_dask_array(self.params['input_dir'], chunk_size=(64, 64, 64))
                if dask_array is None:
                    self.error.emit("Failed to create Dask array. Check input directory.")
                    return

                self.progress.emit(f"Dask array shape: {dask_array.shape}, chunks: {dask_array.chunksize}")

                tasks = []
                # XY Plane Processing
                if self.params['process_xy']:
                    self.progress.emit("Generating XY plane tasks...")
                    for i in range(dask_array.shape[0]):
                        output_path = os.path.join(self.params['output_dir'], f"xy_slice_{i:04d}.png")
                        tasks.append(dask.delayed(run_rawgl_on_slice)(dask_array[i, :, :], output_path, self.params['rawgl_path'], self.params['shader_path']))

                # XZ Plane Processing
                if self.params['process_xz']:
                    self.progress.emit("Generating XZ plane tasks...")
                    for i in range(dask_array.shape[1]):
                         output_path = os.path.join(self.params['output_dir'], f"xz_slice_{i:04d}.png")
                         tasks.append(dask.delayed(run_rawgl_on_slice)(dask_array[:, i, :], output_path, self.params['rawgl_path'], self.params['shader_path']))

                # YZ Plane Processing
                if self.params['process_yz']:
                    self.progress.emit("Generating YZ plane tasks...")
                    for i in range(dask_array.shape[2]):
                        output_path = os.path.join(self.params['output_dir'], f"yz_slice_{i:04d}.png")
                        tasks.append(dask.delayed(run_rawgl_on_slice)(dask_array[:, :, i], output_path, self.params['rawgl_path'], self.params['shader_path']))

                if not tasks:
                    self.error.emit("No processing planes selected. Nothing to do.")
                    return

                self.progress.emit(f"Submitting {len(tasks)} tasks to Dask workers...")
                results = dask.compute(*tasks)

                for res in results:
                    self.progress.emit(res)

        except Exception as e:
            self.error.emit(f"An unexpected error occurred in the Dask thread: {e}")
        finally:
            self.finished.emit()

class RawGLController(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)

        main_layout = QVBoxLayout(self)

        title_label = QLabel("RawGL Dask Processor")
        title_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        main_layout.addWidget(title_label)

        # --- UI for selecting paths ---
        self.rawgl_path = self._create_path_selector("RawGL Executable:")
        main_layout.addLayout(self.rawgl_path['layout'])

        self.shader_path = self._create_path_selector("Shader File:")
        main_layout.addLayout(self.shader_path['layout'])

        self.input_dir = self._create_path_selector("Input Directory:", is_directory=True)
        main_layout.addLayout(self.input_dir['layout'])

        self.output_dir = self._create_path_selector("Output Directory:", is_directory=True)
        main_layout.addLayout(self.output_dir['layout'])

        # --- Dask Worker Settings ---
        dask_layout = QHBoxLayout()
        dask_label = QLabel("Dask Workers:")
        self.worker_spinbox = QSpinBox()
        self.worker_spinbox.setRange(1, 16)
        self.worker_spinbox.setValue(4)
        dask_layout.addWidget(dask_label)
        dask_layout.addWidget(self.worker_spinbox)
        dask_layout.addStretch()
        main_layout.addLayout(dask_layout)

        # --- Plane Selection ---
        plane_layout = QHBoxLayout()
        plane_label = QLabel("Process Planes:")
        self.xy_checkbox = QCheckBox("XY")
        self.xz_checkbox = QCheckBox("XZ")
        self.yz_checkbox = QCheckBox("YZ")
        self.xy_checkbox.setChecked(True)
        plane_layout.addWidget(plane_label)
        plane_layout.addWidget(self.xy_checkbox)
        plane_layout.addWidget(self.xz_checkbox)
        plane_layout.addWidget(self.yz_checkbox)
        plane_layout.addStretch()
        main_layout.addLayout(plane_layout)

        # --- Run Button ---
        self.run_button = QPushButton("Run Dask Processing")
        self.run_button.setFixedHeight(30)
        self.run_button.setStyleSheet("font-size: 12pt;")
        main_layout.addWidget(self.run_button)

        # --- Log Output ---
        log_label = QLabel("Processing Log:")
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setFontFamily("Courier")
        self.log_output.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        main_layout.addWidget(log_label)
        main_layout.addWidget(self.log_output)

        # Connect signals to slots
        self.rawgl_path['button'].clicked.connect(lambda: self._get_path(self.rawgl_path['line_edit']))
        self.shader_path['button'].clicked.connect(lambda: self._get_path(self.shader_path['line_edit']))
        self.input_dir['button'].clicked.connect(lambda: self._get_path(self.input_dir['line_edit'], is_directory=True))
        self.output_dir['button'].clicked.connect(lambda: self._get_path(self.output_dir['line_edit'], is_directory=True))

        self.run_button.clicked.connect(self.run_processing)
        self.processing_thread = None

    def _create_path_selector(self, label_text: str, is_directory: bool = False):
        layout = QHBoxLayout()
        label = QLabel(label_text)
        label.setFixedWidth(120)
        line_edit = QLineEdit()
        button = QPushButton("Browse...")

        layout.addWidget(label)
        layout.addWidget(line_edit)
        layout.addWidget(button)

        return {"layout": layout, "line_edit": line_edit, "button": button}

    def _get_path(self, line_edit: QLineEdit, is_directory: bool = False):
        if is_directory:
            path = QFileDialog.getExistingDirectory(self, "Select Directory", "")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select File", "")

        if path:
            line_edit.setText(path)

    def run_processing(self):
        params = {
            "rawgl_path": self.rawgl_path['line_edit'].text(),
            "shader_path": self.shader_path['line_edit'].text(),
            "input_dir": self.input_dir['line_edit'].text(),
            "output_dir": self.output_dir['line_edit'].text(),
            "workers": self.worker_spinbox.value(),
            "process_xy": self.xy_checkbox.isChecked(),
            "process_xz": self.xz_checkbox.isChecked(),
            "process_yz": self.yz_checkbox.isChecked(),
        }

        for name, path in params.items():
            if name.endswith("_dir") or name.endswith("_path"):
                if not path:
                    QMessageBox.warning(self, "Missing Input", f"Please provide the path for '{name}'.")
                    return

        self.log_output.clear()
        self.run_button.setEnabled(False)
        self.log_output.append("--- Starting Dask processing... ---")

        self.processing_thread = DaskRawGLThread(params)
        self.processing_thread.progress.connect(self.append_log)
        self.processing_thread.error.connect(self.on_processing_error)
        self.processing_thread.finished.connect(self.on_processing_finished)
        self.processing_thread.start()

    def append_log(self, text: str):
        self.log_output.append(text)

    def on_processing_error(self, message: str):
        self.log_output.append(f"\nERROR: {message}\n")
        self.on_processing_finished()

    def on_processing_finished(self):
        self.log_output.append("\n--- Dask processing finished. ---")
        self.run_button.setEnabled(True)
        self.processing_thread = None
