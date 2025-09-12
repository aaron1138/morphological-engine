"""
Copyright (c) 2025 Jules

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QProgressBar, QFileDialog, QMessageBox, QCheckBox,
    QGroupBox, QRadioButton, QButtonGroup, QStackedWidget,
    QGridLayout, QFrame
)
from PySide6.QtCore import Slot, Qt, QSettings
from PySide6.QtGui import QIntValidator
import dask
from dask.distributed import Client, LocalCluster
import dask_engine

class DaskProcessorApp(QWidget):
    """The main application window for the Dask-based processor."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dask-based 3D Print Layer Processor")
        self.settings = QSettings("MyCompany", "DaskProcessorApp")
        self.dask_array = None
        self.client = None
        self.init_ui()
        self._autodetect_uvtools()
        self.load_settings()
        self._connect_signals()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        # --- I/O Section ---
        io_group = QGroupBox("I/O")
        io_layout = QVBoxLayout(io_group)

        input_mode_layout = QHBoxLayout()
        self.input_mode_group = QButtonGroup(self)
        self.folder_mode_radio = QRadioButton("Folder Input Mode")
        self.folder_mode_radio.setChecked(True)
        self.uvtools_mode_radio = QRadioButton("Use UVTools 5.x+")
        self.input_mode_group.addButton(self.folder_mode_radio, 0)
        self.input_mode_group.addButton(self.uvtools_mode_radio, 1)
        input_mode_layout.addWidget(self.folder_mode_radio)
        input_mode_layout.addWidget(self.uvtools_mode_radio)
        input_mode_layout.addStretch(1)
        io_layout.addLayout(input_mode_layout)

        self.io_stacked_widget = QStackedWidget()
        io_layout.addWidget(self.io_stacked_widget)

        # --- Folder Mode Widget ---
        folder_mode_widget = QWidget()
        folder_mode_layout = QGridLayout(folder_mode_widget)
        folder_mode_layout.addWidget(QLabel("Input Folder:"), 0, 0)
        self.input_folder_edit = QLineEdit()
        folder_mode_layout.addWidget(self.input_folder_edit, 0, 1)
        self.input_folder_button = QPushButton("Browse...")
        folder_mode_layout.addWidget(self.input_folder_button, 0, 2)

        folder_mode_layout.addWidget(QLabel("Output Folder:"), 1, 0)
        self.output_folder_edit = QLineEdit()
        folder_mode_layout.addWidget(self.output_folder_edit, 1, 1)
        self.output_folder_button = QPushButton("Browse...")
        folder_mode_layout.addWidget(self.output_folder_button, 1, 2)
        self.io_stacked_widget.addWidget(folder_mode_widget)

        # --- UVTools Mode Widget ---
        uvtools_mode_widget = QWidget()
        uvtools_mode_layout = QGridLayout(uvtools_mode_widget)
        uvtools_mode_layout.addWidget(QLabel("Path to UVToolsCmd.exe:"), 0, 0)
        self.uvtools_path_edit = QLineEdit()
        uvtools_mode_layout.addWidget(self.uvtools_path_edit, 0, 1)
        self.uvtools_path_button = QPushButton("Browse...")
        uvtools_mode_layout.addWidget(self.uvtools_path_button, 0, 2)

        uvtools_mode_layout.addWidget(QLabel("Working Temp Folder:"), 1, 0)
        self.uvtools_temp_folder_edit = QLineEdit()
        uvtools_mode_layout.addWidget(self.uvtools_temp_folder_edit, 1, 1)
        self.uvtools_temp_folder_button = QPushButton("Browse...")
        uvtools_mode_layout.addWidget(self.uvtools_temp_folder_button, 1, 2)

        uvtools_mode_layout.addWidget(QLabel("Input Slice File:"), 2, 0)
        self.uvtools_input_file_edit = QLineEdit()
        uvtools_mode_layout.addWidget(self.uvtools_input_file_edit, 2, 1)
        self.uvtools_input_file_button = QPushButton("Browse...")
        uvtools_mode_layout.addWidget(self.uvtools_input_file_button, 2, 2)

        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setFrameShadow(QFrame.Sunken)
        uvtools_mode_layout.addWidget(divider, 3, 0, 1, 3)

        uvtools_mode_layout.addWidget(QLabel("Output Completed Slice file:"), 4, 0)
        self.uvtools_output_location_group = QButtonGroup(self)
        self.uvtools_output_working_radio = QRadioButton("To Working Folder")
        self.uvtools_output_input_radio = QRadioButton("To Input Slice Folder")
        self.uvtools_output_location_group.addButton(self.uvtools_output_working_radio, 0)
        self.uvtools_output_location_group.addButton(self.uvtools_output_input_radio, 1)
        uvtools_mode_layout.addWidget(self.uvtools_output_working_radio, 4, 1)
        uvtools_mode_layout.addWidget(self.uvtools_output_input_radio, 5, 1)
        self.uvtools_output_working_radio.setChecked(True)

        uvtools_mode_layout.addWidget(QLabel("Output File Prefix:"), 6, 0)
        self.output_prefix_edit = QLineEdit("Dask_Processed_")
        uvtools_mode_layout.addWidget(self.output_prefix_edit, 6, 1, 1, 2)

        self.uvtools_cleanup_checkbox = QCheckBox("Delete Temporary Files on Completion")
        self.uvtools_cleanup_checkbox.setChecked(True)
        uvtools_mode_layout.addWidget(self.uvtools_cleanup_checkbox, 7, 1, 1, 2)

        self.io_stacked_widget.addWidget(uvtools_mode_widget)
        main_layout.addWidget(io_group)

        # --- Dask Settings Section ---
        dask_group = QGroupBox("Dask Settings")
        dask_layout = QVBoxLayout(dask_group)

        worker_layout = QHBoxLayout()
        worker_layout.addWidget(QLabel("Dask Workers:"))
        self.worker_count_edit = QLineEdit("4")
        self.worker_count_edit.setValidator(QIntValidator(1, 128, self))
        self.worker_count_edit.setFixedWidth(60)
        worker_layout.addWidget(self.worker_count_edit)
        worker_layout.addStretch(1)
        dask_layout.addLayout(worker_layout)

        self.numba_checkbox = QCheckBox("Enable Numba JIT Acceleration (Recommended)")
        self.numba_checkbox.setChecked(True)
        dask_layout.addWidget(self.numba_checkbox)

        main_layout.addWidget(dask_group)

        # --- Processing Section ---
        processing_group = QGroupBox("Processing")
        processing_layout = QVBoxLayout(processing_group)

        self.load_button = QPushButton("Load Image Stack to Dask Array")
        processing_layout.addWidget(self.load_button)

        self.dask_array_info_label = QLabel("Dask array not loaded.")
        processing_layout.addWidget(self.dask_array_info_label)

        self.extract_xy_button = QPushButton("Extract XY Planes")
        self.extract_xy_button.setEnabled(False)
        processing_layout.addWidget(self.extract_xy_button)

        self.extract_xz_button = QPushButton("Extract XZ Planes")
        self.extract_xz_button.setEnabled(False)
        processing_layout.addWidget(self.extract_xz_button)

        self.extract_yz_button = QPushButton("Extract YZ Planes")
        self.extract_yz_button.setEnabled(False)
        processing_layout.addWidget(self.extract_yz_button)

        main_layout.addWidget(processing_group)
        main_layout.addStretch(1)

        self.progress_bar = QProgressBar()
        main_layout.addWidget(self.progress_bar)
        self.status_label = QLabel("Status: Ready")
        self.status_label.setWordWrap(True)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.status_label)


    def _connect_signals(self):
        self.input_folder_button.clicked.connect(lambda: self.browse_folder(self.input_folder_edit))
        self.output_folder_button.clicked.connect(lambda: self.browse_folder(self.output_folder_edit))
        self.uvtools_path_button.clicked.connect(lambda: self.browse_file(self.uvtools_path_edit, "Select UVToolsCmd.exe", "Executable Files (*.exe)"))
        self.uvtools_temp_folder_button.clicked.connect(lambda: self.browse_folder(self.uvtools_temp_folder_edit))
        self.uvtools_input_file_button.clicked.connect(lambda: self.browse_file(self.uvtools_input_file_edit, "Select Input Slice File"))
        self.input_mode_group.idClicked.connect(self.on_input_mode_changed)

        self.load_button.clicked.connect(self.load_image_stack)
        self.extract_xy_button.clicked.connect(self.extract_xy_planes)
        self.extract_xz_button.clicked.connect(self.extract_xz_planes)
        self.extract_yz_button.clicked.connect(self.extract_yz_planes)

    def _autodetect_uvtools(self):
        """Checks for UVTools in the default location and populates the path if found."""
        default_path = "C:\\Program Files\\UVTools\\UVToolsCmd.exe"
        if os.path.exists(default_path):
            if not self.uvtools_path_edit.text():
                self.uvtools_path_edit.setText(default_path)

    def on_input_mode_changed(self, stack_index):
        self.io_stacked_widget.setCurrentIndex(stack_index)

    def browse_folder(self, line_edit):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", line_edit.text())
        if folder: line_edit.setText(folder)

    def browse_file(self, line_edit, caption, file_filter="All Files (*)"):
        file, _ = QFileDialog.getOpenFileName(self, caption, line_edit.text(), file_filter)
        if file: line_edit.setText(file)

    def load_settings(self):
        """Loads settings from QSettings into the UI."""
        self.resize(self.settings.value("window_size", self.size()))
        self.move(self.settings.value("window_position", self.pos()))

        self.input_folder_edit.setText(self.settings.value("input_folder", ""))
        self.output_folder_edit.setText(self.settings.value("output_folder", ""))
        self.uvtools_path_edit.setText(self.settings.value("uvtools_path", ""))
        self.uvtools_temp_folder_edit.setText(self.settings.value("uvtools_temp_folder", ""))
        self.uvtools_input_file_edit.setText(self.settings.value("uvtools_input_file", ""))
        self.worker_count_edit.setText(self.settings.value("worker_count", "4"))
        self.numba_checkbox.setChecked(self.settings.value("use_numba", "true") == "true")

    def save_settings(self):
        """Saves current UI settings to QSettings."""
        self.settings.setValue("window_size", self.size())
        self.settings.setValue("window_position", self.pos())
        self.settings.setValue("input_folder", self.input_folder_edit.text())
        self.settings.setValue("output_folder", self.output_folder_edit.text())
        self.settings.setValue("uvtools_path", self.uvtools_path_edit.text())
        self.settings.setValue("uvtools_temp_folder", self.uvtools_temp_folder_edit.text())
        self.settings.setValue("uvtools_input_file", self.uvtools_input_file_edit.text())
        self.settings.setValue("worker_count", self.worker_count_edit.text())
        self.settings.setValue("use_numba", "true" if self.numba_checkbox.isChecked() else "false")

    def closeEvent(self, event):
        self.save_settings()
        if self.client:
            self.client.close()
        event.accept()

    def _setup_dask_client(self):
        if self.client:
            self.client.close()

        n_workers = int(self.worker_count_edit.text())
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
        self.client = Client(cluster)

        if self.numba_checkbox.isChecked():
            dask.config.set(scheduler='threads')
        else:
            dask.config.set(scheduler='processes')


    @Slot()
    def load_image_stack(self):
        self._setup_dask_client()
        input_folder = self.input_folder_edit.text()
        if not input_folder:
            self.show_error_message("Input Error", "Please select an input folder.")
            return

        self.status_label.setText("Status: Loading image stack...")
        try:
            self.dask_array = dask_engine.load_image_stack(input_folder)
            self.dask_array_info_label.setText(f"Dask array loaded: {self.dask_array.shape}, chunks: {self.dask_array.chunksize}")
            self.extract_xy_button.setEnabled(True)
            self.extract_xz_button.setEnabled(True)
            self.extract_yz_button.setEnabled(True)
            self.status_label.setText("Status: Ready")
        except Exception as e:
            self.show_error_message("Error", f"Failed to load image stack:\n{e}")
            self.status_label.setText("Status: Error")

    def _extract_planes(self, plane_type):
        if self.dask_array is None:
            self.show_error_message("Error", "No Dask array loaded.")
            return

        output_folder = self.output_folder_edit.text()
        if not output_folder:
            self.show_error_message("Input Error", "Please select an output folder.")
            return

        self.status_label.setText(f"Status: Extracting {plane_type.upper()} planes...")
        try:
            dask_engine.save_planes(self.dask_array, output_folder, plane_type)
            self.status_label.setText(f"Status: Finished extracting {plane_type.upper()} planes.")
            self.show_info_message("Success", f"{plane_type.upper()} planes extracted successfully to:\n{os.path.join(output_folder, plane_type + '_planes')}")
        except Exception as e:
            self.show_error_message("Error", f"Failed to extract {plane_type.upper()} planes:\n{e}")
            self.status_label.setText("Status: Error")

    @Slot()
    def extract_xy_planes(self):
        self._extract_planes("xy")

    @Slot()
    def extract_xz_planes(self):
        self._extract_planes("xz")

    @Slot()
    def extract_yz_planes(self):
        self._extract_planes("yz")

    def show_error_message(self, title, text):
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle(title)
        msg_box.setText(text)
        msg_box.exec()

    def show_info_message(self, title, text):
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setWindowTitle(title)
        msg_box.setText(text)
        msg_box.exec()
