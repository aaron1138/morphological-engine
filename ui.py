# ui.py

import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QMessageBox, QRadioButton,
    QButtonGroup, QStackedWidget, QGroupBox,
    QGridLayout, QFrame, QCheckBox, QProgressBar, QApplication
)
from PySide6.QtGui import QIntValidator
from PySide6.QtCore import Slot, QSettings, Qt, QThread, Signal

from config import app_config
from dask_engine import DaskEngine
import uvtools_wrapper

class ProcessingThread(QThread):
    status_update = Signal(str)
    progress_update = Signal(int)
    error_signal = Signal(str)
    finished_signal = Signal()

    def __init__(self, dask_engine):
        super().__init__()
        self.dask_engine = dask_engine

    def run(self):
        try:
            input_path = ""
            if app_config.input_mode == "folder":
                input_path = app_config.input_folder
            elif app_config.input_mode == "uvtools":
                self.status_update.emit("Extracting layers with UVTools...")
                input_path = uvtools_wrapper.extract_layers(
                    app_config.uvtools_path,
                    app_config.uvtools_input_file,
                    app_config.uvtools_temp_folder
                )
                self.status_update.emit("Layers extracted. Loading into Dask...")

            dask_array = self.dask_engine.load_images_to_dask_array(input_path)
            self.status_update.emit(f"Dask array loaded with shape {dask_array.shape} and chunk size {dask_array.chunksize}")

        except Exception as e:
            self.error_signal.emit(str(e))
        finally:
            self.finished_signal.emit()

class ModularApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Modular Dask Engine")
        self.settings = QSettings("MyCompany", "ModularDaskApp")
        self.dask_engine = None
        self.processing_thread = None
        self.init_ui()
        self.load_settings()
        self._connect_signals()
        self._autodetect_uvtools()
        self.toggle_processing()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        # I/O Section
        self.io_group = QGroupBox("I/O")
        io_layout = QVBoxLayout(self.io_group)

        input_mode_layout = QHBoxLayout()
        self.input_mode_group = QButtonGroup(self)
        self.folder_mode_radio = QRadioButton("Folder Input Mode")
        self.folder_mode_radio.setChecked(True)
        self.uvtools_mode_radio = QRadioButton("Use UVTools")
        self.input_mode_group.addButton(self.folder_mode_radio, 0)
        self.input_mode_group.addButton(self.uvtools_mode_radio, 1)
        input_mode_layout.addWidget(self.folder_mode_radio)
        input_mode_layout.addWidget(self.uvtools_mode_radio)
        input_mode_layout.addStretch(1)
        io_layout.addLayout(input_mode_layout)

        self.io_stacked_widget = QStackedWidget()
        io_layout.addWidget(self.io_stacked_widget)

        # Folder Mode Widget
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

        # UVTools Mode Widget
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

        self.uvtools_cleanup_checkbox = QCheckBox("Delete Temporary Files on Completion")
        self.uvtools_cleanup_checkbox.setChecked(True)
        uvtools_mode_layout.addWidget(self.uvtools_cleanup_checkbox, 6, 1, 1, 2)

        self.io_stacked_widget.addWidget(uvtools_mode_widget)

        main_layout.addWidget(self.io_group)

        # Dask Engine Section
        self.dask_group = QGroupBox("Dask Engine")
        dask_layout = QHBoxLayout(self.dask_group)
        dask_layout.addWidget(QLabel("Thread Count:"))
        self.thread_count_edit = QLineEdit(str(app_config.thread_count))
        self.thread_count_edit.setValidator(QIntValidator(1, 128, self))
        dask_layout.addWidget(self.thread_count_edit)
        main_layout.addWidget(self.dask_group)

        # Run Button
        self.run_button = QPushButton("Run")
        self.run_button.setMinimumHeight(40)
        main_layout.addWidget(self.run_button)

        self.progress_bar = QProgressBar()
        main_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Status: Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.status_label)

    def _connect_signals(self):
        self.input_folder_button.clicked.connect(lambda: self.browse_folder(self.input_folder_edit))
        self.output_folder_button.clicked.connect(lambda: self.browse_folder(self.output_folder_edit))
        self.uvtools_path_button.clicked.connect(lambda: self.browse_file(self.uvtools_path_edit, "Select UVToolsCmd.exe", "Executable Files (*.exe)"))
        self.uvtools_temp_folder_button.clicked.connect(lambda: self.browse_folder(self.uvtools_temp_folder_edit))
        self.uvtools_input_file_button.clicked.connect(lambda: self.browse_file(self.uvtools_input_file_edit, "Select Input Slice File"))
        self.input_mode_group.idClicked.connect(self.on_input_mode_changed)
        self.run_button.clicked.connect(self.toggle_processing)

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
        if folder:
            line_edit.setText(folder)

    def browse_file(self, line_edit, caption, file_filter="All Files (*)"):
        file, _ = QFileDialog.getOpenFileName(self, caption, line_edit.text(), file_filter)
        if file:
            line_edit.setText(file)

    def load_settings(self):
        self.input_folder_edit.setText(self.settings.value("input_folder", "test_images"))
        self.output_folder_edit.setText(self.settings.value("output_folder", ""))
        self.uvtools_path_edit.setText(self.settings.value("uvtools_path", ""))
        self.uvtools_temp_folder_edit.setText(self.settings.value("uvtools_temp_folder", ""))
        self.uvtools_input_file_edit.setText(self.settings.value("uvtools_input_file", ""))
        self.thread_count_edit.setText(self.settings.value("thread_count", str(app_config.thread_count)))
        self.uvtools_cleanup_checkbox.setChecked(self.settings.value("uvtools_cleanup", True, type=bool))
        self.uvtools_output_working_radio.setChecked(self.settings.value("uvtools_output_to_working", True, type=bool))
        self.uvtools_output_input_radio.setChecked(self.settings.value("uvtools_output_to_input", False, type=bool))


    def save_settings(self):
        self.settings.setValue("input_folder", self.input_folder_edit.text())
        self.settings.setValue("output_folder", self.output_folder_edit.text())
        self.settings.setValue("uvtools_path", self.uvtools_path_edit.text())
        self.settings.setValue("uvtools_temp_folder", self.uvtools_temp_folder_edit.text())
        self.settings.setValue("uvtools_input_file", self.uvtools_input_file_edit.text())
        self.settings.setValue("thread_count", self.thread_count_edit.text())
        self.settings.setValue("uvtools_cleanup", self.uvtools_cleanup_checkbox.isChecked())
        self.settings.setValue("uvtools_output_to_working", self.uvtools_output_working_radio.isChecked())
        self.settings.setValue("uvtools_output_to_input", self.uvtools_output_input_radio.isChecked())

        # Save to config object
        app_config.input_mode = "uvtools" if self.uvtools_mode_radio.isChecked() else "folder"
        app_config.input_folder = self.input_folder_edit.text()
        app_config.output_folder = self.output_folder_edit.text()
        app_config.uvtools_path = self.uvtools_path_edit.text()
        app_config.uvtools_temp_folder = self.uvtools_temp_folder_edit.text()
        app_config.uvtools_input_file = self.uvtools_input_file_edit.text()
        app_config.thread_count = int(self.thread_count_edit.text())
        app_config.uvtools_cleanup = self.uvtools_cleanup_checkbox.isChecked()
        app_config.uvtools_output_location = "input_folder" if self.uvtools_output_input_radio.isChecked() else "working_folder"

    def set_ui_enabled(self, enabled):
        """Toggles the enabled state of all UI widgets."""
        self.io_group.setEnabled(enabled)
        self.dask_group.setEnabled(enabled)
        self.run_button.setEnabled(enabled)
        if not enabled:
            self.run_button.setText("Processing...")
        else:
            self.run_button.setText("Run")

    def toggle_processing(self):
        if self.processing_thread and self.processing_thread.isRunning():
            # This is a placeholder for a proper stop mechanism
            self.status_label.setText("Status: Stopping... (not implemented)")
        else:
            self.start_processing()

    def start_processing(self):
        self.save_settings()
        self.set_ui_enabled(False)

        if self.dask_engine is None:
            self.dask_engine = DaskEngine(app_config.thread_count)
            self.dask_engine.start_dask_client()

        self.processing_thread = ProcessingThread(self.dask_engine)
        self.processing_thread.status_update.connect(self.update_status)
        self.processing_thread.progress_update.connect(self.progress_bar.setValue)
        self.processing_thread.error_signal.connect(self.show_error)
        self.processing_thread.finished_signal.connect(self.processing_finished)
        self.processing_thread.start()

    @Slot(str)
    def update_status(self, message):
        self.status_label.setText(f"Status: {message}")

    @Slot(str)
    def show_error(self, message):
        QMessageBox.critical(self, "Error", message)
        self.processing_finished()

    @Slot()
    def processing_finished(self):
        self.set_ui_enabled(True)
        self.processing_thread = None


    def closeEvent(self, event):
        if self.dask_engine:
            self.dask_engine.stop_dask_client()
        self.save_settings()
        event.accept()
