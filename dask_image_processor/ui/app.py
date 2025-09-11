# ui/app.py

import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QProgressBar, QFileDialog, QMessageBox, QCheckBox,
    QGroupBox, QRadioButton, QButtonGroup, QStackedWidget,
    QGridLayout, QFrame
)
from PySide6.QtCore import Slot, Qt
from PySide6.QtGui import QIntValidator

from core.processing_thread import ProcessingThread

class DaskProcessorApp(QWidget):
    """The main application window for the Dask Image Processor."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dask Image Processor")
        self.processor_thread = None
        self.init_ui()
        self._connect_signals()
        self._autodetect_uvtools()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        # --- I/O Section ---
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

        uvtools_mode_layout.addWidget(QLabel("Output File Prefix:"), 3, 0)
        self.output_prefix_edit = QLineEdit("Dask_Processed_")
        uvtools_mode_layout.addWidget(self.output_prefix_edit, 3, 1, 1, 2)

        self.uvtools_cleanup_checkbox = QCheckBox("Delete Temporary Files on Completion")
        self.uvtools_cleanup_checkbox.setChecked(True)
        uvtools_mode_layout.addWidget(self.uvtools_cleanup_checkbox, 4, 1, 1, 2)

        self.io_stacked_widget.addWidget(uvtools_mode_widget)
        main_layout.addWidget(self.io_group)

        # --- General Settings Section ---
        self.general_group = QGroupBox("General")
        general_layout = QGridLayout(self.general_group)

        general_layout.addWidget(QLabel("Dask Workers:"), 0, 0)
        self.worker_count_edit = QLineEdit(str(os.cpu_count() or 4))
        self.worker_count_edit.setValidator(QIntValidator(1, 128, self))
        self.worker_count_edit.setFixedWidth(60)
        general_layout.addWidget(self.worker_count_edit, 0, 1)

        self.numba_checkbox = QCheckBox("Enable Numba JIT Acceleration")
        self.numba_checkbox.setToolTip("Uses a Just-In-Time compiler (Numba) for certain operations, which can be significantly faster.")
        self.numba_checkbox.setChecked(True)
        general_layout.addWidget(self.numba_checkbox, 1, 0, 1, 2)

        general_layout.setColumnStretch(2, 1)
        main_layout.addWidget(self.general_group)

        main_layout.addStretch(1)

        # --- Controls and Status ---
        self.start_stop_button = QPushButton("Start Processing")
        self.start_stop_button.setMinimumHeight(40)
        main_layout.addWidget(self.start_stop_button)
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
        self.start_stop_button.clicked.connect(self.toggle_processing)

    def _autodetect_uvtools(self):
        default_path = "C:\\Program Files\\UVTools\\UVToolsCmd.exe"
        if os.path.exists(default_path):
            if not self.uvtools_path_edit.text():
                self.uvtools_path_edit.setText(default_path)

    @Slot(int)
    def on_input_mode_changed(self, stack_index):
        self.io_stacked_widget.setCurrentIndex(stack_index)

    @Slot()
    def browse_folder(self, line_edit):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", line_edit.text())
        if folder:
            line_edit.setText(folder)

    @Slot()
    def browse_file(self, line_edit, caption, file_filter="All Files (*)"):
        file, _ = QFileDialog.getOpenFileName(self, caption, line_edit.text(), file_filter)
        if file:
            line_edit.setText(file)

    def gather_config(self):
        config = {
            "input_mode": "uvtools" if self.uvtools_mode_radio.isChecked() else "folder",
            "input_folder": self.input_folder_edit.text(),
            "output_folder": self.output_folder_edit.text(),
            "uvtools_path": self.uvtools_path_edit.text(),
            "uvtools_temp_folder": self.uvtools_temp_folder_edit.text(),
            "uvtools_input_file": self.uvtools_input_file_edit.text(),
            "uvtools_cleanup": self.uvtools_cleanup_checkbox.isChecked(),
            "worker_count": int(self.worker_count_edit.text()),
            "use_numba": self.numba_checkbox.isChecked(),
        }
        return config

    def validate_inputs(self, config):
        if config['input_mode'] == 'folder':
            if not os.path.isdir(config['input_folder']):
                self.show_error_message("Input Error", "Please select a valid input folder.")
                return False
            if not os.path.isdir(config['output_folder']):
                self.show_error_message("Input Error", "Please select a valid output folder.")
                return False
        else: # uvtools mode
            if not os.path.isfile(config['uvtools_path']):
                self.show_error_message("Input Error", "Please select a valid UVToolsCmd.exe path.")
                return False
            if not os.path.isdir(config['uvtools_temp_folder']):
                self.show_error_message("Input Error", "Please select a valid temporary folder.")
                return False
            if not os.path.isfile(config['uvtools_input_file']):
                self.show_error_message("Input Error", "Please select a valid input slice file.")
                return False
        return True

    @Slot()
    def toggle_processing(self):
        if self.processor_thread and self.processor_thread.isRunning():
            self.processor_thread.stop()
            self.start_stop_button.setText("Stopping...")
            self.start_stop_button.setEnabled(False)
        else:
            config = self.gather_config()
            if not self.validate_inputs(config):
                return

            self.set_ui_enabled(False)
            self.progress_bar.setValue(0)
            self.processor_thread = ProcessingThread(config)
            self.processor_thread.status_update.connect(self.update_status)
            self.processor_thread.progress_update.connect(self.progress_bar.setValue)
            self.processor_thread.error_signal.connect(self.on_processing_error)
            self.processor_thread.finished_signal.connect(self.on_processing_finished)
            self.processor_thread.start()

    @Slot(str)
    def update_status(self, message):
        self.status_label.setText(f"Status: {message}")

    @Slot()
    def on_processing_finished(self):
        self.status_label.setText("Status: Finished or Stopped.")
        self.set_ui_enabled(True)
        self.processor_thread = None

    @Slot(str)
    def on_processing_error(self, message):
        self.show_error_message("Processing Error", message)
        self.on_processing_finished()

    def show_error_message(self, title, text):
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle(title)
        msg_box.setText(text)
        msg_box.exec()

    def set_ui_enabled(self, enabled):
        self.io_group.setEnabled(enabled)
        self.general_group.setEnabled(enabled)
        self.start_stop_button.setEnabled(True)
        self.start_stop_button.setText("Start Processing" if enabled else "Stop Processing")

    def closeEvent(self, event):
        if self.processor_thread and self.processor_thread.isRunning():
            self.processor_thread.stop()
            self.processor_thread.wait(5000) # Wait up to 5s for thread to finish
        event.accept()
