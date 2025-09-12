import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QProgressBar, QFileDialog, QMessageBox,
    QGroupBox, QRadioButton, QButtonGroup, QStackedWidget,
    QGridLayout, QFrame
)
from PySide6.QtCore import Slot, Qt
from PySide6.QtGui import QIntValidator

from processing_thread import ProcessingThread

class ImageProcessorApp(QWidget):
    """The main application window for the Dask Image Engine."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dask Image Engine")
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

        uvtools_mode_layout.addWidget(QLabel("Output Folder:"), 3, 0)
        self.uvtools_output_folder_edit = QLineEdit()
        uvtools_mode_layout.addWidget(self.uvtools_output_folder_edit, 3, 1)
        self.uvtools_output_folder_button = QPushButton("Browse...")
        uvtools_mode_layout.addWidget(self.uvtools_output_folder_button, 3, 2)

        self.io_stacked_widget.addWidget(uvtools_mode_widget)

        main_layout.addWidget(self.io_group)

        # --- General Settings Section ---
        self.general_group = QGroupBox("General")
        general_layout = QGridLayout(self.general_group)

        general_layout.addWidget(QLabel("Dask Threads:"), 0, 0)
        self.thread_count_edit = QLineEdit("4")
        self.thread_count_edit.setValidator(QIntValidator(1, 128, self))
        self.thread_count_edit.setFixedWidth(60)
        general_layout.addWidget(self.thread_count_edit, 0, 1)

        general_layout.addWidget(QLabel("Chunk Size (X, Y, Z):"), 1, 0)
        chunk_layout = QHBoxLayout()
        self.chunk_x_edit = QLineEdit("64")
        self.chunk_y_edit = QLineEdit("64")
        self.chunk_z_edit = QLineEdit("64")
        self.chunk_x_edit.setValidator(QIntValidator(1, 4096, self))
        self.chunk_y_edit.setValidator(QIntValidator(1, 4096, self))
        self.chunk_z_edit.setValidator(QIntValidator(1, 4096, self))
        chunk_layout.addWidget(self.chunk_x_edit)
        chunk_layout.addWidget(self.chunk_y_edit)
        chunk_layout.addWidget(self.chunk_z_edit)
        general_layout.addLayout(chunk_layout, 1, 1)

        general_layout.setColumnStretch(2, 1)
        main_layout.addWidget(self.general_group)

        main_layout.addStretch(1)

        # --- Processing Controls ---
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
        self.uvtools_output_folder_button.clicked.connect(lambda: self.browse_folder(self.uvtools_output_folder_edit))
        self.input_mode_group.idClicked.connect(self.on_input_mode_changed)
        self.start_stop_button.clicked.connect(self.toggle_processing)

    def _autodetect_uvtools(self):
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

    def toggle_processing(self):
        if self.processor_thread and self.processor_thread.isRunning():
            self.processor_thread.stop()
            self.start_stop_button.setText("Stopping...")
            self.start_stop_button.setEnabled(False)
        else:
            self.start_processing()

    def start_processing(self):
        try:
            config = self.get_config()
            self.set_ui_enabled(False)
            self.processor_thread = ProcessingThread(config)
            self.processor_thread.status_update.connect(self.update_status)
            self.processor_thread.progress_update.connect(self.progress_bar.setValue)
            self.processor_thread.error_signal.connect(self.show_error)
            self.processor_thread.finished_signal.connect(self.processing_finished)
            self.processor_thread.start()

        except Exception as e:
            self.show_error_message("Configuration Error", str(e))
            self.set_ui_enabled(True)

    def get_config(self):
        config = {
            "mode": "uvtools" if self.uvtools_mode_radio.isChecked() else "folder",
            "num_threads": int(self.thread_count_edit.text()),
            "chunk_size": (
                int(self.chunk_z_edit.text()),
                int(self.chunk_y_edit.text()),
                int(self.chunk_x_edit.text())
            )
        }

        if config["mode"] == "folder":
            config["input_folder"] = self.input_folder_edit.text()
            config["output_folder"] = self.output_folder_edit.text()
            if not os.path.isdir(config["input_folder"]):
                raise ValueError("Input folder is not a valid directory.")
            if not os.path.isdir(config["output_folder"]):
                raise ValueError("Output folder is not a valid directory.")
        else: # uvtools mode
            config["uvtools_path"] = self.uvtools_path_edit.text()
            config["uvtools_temp_folder"] = self.uvtools_temp_folder_edit.text()
            config["uvtools_input_file"] = self.uvtools_input_file_edit.text()
            config["output_folder"] = self.uvtools_output_folder_edit.text()
            config["input_folder"] = "" # Will be set by the thread after extraction
            if not os.path.exists(config["uvtools_path"]):
                raise ValueError("UVToolsCmd.exe path is not valid.")
            if not os.path.isdir(config["uvtools_temp_folder"]):
                raise ValueError("UVTools temp folder is not a valid directory.")
            if not os.path.exists(config["uvtools_input_file"]):
                raise ValueError("Input slice file is not valid.")
            if not os.path.isdir(config["output_folder"]):
                raise ValueError("Output folder is not a valid directory.")
        return config

    @Slot(str)
    def update_status(self, message):
        self.status_label.setText(f"Status: {message}")

    @Slot(str)
    def show_error(self, message):
        self.show_error_message("Processing Error", message)

    @Slot()
    def processing_finished(self):
        self.set_ui_enabled(True)
        self.processor_thread = None
        self.progress_bar.setValue(0)

    def set_ui_enabled(self, enabled):
        self.io_group.setEnabled(enabled)
        self.general_group.setEnabled(enabled)
        self.start_stop_button.setEnabled(True)
        if not enabled:
            self.start_stop_button.setText("Stop Processing")
        else:
            self.start_stop_button.setText("Start Processing")

    def show_error_message(self, title, text):
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle(title)
        msg_box.setText(text)
        msg_box.exec()

    def closeEvent(self, event):
        if self.processor_thread and self.processor_thread.isRunning():
            self.processor_thread.stop()
            self.processor_thread.wait(5000) # Wait up to 5 seconds
        event.accept()
