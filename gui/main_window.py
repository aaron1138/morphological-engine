# gui/main_window.py
import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QGroupBox, QRadioButton, QButtonGroup,
    QStackedWidget, QGridLayout, QProgressBar, QMessageBox
)
from PySide6.QtGui import QIntValidator
from PySide6.QtCore import QSettings, Slot

from utils.config import Config
from gui.processing_thread import ProcessingThread

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dask Image Processor")
        self.settings = QSettings("MyCompany", "DaskImageProcessor")
        self.config = Config()
        self.processing_thread = None
        self.init_ui()
        self.load_settings()
        self.connect_signals()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        # --- I/O Section ---
        io_group = QGroupBox("I/O")
        io_layout = QVBoxLayout(io_group)

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
        self.io_stacked_widget.addWidget(uvtools_mode_widget)
        main_layout.addWidget(io_group)

        # --- General Settings Section ---
        general_group = QGroupBox("General")
        general_layout = QHBoxLayout(general_group)
        general_layout.addWidget(QLabel("Dask Worker Count:"))
        self.thread_count_edit = QLineEdit("4")
        self.thread_count_edit.setValidator(QIntValidator(1, 128, self))
        general_layout.addWidget(self.thread_count_edit)
        general_layout.addStretch(1)
        main_layout.addWidget(general_group)

        # --- Action Buttons & Status ---
        self.start_stop_button = QPushButton("Start Processing")
        self.progress_bar = QProgressBar()
        self.status_label = QLabel("Status: Ready")
        main_layout.addWidget(self.start_stop_button)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.status_label)

        main_layout.addStretch(1)

    def connect_signals(self):
        self.input_folder_button.clicked.connect(lambda: self.browse_folder(self.input_folder_edit))
        self.output_folder_button.clicked.connect(lambda: self.browse_folder(self.output_folder_edit))
        self.uvtools_path_button.clicked.connect(lambda: self.browse_file(self.uvtools_path_edit, "Select UVToolsCmd.exe"))
        self.uvtools_temp_folder_button.clicked.connect(lambda: self.browse_folder(self.uvtools_temp_folder_edit))
        self.uvtools_input_file_button.clicked.connect(lambda: self.browse_file(self.uvtools_input_file_edit, "Select Input Slice File"))
        self.input_mode_group.idClicked.connect(self.io_stacked_widget.setCurrentIndex)
        self.start_stop_button.clicked.connect(self.toggle_processing)

    def browse_folder(self, line_edit):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", line_edit.text())
        if folder:
            line_edit.setText(folder)

    def browse_file(self, line_edit, caption):
        file, _ = QFileDialog.getOpenFileName(self, "Select a File", line_edit.text())
        if file:
            line_edit.setText(file)

    def load_settings(self):
        if os.path.exists("config.json"):
            self.config = Config.load("config.json")

        self.input_folder_edit.setText(self.config.input_folder)
        self.output_folder_edit.setText(self.config.output_folder)
        self.uvtools_path_edit.setText(self.config.uvtools_path)
        self.uvtools_input_file_edit.setText(self.config.uvtools_input_file)
        self.uvtools_temp_folder_edit.setText(self.config.uvtools_temp_folder)
        self.thread_count_edit.setText(str(self.config.thread_count))

    def save_settings(self):
        self.config.input_folder = self.input_folder_edit.text()
        self.config.output_folder = self.output_folder_edit.text()
        self.config.uvtools_path = self.uvtools_path_edit.text()
        self.config.uvtools_input_file = self.uvtools_input_file_edit.text()
        self.config.uvtools_temp_folder = self.uvtools_temp_folder_edit.text()
        self.config.thread_count = int(self.thread_count_edit.text())
        self.config.input_mode = "uvtools" if self.uvtools_mode_radio.isChecked() else "folder"
        self.config.save("config.json")

    def toggle_processing(self):
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop()
            self.start_stop_button.setText("Stopping...")
            self.start_stop_button.setEnabled(False)
        else:
            self.start_processing()

    def start_processing(self):
        self.save_settings()
        self.set_ui_enabled(False)
        self.processing_thread = ProcessingThread(self.config)
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

    def set_ui_enabled(self, enabled):
        self.start_stop_button.setEnabled(True)
        if not enabled:
            self.start_stop_button.setText("Stop Processing")
        else:
            self.start_stop_button.setText("Start Processing")

        self.io_group.setEnabled(enabled)
        self.general_group.setEnabled(enabled)

    def closeEvent(self, event):
        self.save_settings()
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop()
            self.processing_thread.wait()
        event.accept()
