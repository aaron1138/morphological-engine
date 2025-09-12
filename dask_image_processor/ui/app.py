# ui/app.py

import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QProgressBar, QFileDialog, QMessageBox, QCheckBox,
    QGroupBox, QRadioButton, QButtonGroup, QStackedWidget,
    QGridLayout, QFrame, QTabWidget
)
from PySide6.QtCore import Slot, Qt
from PySide6.QtGui import QIntValidator

from core.processing_thread import ProcessingThread
from config import app_config, Config, DEFAULT_WORKER_COUNT
from ui.pipeline_tab import PipelineTab

class DaskProcessorApp(QWidget):
    """The main application window for the Dask Image Processor."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dask Image Processor")
        self.processor_thread = None
        self.init_ui()
        self._connect_signals()
        self._autodetect_uvtools()
        self.load_settings() # Load settings on startup

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # --- Main Settings Tab ---
        self.main_settings_tab = QWidget()
        main_settings_layout = QVBoxLayout(self.main_settings_tab)
        self.tab_widget.addTab(self.main_settings_tab, "Settings")

        # --- I/O Section ---
        self.io_group = QGroupBox("I/O")
        io_layout = QVBoxLayout(self.io_group)

        input_mode_layout = QHBoxLayout()
        self.input_mode_group = QButtonGroup(self)
        self.folder_mode_radio = QRadioButton("Folder Input Mode")
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
        self.uvtools_cleanup_checkbox = QCheckBox("Delete Temporary Files on Completion")
        uvtools_mode_layout.addWidget(self.uvtools_cleanup_checkbox, 4, 1, 1, 2)
        self.io_stacked_widget.addWidget(uvtools_mode_widget)

        main_settings_layout.addWidget(self.io_group)

        # --- General Settings Section ---
        self.general_group = QGroupBox("General")
        general_layout = QGridLayout(self.general_group)
        general_layout.addWidget(QLabel("Dask Workers:"), 0, 0)
        self.worker_count_edit = QLineEdit(str(DEFAULT_WORKER_COUNT))
        self.worker_count_edit.setValidator(QIntValidator(1, 128, self))
        self.worker_count_edit.setFixedWidth(60)
        general_layout.addWidget(self.worker_count_edit, 0, 1)
        self.numba_checkbox = QCheckBox("Enable Numba JIT Acceleration")
        general_layout.addWidget(self.numba_checkbox, 1, 0, 1, 2)
        general_layout.setColumnStretch(2, 1)
        main_settings_layout.addWidget(self.general_group)

        # --- Config Management ---
        config_group = QGroupBox("Configuration")
        config_layout = QHBoxLayout(config_group)
        self.save_config_button = QPushButton("Save Config...")
        self.load_config_button = QPushButton("Load Config...")
        config_layout.addWidget(self.save_config_button)
        config_layout.addWidget(self.load_config_button)
        config_layout.addStretch(1)
        main_settings_layout.addWidget(config_group)

        main_settings_layout.addStretch(1)

        # --- Pipeline Tab ---
        self.pipeline_tab = PipelineTab()
        self.tab_widget.addTab(self.pipeline_tab, "Pipeline")

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
        # I/O
        self.input_folder_button.clicked.connect(lambda: self.browse_folder(self.input_folder_edit))
        self.output_folder_button.clicked.connect(lambda: self.browse_folder(self.output_folder_edit))
        self.uvtools_path_button.clicked.connect(lambda: self.browse_file(self.uvtools_path_edit, "Select UVToolsCmd.exe", "Executable Files (*.exe)"))
        self.uvtools_temp_folder_button.clicked.connect(lambda: self.browse_folder(self.uvtools_temp_folder_edit))
        self.uvtools_input_file_button.clicked.connect(lambda: self.browse_file(self.uvtools_input_file_edit, "Select Input Slice File"))
        self.input_mode_group.idClicked.connect(self.on_input_mode_changed)

        # Config
        self.save_config_button.clicked.connect(self._save_config_to_file)
        self.load_config_button.clicked.connect(self._load_config_from_file)

        # Main control
        self.start_stop_button.clicked.connect(self.toggle_processing)

    def load_settings(self):
        """Loads settings from the global config object into the UI."""
        self.folder_mode_radio.setChecked(app_config.input_mode == "folder")
        self.uvtools_mode_radio.setChecked(app_config.input_mode == "uvtools")
        self.io_stacked_widget.setCurrentIndex(1 if app_config.input_mode == "uvtools" else 0)
        self.input_folder_edit.setText(app_config.input_folder)
        self.output_folder_edit.setText(app_config.output_folder)
        self.uvtools_path_edit.setText(app_config.uvtools_path)
        self.uvtools_temp_folder_edit.setText(app_config.uvtools_temp_folder)
        self.uvtools_input_file_edit.setText(app_config.uvtools_input_file)
        self.uvtools_cleanup_checkbox.setChecked(app_config.uvtools_cleanup)
        self.worker_count_edit.setText(str(app_config.worker_count))
        self.numba_checkbox.setChecked(app_config.use_numba)

        self.pipeline_tab.load_pipeline()
        self.update_status("Configuration loaded.")

    def save_settings(self, save_path="app_config.json"):
        """Saves current UI settings to the global config object and to a file."""
        app_config.input_mode = "uvtools" if self.uvtools_mode_radio.isChecked() else "folder"
        app_config.input_folder = self.input_folder_edit.text()
        app_config.output_folder = self.output_folder_edit.text()
        app_config.uvtools_path = self.uvtools_path_edit.text()
        app_config.uvtools_temp_folder = self.uvtools_temp_folder_edit.text()
        app_config.uvtools_input_file = self.uvtools_input_file_edit.text()
        app_config.uvtools_cleanup = self.uvtools_cleanup_checkbox.isChecked()
        app_config.worker_count = int(self.worker_count_edit.text())
        app_config.use_numba = self.numba_checkbox.isChecked()

        # The pipeline tab now saves its own state directly to the app_config object

        app_config.save(save_path)
        self.update_status("Configuration saved.")

    def _save_config_to_file(self):
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Configuration", "custom_config.json", "JSON Files (*.json)")
        if filepath:
            self.save_settings(filepath)
            self.show_info_message("Success", f"Configuration saved to {filepath}")

    def _load_config_from_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Load Configuration", "", "JSON Files (*.json)")
        if filepath:
            global app_config
            app_config = Config.load(filepath)
            self.load_settings()
            self.show_info_message("Success", f"Configuration loaded from {filepath}")

    def _autodetect_uvtools(self):
        default_path = "C:\\Program Files\\UVTools\\UVToolsCmd.exe"
        if os.path.exists(default_path) and not self.uvtools_path_edit.text():
            self.uvtools_path_edit.setText(default_path)

    @Slot(int)
    def on_input_mode_changed(self, stack_index):
        self.io_stacked_widget.setCurrentIndex(stack_index)

    @Slot()
    def browse_folder(self, line_edit):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", line_edit.text())
        if folder: line_edit.setText(folder)

    @Slot()
    def browse_file(self, line_edit, caption, file_filter="All Files (*)"):
        file, _ = QFileDialog.getOpenFileName(self, caption, line_edit.text(), file_filter)
        if file: line_edit.setText(file)

    def validate_inputs(self):
        self.save_settings() # Save current entries to config object first
        if app_config.input_mode == 'folder':
            if not os.path.isdir(app_config.input_folder) or not os.path.isdir(app_config.output_folder):
                self.show_error_message("Input Error", "Please select valid input and output folders.")
                return False
        else: # uvtools mode
            if not os.path.isfile(app_config.uvtools_path) or not os.path.isdir(app_config.uvtools_temp_folder) or not os.path.isfile(app_config.uvtools_input_file):
                self.show_error_message("Input Error", "Please ensure all UVTools paths and files are valid.")
                return False
        return True

    @Slot()
    def toggle_processing(self):
        if self.processor_thread and self.processor_thread.isRunning():
            self.processor_thread.stop()
            self.start_stop_button.setText("Stopping...")
            self.start_stop_button.setEnabled(False)
        else:
            if not self.validate_inputs():
                return

            self.set_ui_enabled(False)
            self.progress_bar.setValue(0)
            self.processor_thread = ProcessingThread()
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
        self.set_ui_enabled(True)
        if self.processor_thread and not self.processor_thread.was_stopped():
             self.status_label.setText("Status: Finished.")
        else:
             self.status_label.setText("Status: Stopped.")
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
        msg_box.setDetailedText(text)
        msg_box.exec()

    def show_info_message(self, title, text):
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setWindowTitle(title)
        msg_box.setText(text)
        msg_box.exec()

    def set_ui_enabled(self, enabled):
        self.tab_widget.setEnabled(enabled)
        self.start_stop_button.setEnabled(True)
        self.start_stop_button.setText("Start Processing" if enabled else "Stop Processing")

    def closeEvent(self, event):
        self.save_settings() # Auto-save on exit
        if self.processor_thread and self.processor_thread.isRunning():
            self.processor_thread.stop()
            self.processor_thread.wait(5000)
        event.accept()
