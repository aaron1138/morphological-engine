# src/gui/main_window.py

import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QProgressBar, QFileDialog, QMessageBox, QCheckBox,
    QTabWidget, QGroupBox, QRadioButton, QButtonGroup, QStackedWidget,
    QGridLayout, QFrame
)
from PySide6.QtCore import QSettings, Qt, Slot
from PySide6.QtGui import QIntValidator

from src.utils.config import app_config, Config, DEFAULT_NUM_WORKERS
from src.gui.processing_thread import ProcessingThread

class MainWindow(QWidget):
    """The main application window."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dask-Based Image Stack Processor")
        self.settings = QSettings("MyCompany", "DaskImageProcessor")
        self.processor_thread = None
        self.init_ui()
        self._autodetect_uvtools()
        self.load_settings()
        self._connect_signals()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # --- I/O Tab ---
        self.io_tab = QWidget()
        io_tab_layout = QVBoxLayout(self.io_tab)
        self.tab_widget.addTab(self.io_tab, "I/O")

        # --- Dask Tab ---
        self.dask_tab = QWidget()
        dask_tab_layout = QVBoxLayout(self.dask_tab)
        self.tab_widget.addTab(self.dask_tab, "Dask Settings")

        # --- I/O Section ---
        io_group = QGroupBox("I/O")
        io_layout = QVBoxLayout(io_group)
        io_tab_layout.addWidget(io_group)

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

        folder_mode_layout.addWidget(QLabel("Start Index:"), 2, 0)
        self.start_idx_edit = QLineEdit("0")
        folder_mode_layout.addWidget(self.start_idx_edit, 2, 1)

        folder_mode_layout.addWidget(QLabel("Stop Index:"), 3, 0)
        self.stop_idx_edit = QLineEdit()
        folder_mode_layout.addWidget(self.stop_idx_edit, 3, 1)
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
        io_tab_layout.addStretch(1)

        # --- Dask Settings Section ---
        dask_group = QGroupBox("Dask Settings")
        dask_layout = QGridLayout(dask_group)
        dask_tab_layout.addWidget(dask_group)

        dask_layout.addWidget(QLabel("Chunk Size (X, Y, Z):"), 0, 0)
        self.chunk_x_edit = QLineEdit("64")
        self.chunk_x_edit.setValidator(QIntValidator(1, 4096, self))
        self.chunk_y_edit = QLineEdit("64")
        self.chunk_y_edit.setValidator(QIntValidator(1, 4096, self))
        self.chunk_z_edit = QLineEdit("64")
        self.chunk_z_edit.setValidator(QIntValidator(1, 4096, self))
        chunk_layout = QHBoxLayout()
        chunk_layout.addWidget(self.chunk_x_edit)
        chunk_layout.addWidget(self.chunk_y_edit)
        chunk_layout.addWidget(self.chunk_z_edit)
        dask_layout.addLayout(chunk_layout, 0, 1)

        dask_layout.addWidget(QLabel("Thread Count:"), 1, 0)
        self.thread_count_edit = QLineEdit(str(DEFAULT_NUM_WORKERS))
        self.thread_count_edit.setValidator(QIntValidator(1, 128, self))
        dask_layout.addWidget(self.thread_count_edit, 1, 1)

        dask_tab_layout.addStretch(1)

        # --- Controls ---
        self.start_stop_button = QPushButton("Start Processing")
        self.start_stop_button.setMinimumHeight(40)
        main_layout.addWidget(self.start_stop_button)
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
        self.start_stop_button.clicked.connect(self.toggle_processing)

    def toggle_processing(self):
        if self.processor_thread and self.processor_thread.isRunning():
            self.processor_thread.stop()
            self.start_stop_button.setText("Stopping...")
            self.start_stop_button.setEnabled(False)
        else:
            self.start_processing()

    def start_processing(self):
        """Validates inputs and starts the processing thread."""
        try:
            self.save_settings()

            if app_config.input_mode == "folder":
                if not app_config.input_folder or not os.path.isdir(app_config.input_folder):
                    raise ValueError("Input folder must be a valid, existing directory.")
            elif app_config.input_mode == "uvtools":
                if not app_config.uvtools_path or not os.path.exists(app_config.uvtools_path):
                    raise ValueError("UVToolsCmd.exe path is not valid.")
                if not app_config.uvtools_temp_folder or not os.path.isdir(app_config.uvtools_temp_folder):
                    raise ValueError("Working Temp Folder must be a valid, existing directory.")
                if not app_config.uvtools_input_file or not os.path.exists(app_config.uvtools_input_file):
                    raise ValueError("Input Slice File is not valid.")

            self.set_ui_enabled(False)
            self.processor_thread = ProcessingThread(app_config=app_config)
            self.processor_thread.status_update.connect(self.update_status)
            self.processor_thread.progress_update.connect(self.progress_bar.setValue)
            self.processor_thread.error_signal.connect(self.show_error)
            self.processor_thread.finished_signal.connect(self.processing_finished)
            self.processor_thread.start()

        except Exception as e:
            self.show_error_message("Input Error", str(e))
            self.processing_finished()

    @Slot(str)
    def update_status(self, message):
        self.status_label.setText(f"Status: {message}")

    @Slot(str)
    def show_error(self, message):
        print(f"\n--- PROCESSING ERROR ---\n{message}\n------------------------\n")
        self.show_error_message("Processing Error", message, is_detailed=True)
        self.processing_finished()

    @Slot()
    def processing_finished(self):
        self.status_label.setText("Status: Finished or Stopped.")
        self.set_ui_enabled(True)
        self.processor_thread = None

    def show_error_message(self, title, text, is_detailed=False):
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle(title)
        if is_detailed:
            msg_box.setText("An error occurred. See details for more information.")
            msg_box.setDetailedText(text)
        else:
            msg_box.setText(text)
            msg_box.setTextInteractionFlags(Qt.TextSelectableByMouse)
        msg_box.exec()

    def set_ui_enabled(self, enabled):
        """Toggles the enabled state of all UI widgets."""
        self.tab_widget.setEnabled(enabled)
        self.start_stop_button.setEnabled(True)
        if not enabled:
            self.start_stop_button.setText("Stop Processing")
        else:
            self.start_stop_button.setText("Start Processing")

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
        """Loads settings from the global config object into the UI."""
        self.resize(self.settings.value("window_size", self.size()))
        self.move(self.settings.value("window_position", self.pos()))

        self.folder_mode_radio.setChecked(app_config.input_mode == "folder")
        self.uvtools_mode_radio.setChecked(app_config.input_mode == "uvtools")
        self.io_stacked_widget.setCurrentIndex(1 if app_config.input_mode == "uvtools" else 0)
        self.input_folder_edit.setText(app_config.input_folder)
        self.output_folder_edit.setText(app_config.output_folder)
        self.start_idx_edit.setText(str(app_config.start_index) if app_config.start_index is not None else "")
        self.stop_idx_edit.setText(str(app_config.stop_index) if app_config.stop_index is not None else "")
        self.uvtools_path_edit.setText(app_config.uvtools_path)
        self.uvtools_temp_folder_edit.setText(app_config.uvtools_temp_folder)
        self.uvtools_input_file_edit.setText(app_config.uvtools_input_file)
        self.output_prefix_edit.setText(app_config.output_file_prefix)
        self.uvtools_cleanup_checkbox.setChecked(app_config.uvtools_delete_temp_on_completion)
        self.uvtools_output_working_radio.setChecked(app_config.uvtools_output_location == "working_folder")
        self.uvtools_output_input_radio.setChecked(app_config.uvtools_output_location == "input_folder")

        self.chunk_x_edit.setText(str(app_config.dask_chunk_size_x))
        self.chunk_y_edit.setText(str(app_config.dask_chunk_size_y))
        self.chunk_z_edit.setText(str(app_config.dask_chunk_size_z))
        self.thread_count_edit.setText(str(app_config.dask_thread_count))

    def save_settings(self):
        """Saves current UI settings to the global config object and QSettings."""
        self.settings.setValue("window_size", self.size())
        self.settings.setValue("window_position", self.pos())

        app_config.input_mode = "uvtools" if self.uvtools_mode_radio.isChecked() else "folder"
        app_config.input_folder = self.input_folder_edit.text()
        app_config.output_folder = self.output_folder_edit.text()
        try:
            app_config.start_index = int(s) if (s := self.start_idx_edit.text()) else None
        except ValueError:
            app_config.start_index = None
        try:
            app_config.stop_index = int(s) if (s := self.stop_idx_edit.text()) else None
        except ValueError:
            app_config.stop_index = None
        app_config.uvtools_path = self.uvtools_path_edit.text()
        app_config.uvtools_temp_folder = self.uvtools_temp_folder_edit.text()
        app_config.uvtools_input_file = self.uvtools_input_file_edit.text()
        app_config.output_file_prefix = self.output_prefix_edit.text()
        app_config.uvtools_delete_temp_on_completion = self.uvtools_cleanup_checkbox.isChecked()
        app_config.uvtools_output_location = "input_folder" if self.uvtools_output_input_radio.isChecked() else "working_folder"

        try: app_config.dask_chunk_size_x = int(self.chunk_x_edit.text())
        except ValueError: pass
        try: app_config.dask_chunk_size_y = int(self.chunk_y_edit.text())
        except ValueError: pass
        try: app_config.dask_chunk_size_z = int(self.chunk_z_edit.text())
        except ValueError: pass
        try: app_config.dask_thread_count = int(self.thread_count_edit.text())
        except ValueError: pass

        app_config.save("app_config.json")

    def closeEvent(self, event):
        self.save_settings()
        if self.processor_thread and self.processor_thread.isRunning():
            # self.processor_thread.stop_processing() # To be implemented
            self.processor_thread.wait(5000)
        event.accept()
