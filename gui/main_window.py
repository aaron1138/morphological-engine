import os
import sys
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QProgressBar, QFileDialog, QMessageBox, QCheckBox,
    QGroupBox, QRadioButton, QButtonGroup, QStackedWidget,
    QGridLayout, QFrame
)
from PySide6.QtCore import Qt, QSettings, Slot
from PySide6.QtGui import QIntValidator

# Import local modules
from utils.config import app_config, Config, DEFAULT_DASK_WORKERS
from gui.processing_thread import ProcessingThread

class DaskEngineApp(QWidget):
    """The main application window for the Dask-based image processing engine."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dask-Based 3D Print Plane Extractor")
        # Use QSettings to remember window size and position
        self.settings = QSettings("MyCompany", "DaskEngineApp")
        self.processor_thread = None

        self.init_ui()
        self._connect_signals()
        self._autodetect_uvtools()
        self.load_settings()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        # --- I/O Section ---
        io_group = QGroupBox("Input & Output")
        io_layout = QVBoxLayout(io_group)
        main_layout.addWidget(io_group)

        # Input Mode Selection (Folder vs UVTools)
        input_mode_layout = QHBoxLayout()
        self.input_mode_group = QButtonGroup(self)
        self.folder_mode_radio = QRadioButton("Folder Input Mode")
        self.folder_mode_radio.setChecked(True)
        self.uvtools_mode_radio = QRadioButton("UVTools Input Mode")
        self.input_mode_group.addButton(self.folder_mode_radio, 0)
        self.input_mode_group.addButton(self.uvtools_mode_radio, 1)
        input_mode_layout.addWidget(self.folder_mode_radio)
        input_mode_layout.addWidget(self.uvtools_mode_radio)
        input_mode_layout.addStretch(1)
        io_layout.addLayout(input_mode_layout)

        self.io_stacked_widget = QStackedWidget()
        io_layout.addWidget(self.io_stacked_widget)

        # --- Panel 1: Folder Mode ---
        folder_mode_widget = QWidget()
        folder_layout = QGridLayout(folder_mode_widget)
        folder_layout.addWidget(QLabel("Input Folder (PNGs):"), 0, 0)
        self.input_folder_edit = QLineEdit()
        folder_layout.addWidget(self.input_folder_edit, 0, 1)
        self.input_folder_button = QPushButton("Browse...")
        folder_layout.addWidget(self.input_folder_button, 0, 2)
        self.io_stacked_widget.addWidget(folder_mode_widget)

        # --- Panel 2: UVTools Mode ---
        uvtools_mode_widget = QWidget()
        uvtools_layout = QGridLayout(uvtools_mode_widget)
        uvtools_layout.addWidget(QLabel("Path to UVToolsCmd.exe:"), 0, 0)
        self.uvtools_path_edit = QLineEdit()
        uvtools_layout.addWidget(self.uvtools_path_edit, 0, 1)
        self.uvtools_path_button = QPushButton("Browse...")
        uvtools_layout.addWidget(self.uvtools_path_button, 0, 2)
        uvtools_layout.addWidget(QLabel("Working Temp Folder:"), 1, 0)
        self.uvtools_temp_folder_edit = QLineEdit()
        uvtools_layout.addWidget(self.uvtools_temp_folder_edit, 1, 1)
        self.uvtools_temp_folder_button = QPushButton("Browse...")
        uvtools_layout.addWidget(self.uvtools_temp_folder_button, 1, 2)
        uvtools_layout.addWidget(QLabel("Input Slice File:"), 2, 0)
        self.uvtools_input_file_edit = QLineEdit()
        uvtools_layout.addWidget(self.uvtools_input_file_edit, 2, 1)
        self.uvtools_input_file_button = QPushButton("Browse...")
        uvtools_layout.addWidget(self.uvtools_input_file_button, 2, 2)
        self.uvtools_cleanup_checkbox = QCheckBox("Delete Temporary Files on Completion")
        self.uvtools_cleanup_checkbox.setChecked(True)
        uvtools_layout.addWidget(self.uvtools_cleanup_checkbox, 3, 1, 1, 2)
        self.io_stacked_widget.addWidget(uvtools_mode_widget)

        # Common Output Folder
        output_folder_layout = QGridLayout()
        output_folder_layout.addWidget(QLabel("Main Output Folder:"), 0, 0)
        self.output_folder_edit = QLineEdit()
        output_folder_layout.addWidget(self.output_folder_edit, 0, 1)
        self.output_folder_button = QPushButton("Browse...")
        output_folder_layout.addWidget(self.output_folder_button, 0, 2)
        io_layout.addLayout(output_folder_layout)

        # --- Dask & General Settings Section ---
        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout(settings_group)
        main_layout.addWidget(settings_group)

        dask_layout = QHBoxLayout()
        dask_layout.addWidget(QLabel("Dask Worker Count:"))
        self.dask_workers_edit = QLineEdit(str(DEFAULT_DASK_WORKERS))
        self.dask_workers_edit.setValidator(QIntValidator(1, 128, self))
        self.dask_workers_edit.setFixedWidth(60)
        dask_layout.addWidget(self.dask_workers_edit)
        dask_layout.addStretch(1)
        settings_layout.addLayout(dask_layout)

        self.numba_checkbox = QCheckBox("Enable Numba Acceleration (if available)")
        self.numba_checkbox.setChecked(True)
        settings_layout.addWidget(self.numba_checkbox)

        config_buttons_layout = QHBoxLayout()
        self.save_config_button = QPushButton("Save Config...")
        config_buttons_layout.addWidget(self.save_config_button)
        self.load_config_button = QPushButton("Load Config...")
        config_buttons_layout.addWidget(self.load_config_button)
        config_buttons_layout.addStretch(1)
        settings_layout.addLayout(config_buttons_layout)

        # --- Processing Section ---
        processing_group = QGroupBox("Processing Controls")
        processing_layout = QGridLayout(processing_group)
        main_layout.addWidget(processing_group)

        self.extract_xy_button = QPushButton("Extract XY Planes")
        self.extract_xz_button = QPushButton("Extract XZ Planes")
        self.extract_yz_button = QPushButton("Extract YZ Planes")
        self.stop_button = QPushButton("Stop Processing")
        self.stop_button.setEnabled(False)

        processing_layout.addWidget(self.extract_xy_button, 0, 0)
        processing_layout.addWidget(self.extract_xz_button, 0, 1)
        processing_layout.addWidget(self.extract_yz_button, 0, 2)
        processing_layout.addWidget(self.stop_button, 1, 0, 1, 3)

        # --- Status Section ---
        main_layout.addStretch(1)
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        main_layout.addWidget(self.progress_bar)
        self.status_label = QLabel("Status: Ready")
        self.status_label.setWordWrap(True)
        main_layout.addWidget(self.status_label)

    def _connect_signals(self):
        # I/O controls
        self.input_folder_button.clicked.connect(lambda: self.browse_folder(self.input_folder_edit))
        self.output_folder_button.clicked.connect(lambda: self.browse_folder(self.output_folder_edit))
        self.uvtools_path_button.clicked.connect(lambda: self.browse_file(self.uvtools_path_edit, "Select UVToolsCmd.exe", "Executable Files (*.exe)"))
        self.uvtools_temp_folder_button.clicked.connect(lambda: self.browse_folder(self.uvtools_temp_folder_edit))
        self.uvtools_input_file_button.clicked.connect(lambda: self.browse_file(self.uvtools_input_file_edit, "Select Input Slice File"))
        self.input_mode_group.idClicked.connect(self.io_stacked_widget.setCurrentIndex)

        # Config controls
        self.save_config_button.clicked.connect(self._save_config_to_file)
        self.load_config_button.clicked.connect(self._load_config_from_file)

        # Processing controls
        self.extract_xy_button.clicked.connect(lambda: self.start_processing(axis=0))
        self.extract_xz_button.clicked.connect(lambda: self.start_processing(axis=1))
        self.extract_yz_button.clicked.connect(lambda: self.start_processing(axis=2))
        self.stop_button.clicked.connect(self.stop_processing)

    def _autodetect_uvtools(self):
        """Checks for UVTools in the default location and populates the path if found."""
        default_path = "C:\\Program Files\\UVTools\\UVToolsCmd.exe"
        if os.path.exists(default_path) and not self.uvtools_path_edit.text():
            self.uvtools_path_edit.setText(default_path)

    def browse_folder(self, line_edit: QLineEdit):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", line_edit.text())
        if folder:
            line_edit.setText(folder)

    def browse_file(self, line_edit: QLineEdit, caption: str, file_filter: str = "All Files (*)"):
        file, _ = QFileDialog.getOpenFileName(self, caption, line_edit.text(), file_filter)
        if file:
            line_edit.setText(file)

    def load_settings(self):
        """Loads settings from the global config object into the UI."""
        self.resize(self.settings.value("window_size", self.sizeHint()))
        self.move(self.settings.value("window_position", self.pos()))

        self.folder_mode_radio.setChecked(app_config.input_mode == "folder")
        self.uvtools_mode_radio.setChecked(app_config.input_mode == "uvtools")
        self.io_stacked_widget.setCurrentIndex(1 if app_config.input_mode == "uvtools" else 0)

        self.input_folder_edit.setText(app_config.input_folder)
        self.output_folder_edit.setText(app_config.output_folder)
        self.uvtools_path_edit.setText(app_config.uvtools_path)
        self.uvtools_temp_folder_edit.setText(app_config.uvtools_temp_folder)
        self.uvtools_input_file_edit.setText(app_config.uvtools_input_file)
        self.uvtools_cleanup_checkbox.setChecked(app_config.uvtools_delete_temp_on_completion)

        self.dask_workers_edit.setText(str(app_config.dask_workers))
        self.numba_checkbox.setChecked(app_config.use_numba)

    def save_settings(self):
        """Saves current UI settings to the global config object."""
        self.settings.setValue("window_size", self.size())
        self.settings.setValue("window_position", self.pos())

        app_config.input_mode = "uvtools" if self.uvtools_mode_radio.isChecked() else "folder"
        app_config.input_folder = self.input_folder_edit.text()
        app_config.output_folder = self.output_folder_edit.text()
        app_config.uvtools_path = self.uvtools_path_edit.text()
        app_config.uvtools_temp_folder = self.uvtools_temp_folder_edit.text()
        app_config.uvtools_input_file = self.uvtools_input_file_edit.text()
        app_config.uvtools_delete_temp_on_completion = self.uvtools_cleanup_checkbox.isChecked()

        try:
            app_config.dask_workers = int(self.dask_workers_edit.text())
        except ValueError:
            app_config.dask_workers = DEFAULT_DASK_WORKERS
        app_config.use_numba = self.numba_checkbox.isChecked()

        app_config.save("dask_engine_config.json")

    def _save_config_to_file(self):
        self.save_settings()
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Configuration", "dask_engine_custom.json", "JSON Files (*.json)")
        if filepath:
            try:
                app_config.save(filepath)
                self.show_info_message("Success", "Configuration saved.")
            except Exception as e:
                self.show_error_message("Save Error", f"Failed to save configuration:\n{e}")

    def _load_config_from_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Load Configuration", "", "JSON Files (*.json)")
        if filepath:
            try:
                # Load into a new temp config object first
                loaded_config = Config.load(filepath)
                # Update the global config object
                app_config.__dict__.update(loaded_config.__dict__)
                self.load_settings()
                self.show_info_message("Success", "Configuration loaded.")
            except Exception as e:
                self.show_error_message("Load Error", f"Failed to load configuration:\n{e}")

    def closeEvent(self, event):
        """Save settings when the application is closed."""
        self.save_settings()
        if self.processor_thread and self.processor_thread.isRunning():
            self.stop_processing()
            self.processor_thread.wait(5000) # Wait up to 5s for the thread to finish
        event.accept()

    def start_processing(self, axis: int):
        """Validates inputs and starts the processing thread."""
        if self.processor_thread and self.processor_thread.isRunning():
            self.show_error_message("Busy", "A process is already running. Please wait or stop the current process.")
            return

        try:
            self.save_settings()
            # Input validation
            if not app_config.output_folder or not os.path.isdir(app_config.output_folder):
                raise ValueError("Main Output Folder must be a valid, existing directory.")
            if app_config.input_mode == "folder":
                if not app_config.input_folder or not os.path.isdir(app_config.input_folder):
                    raise ValueError("Input Folder (PNGs) must be a valid, existing directory.")
            elif app_config.input_mode == "uvtools":
                if not os.path.exists(app_config.uvtools_path):
                    raise ValueError("UVToolsCmd.exe path is not valid.")
                if not app_config.uvtools_temp_folder or not os.path.isdir(app_config.uvtools_temp_folder):
                    raise ValueError("Working Temp Folder must be a valid, existing directory.")
                if not os.path.exists(app_config.uvtools_input_file):
                    raise ValueError("Input Slice File is not valid.")

            self.set_ui_enabled(False)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("%p%")

            self.processor_thread = ProcessingThread(app_config=app_config, axis_to_extract=axis)
            self.processor_thread.status_update.connect(self.update_status)
            self.processor_thread.progress_update.connect(self.update_progress)
            self.processor_thread.error_signal.connect(self.show_error)
            self.processor_thread.finished_signal.connect(self.processing_finished)
            self.processor_thread.start()

        except Exception as e:
            self.show_error_message("Input Error", str(e))
            self.set_ui_enabled(True)

    def stop_processing(self):
        if self.processor_thread and self.processor_thread.isRunning():
            self.processor_thread.stop()
            self.stop_button.setText("Stopping...")
            self.stop_button.setEnabled(False)

    @Slot(str)
    def update_status(self, message: str):
        self.status_label.setText(f"Status: {message}")

    @Slot(int, int)
    def update_progress(self, current_val: int, max_val: int):
        if max_val > 0:
            self.progress_bar.setMaximum(max_val)
            self.progress_bar.setValue(current_val)
            percentage = (current_val / max_val) * 100
            self.progress_bar.setFormat(f"%p% ({current_val}/{max_val})")
        else:
            self.progress_bar.setFormat("Processing...")


    @Slot(str)
    def show_error(self, message: str):
        self.show_error_message("Processing Error", message, is_detailed=True)
        self.processing_finished()

    @Slot()
    def processing_finished(self):
        self.status_label.setText("Status: Finished, Stopped, or Error.")
        self.set_ui_enabled(True)
        self.processor_thread = None
        self.progress_bar.setFormat("Done")

    def show_error_message(self, title: str, text: str, is_detailed: bool = False):
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setWindowTitle(title)
        if is_detailed:
            msg_box.setText("An error occurred. See details for more information.")
            msg_box.setDetailedText(text)
        else:
            msg_box.setText(text)
        msg_box.exec()

    def show_info_message(self, title: str, text: str):
        QMessageBox.information(self, title, text)

    def set_ui_enabled(self, enabled: bool):
        """Toggles the enabled state of UI widgets during processing."""
        self.extract_xy_button.setEnabled(enabled)
        self.extract_xz_button.setEnabled(enabled)
        self.extract_yz_button.setEnabled(enabled)
        self.stop_button.setEnabled(not enabled)

        # Also disable settings groups
        for child in self.findChildren(QGroupBox):
            child.setEnabled(enabled)

        if not enabled:
            self.stop_button.setText("Stop Processing")
