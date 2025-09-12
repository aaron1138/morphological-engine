import os
import subprocess
import datetime
import shutil
import re

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QProgressBar, QFileDialog, QMessageBox, QCheckBox,
    QGroupBox, QRadioButton, QButtonGroup, QStackedWidget,
    QGridLayout, QFrame
)
from PySide6.QtCore import QThread, Signal, Slot, Qt, QSettings, QSize
from PySide6.QtGui import QIntValidator

from config import app_config, Config, DEFAULT_NUM_WORKERS
from dask_engine import DaskEngine

class ProcessingThread(QThread):
    """
    Manages the Dask processing in a separate thread.
    """
    status_update = Signal(str)
    progress_update = Signal(int)
    error_signal = Signal(str)
    finished_signal = Signal()

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self._is_running = True
        self.run_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.session_temp_folder = ""

    def _run_uvtools_extraction(self) -> str:
        """
        Executes UVToolsCmd.exe to extract layers into a timestamped temp folder.
        """
        self.status_update.emit("Starting UVTools slice extraction...")

        self.session_temp_folder = os.path.join(self.config.uvtools_temp_folder, f"dask_engine_{self.run_timestamp}")
        input_folder = os.path.join(self.session_temp_folder, "Input")

        os.makedirs(input_folder, exist_ok=True)

        command = [
            self.config.uvtools_path, "extract", self.config.uvtools_input_file,
            input_folder, "--content", "Layers"
        ]
        self.status_update.emit(f"Running command: {' '.join(command)}")

        try:
            creation_flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            process = subprocess.run(command, capture_output=True, text=True, check=True, creationflags=creation_flags)

            if process.returncode not in [0, 1]:
                raise RuntimeError(f"UVTools exited with an error (code {process.returncode}):\n\n{process.stderr}")

            self.status_update.emit("UVTools extraction completed.")
            return input_folder
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"UVTools execution failed with exit code {e.returncode}:\n{e.stderr}")
        except Exception as e:
            raise RuntimeError(f"UVTools extraction failed: {e}")

    def run(self):
        """Main processing loop."""
        self.status_update.emit("Processing started...")
        try:
            input_path = ""
            if self.config.input_mode == "uvtools":
                if not self._is_running: return
                input_path = self._run_uvtools_extraction()
            else:
                input_path = self.config.input_folder

            if not self._is_running: return
            self.status_update.emit("Initializing Dask engine...")
            engine = DaskEngine(self.config)

            self.status_update.emit(f"Loading images from: {input_path}")
            stack = engine.load_images_to_dask_stack(input_path)

            self.status_update.emit("Dask array created successfully.")
            self.status_update.emit(f"Stack shape: {stack.shape}, Chunks: {stack.chunksize}")
            self.progress_update.emit(25)

            self.status_update.emit("Performing a test computation (mean of stack)...")
            mean_value = stack.mean().compute()
            self.status_update.emit(f"Test computation complete. Mean pixel value: {mean_value:.2f}")
            self.progress_update.emit(100)

        except Exception as e:
            import traceback
            error_info = f"Error in processing thread: {e}\n\n{traceback.format_exc()}"
            self.error_signal.emit(error_info)
        finally:
            if self.config.input_mode == "uvtools" and self.config.uvtools_delete_temp_on_completion:
                if self.session_temp_folder and os.path.isdir(self.session_temp_folder):
                    self.status_update.emit(f"Deleting temporary folder: {self.session_temp_folder}")
                    try:
                        shutil.rmtree(self.session_temp_folder)
                        self.status_update.emit("Temporary files deleted.")
                    except Exception as e:
                        self.error_signal.emit(f"Could not delete temp folder: {e}")

            if self._is_running:
                self.status_update.emit("Processing complete!")
            else:
                self.status_update.emit("Processing stopped by user.")
            self.finished_signal.emit()

    def stop(self):
        self._is_running = False
        self.status_update.emit("Stopping process...")

class DaskProcessorApp(QWidget):
    """The main application window."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dask-Based 3D Image Processor")
        self.settings = QSettings("MyCompany", "DaskImageProcessor")
        self.processor_thread = None
        self.init_ui()
        self._autodetect_uvtools()
        self.load_settings()
        self._connect_signals()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        io_group = QGroupBox("I/O Configuration")
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

        folder_mode_widget = QWidget()
        folder_mode_layout = QGridLayout(folder_mode_widget)
        folder_mode_layout.addWidget(QLabel("Input Folder:"), 0, 0)
        self.input_folder_edit = QLineEdit()
        folder_mode_layout.addWidget(self.input_folder_edit, 0, 1)
        self.input_folder_button = QPushButton("Browse...")
        folder_mode_layout.addWidget(self.input_folder_button, 0, 2)
        folder_mode_layout.addWidget(QLabel("Output Folder (for future use):"), 1, 0)
        self.output_folder_edit = QLineEdit()
        folder_mode_layout.addWidget(self.output_folder_edit, 1, 1)
        self.output_folder_button = QPushButton("Browse...")
        folder_mode_layout.addWidget(self.output_folder_button, 1, 2)
        self.io_stacked_widget.addWidget(folder_mode_widget)

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
        self.uvtools_cleanup_checkbox.setChecked(True)
        uvtools_mode_layout.addWidget(self.uvtools_cleanup_checkbox, 3, 1, 1, 2)
        self.io_stacked_widget.addWidget(uvtools_mode_widget)
        main_layout.addWidget(io_group)

        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout(settings_group)
        dask_layout = QGridLayout()
        dask_layout.addWidget(QLabel("Worker Threads:"), 0, 0)
        self.thread_count_edit = QLineEdit(str(DEFAULT_NUM_WORKERS))
        self.thread_count_edit.setValidator(QIntValidator(1, 128, self))
        dask_layout.addWidget(self.thread_count_edit, 0, 1)
        dask_layout.addWidget(QLabel("Chunk Size (Z, Y, X):"), 1, 0)
        self.chunk_z_edit = QLineEdit("64"); self.chunk_y_edit = QLineEdit("64"); self.chunk_x_edit = QLineEdit("64")
        chunk_layout = QHBoxLayout()
        chunk_layout.addWidget(self.chunk_z_edit); chunk_layout.addWidget(QLabel("Z"))
        chunk_layout.addWidget(self.chunk_y_edit); chunk_layout.addWidget(QLabel("Y"))
        chunk_layout.addWidget(self.chunk_x_edit); chunk_layout.addWidget(QLabel("X"))
        dask_layout.addLayout(chunk_layout, 1, 1)
        settings_layout.addLayout(dask_layout)
        config_buttons_layout = QHBoxLayout()
        self.save_config_button = QPushButton("Save Config As...")
        config_buttons_layout.addWidget(self.save_config_button)
        self.load_config_button = QPushButton("Load Config...")
        config_buttons_layout.addWidget(self.load_config_button)
        config_buttons_layout.addStretch(1)
        settings_layout.addLayout(config_buttons_layout)
        main_layout.addWidget(settings_group)

        main_layout.addStretch(1)

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
        self.input_folder_button.clicked.connect(lambda: self.browse_folder(self.input_folder_edit, "Select Input Folder"))
        self.output_folder_button.clicked.connect(lambda: self.browse_folder(self.output_folder_edit, "Select Output Folder"))
        self.uvtools_path_button.clicked.connect(lambda: self.browse_file(self.uvtools_path_edit, "Select UVToolsCmd.exe", "Executable Files (*.exe)"))
        self.uvtools_temp_folder_button.clicked.connect(lambda: self.browse_folder(self.uvtools_temp_folder_edit, "Select Temporary Folder"))
        self.uvtools_input_file_button.clicked.connect(lambda: self.browse_file(self.uvtools_input_file_edit, "Select Input Slice File"))
        self.input_mode_group.idClicked.connect(self.io_stacked_widget.setCurrentIndex)
        self.start_stop_button.clicked.connect(self.toggle_processing)
        self.save_config_button.clicked.connect(self._save_config_to_file)
        self.load_config_button.clicked.connect(self._load_config_from_file)

    def _autodetect_uvtools(self):
        paths_to_check = ["C:\\Program Files\\UVTools\\UVToolsCmd.exe", os.path.expanduser("~\\AppData\\Local\\UVTools\\UVToolsCmd.exe")]
        for path in paths_to_check:
            if os.path.exists(path) and not self.uvtools_path_edit.text():
                self.uvtools_path_edit.setText(path); break

    def browse_folder(self, line_edit, caption):
        folder = QFileDialog.getExistingDirectory(self, caption, line_edit.text())
        if folder: line_edit.setText(folder)

    def browse_file(self, line_edit, caption, file_filter="All Files (*)"):
        file, _ = QFileDialog.getOpenFileName(self, caption, line_edit.text(), file_filter)
        if file: line_edit.setText(file)

    def _update_ui_from_config(self):
        self.folder_mode_radio.setChecked(app_config.input_mode == "folder")
        self.uvtools_mode_radio.setChecked(app_config.input_mode == "uvtools")
        self.io_stacked_widget.setCurrentIndex(1 if app_config.input_mode == "uvtools" else 0)
        self.input_folder_edit.setText(app_config.input_folder)
        self.output_folder_edit.setText(app_config.output_folder)
        self.uvtools_path_edit.setText(app_config.uvtools_path)
        self.uvtools_temp_folder_edit.setText(app_config.uvtools_temp_folder)
        self.uvtools_input_file_edit.setText(app_config.uvtools_input_file)
        self.uvtools_cleanup_checkbox.setChecked(app_config.uvtools_delete_temp_on_completion)
        self.thread_count_edit.setText(str(app_config.thread_count))
        self.chunk_x_edit.setText(str(app_config.chunk_size_x))
        self.chunk_y_edit.setText(str(app_config.chunk_size_y))
        self.chunk_z_edit.setText(str(app_config.chunk_size_z))

    def _save_ui_to_config(self):
        app_config.input_mode = "uvtools" if self.uvtools_mode_radio.isChecked() else "folder"
        app_config.input_folder = self.input_folder_edit.text()
        app_config.output_folder = self.output_folder_edit.text()
        app_config.uvtools_path = self.uvtools_path_edit.text()
        app_config.uvtools_temp_folder = self.uvtools_temp_folder_edit.text()
        app_config.uvtools_input_file = self.uvtools_input_file_edit.text()
        app_config.uvtools_delete_temp_on_completion = self.uvtools_cleanup_checkbox.isChecked()
        try: app_config.thread_count = int(self.thread_count_edit.text())
        except ValueError: app_config.thread_count = DEFAULT_NUM_WORKERS
        try: app_config.chunk_size_x = int(self.chunk_x_edit.text())
        except ValueError: app_config.chunk_size_x = 64
        try: app_config.chunk_size_y = int(self.chunk_y_edit.text())
        except ValueError: app_config.chunk_size_y = 64
        try: app_config.chunk_size_z = int(self.chunk_z_edit.text())
        except ValueError: app_config.chunk_size_z = 64

    def load_settings(self, from_file=None):
        global app_config
        if from_file:
            app_config = Config.load(from_file)
        else:
            self.resize(self.settings.value("window_size", QSize(600, 550)))
            self.move(self.settings.value("window_position", self.pos()))
            app_config.input_mode = self.settings.value("input_mode", "folder")
            app_config.input_folder = self.settings.value("input_folder", "")
            app_config.output_folder = self.settings.value("output_folder", "")
            app_config.uvtools_path = self.settings.value("uvtools_path", "")
            app_config.uvtools_temp_folder = self.settings.value("uvtools_temp_folder", "")
            app_config.uvtools_input_file = self.settings.value("uvtools_input_file", "")
            app_config.uvtools_delete_temp_on_completion = self.settings.value("uvtools_delete_temp", "true") == "true"
            app_config.thread_count = int(self.settings.value("thread_count", DEFAULT_NUM_WORKERS))
            app_config.chunk_size_x = int(self.settings.value("chunk_x", 64))
            app_config.chunk_size_y = int(self.settings.value("chunk_y", 64))
            app_config.chunk_size_z = int(self.settings.value("chunk_z", 64))
        self._update_ui_from_config()

    def save_settings(self, to_file=None):
        self._save_ui_to_config()
        if to_file:
            app_config.save(to_file)
        else:
            self.settings.setValue("window_size", self.size())
            self.settings.setValue("window_position", self.pos())
            self.settings.setValue("input_mode", app_config.input_mode)
            self.settings.setValue("input_folder", app_config.input_folder)
            self.settings.setValue("output_folder", app_config.output_folder)
            self.settings.setValue("uvtools_path", app_config.uvtools_path)
            self.settings.setValue("uvtools_temp_folder", app_config.uvtools_temp_folder)
            self.settings.setValue("uvtools_input_file", app_config.uvtools_input_file)
            self.settings.setValue("uvtools_delete_temp", "true" if app_config.uvtools_delete_temp_on_completion else "false")
            self.settings.setValue("thread_count", app_config.thread_count)
            self.settings.setValue("chunk_x", app_config.chunk_size_x)
            self.settings.setValue("chunk_y", app_config.chunk_size_y)
            self.settings.setValue("chunk_z", app_config.chunk_size_z)

    def _save_config_to_file(self):
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Configuration", "dask_engine_config.json", "JSON Files (*.json)")
        if filepath:
            try:
                self.save_settings(to_file=filepath)
                self.show_info_message("Success", f"Configuration saved to:\n{filepath}")
            except Exception as e:
                self.show_error_message("Save Error", f"Failed to save configuration:\n{e}")

    def _load_config_from_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Load Configuration", "", "JSON Files (*.json)")
        if filepath:
            try:
                self.load_settings(from_file=filepath)
                self.show_info_message("Success", f"Configuration loaded from:\n{filepath}")
            except Exception as e:
                self.show_error_message("Load Error", f"Failed to load configuration:\n{e}")

    def closeEvent(self, event):
        self.save_settings()
        if self.processor_thread and self.processor_thread.isRunning():
            self.processor_thread.stop(); self.processor_thread.wait(5000)
        event.accept()

    def toggle_processing(self):
        if self.processor_thread and self.processor_thread.isRunning():
            self.processor_thread.stop()
            self.start_stop_button.setText("Stopping...")
            self.start_stop_button.setEnabled(False)
        else: self.start_processing()

    def start_processing(self):
        try:
            self._save_ui_to_config()
            if app_config.input_mode == "folder" and (not app_config.input_folder or not os.path.isdir(app_config.input_folder)):
                raise ValueError("Input folder must be a valid, existing directory.")
            elif app_config.input_mode == "uvtools":
                if not app_config.uvtools_path or not os.path.exists(app_config.uvtools_path): raise ValueError("UVToolsCmd.exe path is not valid.")
                if not app_config.uvtools_temp_folder or not os.path.isdir(app_config.uvtools_temp_folder): raise ValueError("Working Temp Folder must be a valid, existing directory.")
                if not app_config.uvtools_input_file or not os.path.exists(app_config.uvtools_input_file): raise ValueError("Input Slice File is not valid.")

            self.set_ui_enabled(False)
            self.processor_thread = ProcessingThread(app_config)
            self.processor_thread.status_update.connect(self.update_status)
            self.processor_thread.progress_update.connect(self.progress_bar.setValue)
            self.processor_thread.error_signal.connect(self.show_error)
            self.processor_thread.finished_signal.connect(self.processing_finished)
            self.processor_thread.start()
        except ValueError as e:
            self.show_error_message("Input Error", str(e)); self.set_ui_enabled(True)
        except Exception as e:
            self.show_error_message("Error", f"An unexpected error occurred: {e}"); self.set_ui_enabled(True)

    @Slot(str)
    def update_status(self, message): self.status_label.setText(f"Status: {message}")

    @Slot(str)
    def show_error(self, message):
        self.show_error_message("Processing Error", message, is_detailed=True)
        self.processing_finished()

    @Slot()
    def processing_finished(self):
        self.set_ui_enabled(True)
        self.processor_thread = None

    def set_ui_enabled(self, enabled):
        self.io_group.setEnabled(enabled)
        self.settings_group.setEnabled(enabled)
        self.start_stop_button.setEnabled(True)
        self.start_stop_button.setText("Start Processing" if enabled else "Stop")

    def show_info_message(self, title, text):
        msg_box = QMessageBox(self); msg_box.setIcon(QMessageBox.Information)
        msg_box.setWindowTitle(title); msg_box.setText(text); msg_box.exec()

    def show_error_message(self, title, text, is_detailed=False):
        msg_box = QMessageBox(self); msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle(title)
        msg_box.setDetailedText(text) if is_detailed else msg_box.setText(text)
        msg_box.exec()
