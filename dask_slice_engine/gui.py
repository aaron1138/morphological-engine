import os
import subprocess
import datetime
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QProgressBar, QFileDialog, QMessageBox, QCheckBox,
    QTabWidget, QGroupBox, QRadioButton, QButtonGroup, QStackedWidget,
    QGridLayout, QFrame
)
from PySide6.QtCore import QThread, Signal, Slot, Qt, QSettings
from PySide6.QtGui import QIntValidator

from dask_slice_engine.config import app_config, Config
from dask_slice_engine.dask_engine import DaskEngine

class ProcessingThread(QThread):
    """
    Manages the Dask processing in a separate thread.
    """
    status_update = Signal(str)
    progress_update = Signal(int)
    error_signal = Signal(str)
    finished_signal = Signal()

    def __init__(self, app_config: Config):
        super().__init__()
        self.app_config = app_config
        self._is_running = True

    def run(self):
        """
        The main processing loop.
        """
        self.status_update.emit("Processing started...")
        try:
            input_path = self.app_config.input_folder
            output_path = self.app_config.output_folder

            if self.app_config.input_mode == "uvtools":
                self.status_update.emit("Starting UVTools slice extraction...")

                session_temp_folder = os.path.join(self.app_config.uvtools_temp_folder, f"dask_engine_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
                input_path = os.path.join(session_temp_folder, "Input")
                os.makedirs(input_path, exist_ok=True)

                command = [
                    self.app_config.uvtools_path, "extract", self.app_config.uvtools_input_file,
                    input_path, "--content", "Layers"
                ]
                self.status_update.emit(f"Running command: {' '.join(command)}")

                try:
                    creation_flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                    process = subprocess.run(command, capture_output=True, text=True, check=True, creationflags=creation_flags)
                    self.status_update.emit("UVTools extraction completed.")
                except subprocess.CalledProcessError as e:
                    raise RuntimeError(f"UVTools exited with an error (code {e.returncode}):\n\n{e.stderr}")
                except Exception as e:
                    raise RuntimeError(f"UVTools extraction failed: {e}")

            if not os.path.isdir(input_path):
                raise ValueError(f"Input folder not found: {input_path}")

            engine = DaskEngine(input_path)

            num_images = engine.image_stack.shape[0]
            slice_indices = range(num_images) # Process all slices

            total_operations = len(slice_indices) * 3  # 3 planes per slice
            operations_done = 0

            for i, slice_index in enumerate(slice_indices):
                if not self._is_running:
                    break

                # Process XY plane
                self.status_update.emit(f"Processing XY plane {slice_index}...")
                xy_plane = engine.get_xy_plane(slice_index)
                engine.save_plane(xy_plane, os.path.join(output_path, f"plane_xy_{slice_index}.png"))
                operations_done += 1
                self.progress_update.emit(int((operations_done / total_operations) * 100))

                # Process XZ plane
                self.status_update.emit(f"Processing XZ plane {slice_index}...")
                xz_plane = engine.get_xz_plane(slice_index)
                engine.save_plane(xz_plane, os.path.join(output_path, f"plane_xz_{slice_index}.png"))
                operations_done += 1
                self.progress_update.emit(int((operations_done / total_operations) * 100))

                # Process YZ plane
                self.status_update.emit(f"Processing YZ plane {slice_index}...")
                yz_plane = engine.get_yz_plane(slice_index)
                engine.save_plane(yz_plane, os.path.join(output_path, f"plane_yz_{slice_index}.png"))
                operations_done += 1
                self.progress_update.emit(int((operations_done / total_operations) * 100))

        except Exception as e:
            import traceback
            error_info = f"Error in processing thread: {e}\n\n{traceback.format_exc()}"
            self.error_signal.emit(error_info)
        finally:
            self.finished_signal.emit()

    def stop(self):
        self._is_running = False

class DaskEngineApp(QWidget):
    """The main application window."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dask-Powered 3D Image-Stack-Slice-Engine")
        self.settings = QSettings("MyCompany", "DaskSliceEngine")
        self.processor_thread = None
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
        self.io_stacked_widget.addWidget(uvtools_mode_widget)

        main_layout.addWidget(io_group)

        # --- Engine Settings Section ---
        engine_group = QGroupBox("Engine Settings")
        engine_layout = QVBoxLayout(engine_group)

        thread_layout = QHBoxLayout()
        thread_layout.addWidget(QLabel("Thread Count:"))
        self.thread_count_edit = QLineEdit(str(app_config.thread_count))
        self.thread_count_edit.setValidator(QIntValidator(1, 128, self))
        self.thread_count_edit.setFixedWidth(60)
        thread_layout.addWidget(self.thread_count_edit)
        thread_layout.addStretch(1)
        engine_layout.addLayout(thread_layout)

        main_layout.addWidget(engine_group)
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
        self.input_folder_button.clicked.connect(lambda: self.browse_folder(self.input_folder_edit))
        self.output_folder_button.clicked.connect(lambda: self.browse_folder(self.output_folder_edit))
        self.uvtools_path_button.clicked.connect(lambda: self.browse_file(self.uvtools_path_edit, "Select UVToolsCmd.exe", "Executable Files (*.exe)"))
        self.uvtools_temp_folder_button.clicked.connect(lambda: self.browse_folder(self.uvtools_temp_folder_edit))
        self.uvtools_input_file_button.clicked.connect(lambda: self.browse_file(self.uvtools_input_file_edit, "Select Input Slice File"))
        self.input_mode_group.idClicked.connect(self.on_input_mode_changed)
        self.start_stop_button.clicked.connect(self.toggle_processing)

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
        """Loads settings from the global config object into the UI."""
        self.resize(self.settings.value("window_size", self.size()))
        self.move(self.settings.value("window_position", self.pos()))

        self.folder_mode_radio.setChecked(app_config.input_mode == "folder")
        self.uvtools_mode_radio.setChecked(app_config.input_mode == "uvtools")
        self.io_stacked_widget.setCurrentIndex(1 if app_config.input_mode == "uvtools" else 0)
        self.input_folder_edit.setText(app_config.input_folder)
        self.output_folder_edit.setText(app_config.output_folder)
        self.uvtools_path_edit.setText(app_config.uvtools_path)
        self.uvtools_temp_folder_edit.setText(app_config.uvtools_temp_folder)
        self.uvtools_input_file_edit.setText(app_config.uvtools_input_file)
        self.thread_count_edit.setText(str(app_config.thread_count))

    def save_settings(self):
        """Saves current UI settings to the global config object and QSettings."""
        self.settings.setValue("window_size", self.size())
        self.settings.setValue("window_position", self.pos())

        app_config.input_mode = "uvtools" if self.uvtools_mode_radio.isChecked() else "folder"
        app_config.input_folder = self.input_folder_edit.text()
        app_config.output_folder = self.output_folder_edit.text()
        app_config.uvtools_path = self.uvtools_path_edit.text()
        app_config.uvtools_temp_folder = self.uvtools_temp_folder_edit.text()
        app_config.uvtools_input_file = self.uvtools_input_file_edit.text()
        try:
            app_config.thread_count = int(self.thread_count_edit.text())
        except ValueError:
            app_config.thread_count = os.cpu_count() or 4

        app_config.save("app_config.json")

    def closeEvent(self, event):
        self.save_settings()
        if self.processor_thread and self.processor_thread.isRunning():
            self.processor_thread.stop()
            self.processor_thread.wait(5000)
        event.accept()

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
                if not app_config.output_folder:
                    raise ValueError("Output folder must be set.")
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

    def show_info_message(self, title, text):
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setWindowTitle(title)
        msg_box.setText(text)
        msg_box.setTextInteractionFlags(Qt.TextSelectableByMouse)
        msg_box.exec()

    def set_ui_enabled(self, enabled):
        """Toggles the enabled state of all UI widgets."""
        self.io_group.setEnabled(enabled)
        self.engine_group.setEnabled(enabled)
        self.start_stop_button.setEnabled(True)
        if not enabled:
            self.start_stop_button.setText("Stop Processing")
        else:
            self.start_stop_button.setText("Start Processing")
