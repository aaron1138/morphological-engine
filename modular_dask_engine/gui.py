import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QGroupBox, QRadioButton, QHBoxLayout,
    QStackedWidget, QGridLayout, QLabel, QLineEdit, QPushButton, QButtonGroup,
    QFrame, QTabWidget, QProgressBar
)
import subprocess
import os
import re
import shutil
import datetime
from PySide6.QtCore import QSettings, Signal, QThread, Slot, Qt
from PySide6.QtWidgets import QFileDialog, QMessageBox
from config import app_config
from dask_engine import DaskEngine

class ProcessingThread(QThread):
    status_update = Signal(str)
    progress_update = Signal(int)
    error_signal = Signal(str)
    finished_signal = Signal()

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dask_engine = DaskEngine(config)
        self._is_running = True
        self.run_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.session_temp_folder = ""

    def _run_uvtools_extraction(self) -> str:
        """
        Executes UVToolsCmd.exe to extract layers into a timestamped temp folder.
        """
        self.status_update.emit("Starting UVTools slice extraction...")

        self.session_temp_folder = os.path.join(self.config.uvtools_temp_folder, f"{self.config.output_file_prefix}{self.run_timestamp}")
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
            self.status_update.emit("UVTools extraction completed.")
            return input_folder
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"UVTools exited with an error (code {e.returncode}):\n\n{e.stderr}")
        except Exception as e:
            raise RuntimeError(f"UVTools extraction failed: {e}")

    def _generate_uvtop_file(self, processed_images_folder: str) -> str:
        """Generates the .uvtop XML file for repacking."""
        self.status_update.emit("Generating UVTools operation file...")

        numeric_pattern = re.compile(r'(\d+)\.\w+$')
        def get_numeric_part(filename):
            match = numeric_pattern.search(filename)
            return int(match.group(1)) if match else float('inf')

        processed_files = sorted(
            [os.path.join(processed_images_folder, f) for f in os.listdir(processed_images_folder) if f.lower().endswith('.png')],
            key=get_numeric_part
        )

        if not processed_files:
            raise RuntimeError("No processed image files found to generate .uvtop file.")

        xml_content = '<?xml version="1.0" encoding="utf-8" standalone="no"?>\n'
        xml_content += '<OperationLayerImport xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema">\n'
        xml_content += '  <LayerRangeSelection>None</LayerRangeSelection>\n'
        xml_content += '  <ImportType>Replace</ImportType>\n'
        xml_content += '  <Files>\n'
        for f_path in processed_files:
            xml_content += '    <GenericFileRepresentation>\n'
            xml_content += f'      <FilePath>{f_path}</FilePath>\n'
            xml_content += '    </GenericFileRepresentation>\n'
        xml_content += '  </Files>\n'
        xml_content += '</OperationLayerImport>\n'

        uvtop_filename = f"repack_operations_{self.run_timestamp}.uvtop"
        uvtop_filepath = os.path.join(self.session_temp_folder, uvtop_filename)

        with open(uvtop_filepath, 'w', encoding='utf-8') as f:
            f.write(xml_content)

        self.status_update.emit("Operation file generated.")
        return uvtop_filepath

    def _run_uvtools_repack(self, uvtop_filepath: str, output_folder: str):
        """Executes UVToolsCmd.exe to repack the processed layers."""
        self.status_update.emit("Repacking slice file with processed layers...")

        original_filename = os.path.basename(self.config.uvtools_input_file)
        output_filename = f"{self.config.output_file_prefix}{self.run_timestamp}_{original_filename}"

        output_directory = ""
        if self.config.uvtools_output_location == "input_folder":
            output_directory = os.path.dirname(self.config.uvtools_input_file)
        else: # Default to working_folder
            output_directory = self.config.uvtools_temp_folder

        final_output_path = os.path.join(output_directory, output_filename)

        command = [
            self.config.uvtools_path,
            "run",
            self.config.uvtools_input_file,
            uvtop_filepath,
            "--output",
            final_output_path
        ]
        self.status_update.emit(f"Running command: {' '.join(command)}")

        try:
            creation_flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            process = subprocess.run(command, capture_output=True, text=True, check=True, creationflags=creation_flags)
            self.status_update.emit(f"Successfully created: {output_filename}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"UVTools exited with an error (code {e.returncode}):\n\n{e.stderr}")
        except Exception as e:
            raise RuntimeError(f"UVTools repacking failed: {e}")

    def run(self):
        try:
            input_path = ""
            processing_output_path = ""

            if self.config.input_mode == "uvtools":
                input_path = self._run_uvtools_extraction()
                # For now, assume output is same as input until processing is implemented
                processing_output_path = input_path
            else:
                input_path = self.config.input_folder
                processing_output_path = self.config.output_folder


            self.status_update.emit("Starting Dask client...")
            self.dask_engine.start_dask_client()

            self.status_update.emit("Loading images to Dask array...")
            self.dask_engine.load_images_to_dask_array(input_path)

            # Placeholder for actual processing
            self.status_update.emit("Processing...")
            self.progress_update.emit(50)

            # In a real application, you would now extract planes and apply filters.
            # For this example, we'll just wait a bit.
            self.sleep(2)

            self.progress_update.emit(100)
            self.status_update.emit("Processing complete.")

            if self.config.input_mode == "uvtools" and self._is_running:
                uvtop_file = self._generate_uvtop_file(processing_output_path)
                self._run_uvtools_repack(uvtop_file, processing_output_path)

        except Exception as e:
            self.error_signal.emit(str(e))
        finally:
            if self.dask_engine:
                self.dask_engine.stop_dask_client()
            self.finished_signal.emit()

    def stop(self):
        self._is_running = False
        self.status_update.emit("Stopping...")

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Modular Dask Engine")
        self.settings = QSettings("MyCompany", "ModularDaskEngine")
        self.processor_thread = None
        self.init_ui()
        self.load_settings()
        self._connect_signals()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        self.main_processing_tab = QWidget()
        main_processing_layout = QVBoxLayout(self.main_processing_tab)
        self.tab_widget.addTab(self.main_processing_tab, "Main Processing")

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

        self.io_stacked_widget.addWidget(uvtools_mode_widget)

        main_processing_layout.addWidget(io_group)

        # --- General Settings Section ---
        general_group = QGroupBox("General")
        general_layout = QVBoxLayout(general_group)

        thread_layout = QHBoxLayout()
        thread_layout.addWidget(QLabel("Thread Count:"))
        self.thread_count_edit = QLineEdit("4")
        self.thread_count_edit.setFixedWidth(60)
        thread_layout.addWidget(self.thread_count_edit)
        thread_layout.addStretch(1)
        general_layout.addLayout(thread_layout)

        main_processing_layout.addWidget(general_group)
        main_processing_layout.addStretch(1)

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
        self.input_mode_group.idClicked.connect(self.on_input_mode_changed)
        self.input_folder_button.clicked.connect(lambda: self.browse_folder(self.input_folder_edit))
        self.output_folder_button.clicked.connect(lambda: self.browse_folder(self.output_folder_edit))
        self.uvtools_path_button.clicked.connect(lambda: self.browse_file(self.uvtools_path_edit, "Select UVToolsCmd.exe", "Executable Files (*.exe)"))
        self.uvtools_temp_folder_button.clicked.connect(lambda: self.browse_folder(self.uvtools_temp_folder_edit))
        self.uvtools_input_file_button.clicked.connect(lambda: self.browse_file(self.uvtools_input_file_edit, "Select Input Slice File"))
        self.start_stop_button.clicked.connect(self.toggle_processing)

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
        self.output_prefix_edit.setText(app_config.output_file_prefix)
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
        app_config.output_file_prefix = self.output_prefix_edit.text()
        try:
            app_config.thread_count = int(self.thread_count_edit.text())
        except ValueError:
            app_config.thread_count = 4 # default

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
        self.save_settings()

        # Basic input validation
        if app_config.input_mode == "folder" and not app_config.input_folder:
            self.show_error_message("Input Error", "Please select an input folder.")
            return

        self.set_ui_enabled(False)
        self.processor_thread = ProcessingThread(app_config)
        self.processor_thread.status_update.connect(self.update_status)
        self.processor_thread.progress_update.connect(self.progress_bar.setValue)
        self.processor_thread.error_signal.connect(self.show_error)
        self.processor_thread.finished_signal.connect(self.processing_finished)
        self.processor_thread.start()

    @Slot(str)
    def update_status(self, message):
        self.status_label.setText(f"Status: {message}")

    @Slot(str)
    def show_error(self, message):
        self.show_error_message("Processing Error", message)
        self.processing_finished()

    @Slot()
    def processing_finished(self):
        self.status_label.setText("Status: Finished or Stopped.")
        self.set_ui_enabled(True)
        self.processor_thread = None

    def show_error_message(self, title, text):
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle(title)
        msg_box.setText(text)
        msg_box.exec()

    def set_ui_enabled(self, enabled):
        """Toggles the enabled state of all UI widgets."""
        self.tab_widget.setEnabled(enabled)
        self.start_stop_button.setEnabled(True)
        if not enabled:
            self.start_stop_button.setText("Stop Processing")
        else:
            self.start_stop_button.setText("Start Processing")
