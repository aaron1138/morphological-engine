import os
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLineEdit,
    QFileDialog,
    QProgressBar,
    QLabel,
    QGroupBox,
    QGridLayout
)
from PySide6.QtCore import QThread, Signal, Slot
from PySide6.QtGui import QIntValidator
import dask
from core.dask_processor import extract_orthogonal_slices
from config import Config, DEFAULT_NUM_WORKERS

class DaskSlicerThread(QThread):
    """
    Manages the Dask processing pipeline in a separate thread.
    """
    status_update = Signal(str)
    progress_update = Signal(int)
    error_signal = Signal(str)
    finished_signal = Signal()

    def __init__(self, input_dir, output_dir, thread_count):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.thread_count = thread_count
        self._is_running = True

    def run(self):
        self.status_update.emit("Starting Dask orthogonal slice extraction...")
        try:
            dask.config.set(scheduler='threads', num_workers=self.thread_count)

            def progress_callback(current, total, message):
                if not self._is_running:
                    raise InterruptedError("Processing stopped by user.")
                self.status_update.emit(message)
                if total > 0:
                    self.progress_update.emit(int((current / total) * 100))

            extract_orthogonal_slices(
                input_dir=self.input_dir,
                output_dir=self.output_dir,
                chunk_size=(64, 64, 64),
                progress_callback=progress_callback
            )
        except InterruptedError as e:
             self.status_update.emit(str(e))
        except Exception as e:
            import traceback
            error_info = f"Error in Dask processing thread: {e}\n{traceback.format_exc()}"
            self.error_signal.emit(error_info)
        finally:
            if self._is_running:
                self.status_update.emit("Processing complete!")
            self.finished_signal.emit()

    def stop_processing(self):
        self._is_running = False

class DaskSlicerPanel(QWidget):
    """
    UI Panel for the Dask Orthogonal Slicer.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.processor_thread = None
        self.init_ui()
        self.connect_signals()

    def init_ui(self):
        layout = QVBoxLayout(self)

        io_group = QGroupBox("I/O")
        io_layout = QGridLayout(io_group)
        io_layout.addWidget(QLabel("Input Folder:"), 0, 0)
        self.input_folder_edit = QLineEdit()
        io_layout.addWidget(self.input_folder_edit, 0, 1)
        self.input_folder_button = QPushButton("Browse...")
        io_layout.addWidget(self.input_folder_button, 0, 2)

        io_layout.addWidget(QLabel("Output Folder:"), 1, 0)
        self.output_folder_edit = QLineEdit()
        io_layout.addWidget(self.output_folder_edit, 1, 1)
        self.output_folder_button = QPushButton("Browse...")
        io_layout.addWidget(self.output_folder_button, 1, 2)
        layout.addWidget(io_group)

        dask_group = QGroupBox("Dask Settings")
        dask_layout = QHBoxLayout(dask_group)
        dask_layout.addWidget(QLabel("Thread Count:"))
        self.thread_count_edit = QLineEdit(str(DEFAULT_NUM_WORKERS))
        self.thread_count_edit.setValidator(QIntValidator(1, 128, self))
        dask_layout.addWidget(self.thread_count_edit)
        dask_layout.addStretch(1)
        layout.addWidget(dask_group)

        layout.addStretch(1)

        self.start_stop_button = QPushButton("Extract Orthogonal Slices")
        self.progress_bar = QProgressBar()
        self.status_label = QLabel("Status: Ready")

        layout.addWidget(self.start_stop_button)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)

    def connect_signals(self):
        self.input_folder_button.clicked.connect(lambda: self.browse_folder(self.input_folder_edit))
        self.output_folder_button.clicked.connect(lambda: self.browse_folder(self.output_folder_edit))
        self.start_stop_button.clicked.connect(self.toggle_processing)

    def browse_folder(self, line_edit):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", line_edit.text())
        if folder: line_edit.setText(folder)

    def toggle_processing(self):
        if self.processor_thread and self.processor_thread.isRunning():
            self.processor_thread.stop_processing()
            self.start_stop_button.setText("Stopping...")
            self.start_stop_button.setEnabled(False)
        else:
            self.start_processing()

    def start_processing(self):
        try:
            input_dir = self.input_folder_edit.text()
            output_dir = self.output_folder_edit.text()
            thread_count = int(self.thread_count_edit.text())

            if not input_dir or not os.path.isdir(input_dir):
                raise ValueError("Input folder must be a valid, existing directory.")
            if not output_dir:
                output_dir = os.path.join(input_dir, "orthogonal_output")
                os.makedirs(output_dir, exist_ok=True)
                self.output_folder_edit.setText(output_dir)
            if not os.path.isdir(output_dir):
                 os.makedirs(output_dir, exist_ok=True)

            self.set_ui_enabled(False)
            self.processor_thread = DaskSlicerThread(input_dir, output_dir, thread_count)
            self.processor_thread.status_update.connect(self.status_label.setText)
            self.processor_thread.progress_update.connect(self.progress_bar.setValue)
            self.processor_thread.error_signal.connect(self.show_error)
            self.processor_thread.finished_signal.connect(self.processing_finished)
            self.processor_thread.start()

        except Exception as e:
            self.show_error(f"Input Error: {e}")
            self.set_ui_enabled(True)

    @Slot()
    def processing_finished(self):
        self.status_label.setText("Status: Finished or Stopped.")
        self.set_ui_enabled(True)
        self.processor_thread = None

    @Slot(str)
    def show_error(self, message):
        self.status_label.setText(f"ERROR: {message}")
        print(message)
        self.set_ui_enabled(True)
        self.processor_thread = None

    def set_ui_enabled(self, enabled):
        self.start_stop_button.setEnabled(True)
        self.start_stop_button.setText("Extract Orthogonal Slices" if enabled else "Stop Processing")
        self.input_folder_edit.setEnabled(enabled)
        self.output_folder_edit.setEnabled(enabled)
        self.thread_count_edit.setEnabled(enabled)
        self.input_folder_button.setEnabled(enabled)
        self.output_folder_button.setEnabled(enabled)
