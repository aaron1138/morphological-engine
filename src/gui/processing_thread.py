# src/gui/processing_thread.py

from PySide6.QtCore import QThread, Signal
from src.utils.config import Config
from src.engine.dask_engine import DaskEngine
from src.utils.uvtools import run_uvtools_extraction
import datetime
import shutil
import os

class ProcessingThread(QThread):
    """
    Manages the Dask operations in a separate thread.
    """
    status_update = Signal(str)
    progress_update = Signal(int)
    error_signal = Signal(str)
    finished_signal = Signal()

    def __init__(self, app_config: Config):
        super().__init__()
        self.app_config = app_config
        self._is_running = True
        self.run_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.session_temp_folder = ""

    def run(self):
        """The main processing loop."""
        try:
            self.status_update.emit("Processing started...")

            input_path = ""
            if self.app_config.input_mode == "uvtools":
                self.status_update.emit("Extracting images with UVTools...")
                input_path = run_uvtools_extraction(
                    self.app_config.uvtools_path,
                    self.app_config.uvtools_input_file,
                    self.app_config.uvtools_temp_folder,
                    self.run_timestamp
                )
                self.session_temp_folder = os.path.dirname(input_path)
                self.status_update.emit(f"Images extracted to: {input_path}")
            else:
                input_path = self.app_config.input_folder

            self.status_update.emit("Loading image stack with Dask...")
            engine = DaskEngine()

            chunks = (
                self.app_config.dask_chunk_size_z,
                self.app_config.dask_chunk_size_y,
                self.app_config.dask_chunk_size_x,
            )

            stack = engine.load_image_stack(
                input_folder=input_path,
                chunks=chunks
            )

            # For now, we just load the stack. In the future, we would
            # apply filters here.
            self.status_update.emit("Dask array created successfully.")
            self.progress_update.emit(50) # Placeholder progress

            # Here you would typically compute something.
            # For this task, we are just setting up the engine.
            # Let's compute the mean of the stack as a dummy operation.
            self.status_update.emit("Performing a dummy computation on the Dask array...")
            mean_value = stack.mean().compute()
            self.status_update.emit(f"Dummy computation result (mean pixel value): {mean_value:.2f}")

            self.progress_update.emit(100)
            self.status_update.emit("Processing complete!")

        except Exception as e:
            import traceback
            error_info = f"Error in processing thread: {e}\n\n{traceback.format_exc()}"
            self.error_signal.emit(error_info)
        finally:
            if self.app_config.input_mode == "uvtools" and self.app_config.uvtools_delete_temp_on_completion:
                if self.session_temp_folder and os.path.isdir(self.session_temp_folder):
                    self.status_update.emit(f"Deleting temporary folder: {self.session_temp_folder}")
                    try:
                        shutil.rmtree(self.session_temp_folder)
                        self.status_update.emit("Temporary files deleted.")
                    except Exception as e:
                        self.error_signal.emit(f"Could not delete temp folder: {e}")

            self.finished_signal.emit()

    def stop(self):
        self._is_running = False
        self.status_update.emit("Stopping...")
