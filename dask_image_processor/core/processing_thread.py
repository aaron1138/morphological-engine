# processing_thread.py

import os
import shutil
import time
from PySide6.QtCore import QThread, Signal
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar

from core import engine
from utils import uvtools

class ProcessingThread(QThread):
    """
    Manages the Dask processing in a separate thread.
    """
    status_update = Signal(str)
    progress_update = Signal(int)
    error_signal = Signal(str)
    finished_signal = Signal()

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self._is_running = True
        self.session_temp_folder = ""

    def run(self):
        """The main entry point for the processing thread."""
        client = None
        input_image_folder = ""
        is_uvtools_mode = self.config.get("input_mode") == "uvtools"

        try:
            self.status_update.emit("Initializing Dask client...")
            cluster = LocalCluster(n_workers=self.config['worker_count'], threads_per_worker=1)
            client = Client(cluster)
            self.status_update.emit(f"Dask client ready. Dashboard at: {client.dashboard_link}")

            if is_uvtools_mode:
                self.session_temp_folder = os.path.join(self.config['uvtools_temp_folder'], f"dask_proc_{int(time.time())}")
                self.status_update.emit(f"UVTools Mode: Extracting layers to {self.session_temp_folder}...")
                input_image_folder = uvtools.extract_layers(
                    self.config['uvtools_path'],
                    self.config['uvtools_input_file'],
                    self.session_temp_folder
                )
                self.status_update.emit("Layer extraction complete.")
            else:
                input_image_folder = self.config['input_folder']

            if not self._is_running: return

            self.status_update.emit("Creating Dask array from images...")
            dask_stack = engine.create_dask_stack(input_image_folder)

            if not self._is_running: return

            self.status_update.emit("Dask array created. Starting sample computation (mean)...")

            # Use Dask's progress bar to report progress
            pbar = ProgressBar()
            pbar.register()

            # A simple computation to verify the whole stack can be processed
            result = dask_stack.mean().compute()

            pbar.unregister()

            self.status_update.emit(f"Sample computation complete. Mean pixel value: {result:.2f}")

            # Placeholder for future processing steps
            # For example, applying a filter to each XY slice:
            # processed_stack = dask_stack.map_blocks(engine.process_xy_slice, dtype=dask_stack.dtype)
            # self.status_update.emit("Applying XY filter...")
            # processed_stack.persist() # or .compute()

            self.progress_update.emit(100)

        except Exception as e:
            import traceback
            error_info = f"An error occurred: {e}\n{traceback.format_exc()}"
            self.error_signal.emit(error_info)
        finally:
            if client:
                self.status_update.emit("Shutting down Dask client...")
                client.close()
                self.status_update.emit("Dask client shut down.")

            if is_uvtools_mode and self.config['uvtools_cleanup'] and os.path.isdir(self.session_temp_folder):
                self.status_update.emit("Cleaning up temporary files...")
                try:
                    shutil.rmtree(self.session_temp_folder)
                    self.status_update.emit("Temporary files deleted.")
                except Exception as e:
                    self.error_signal.emit(f"Could not delete temp folder: {e}")

            self.finished_signal.emit()

    def stop(self):
        """Signals the thread to stop processing."""
        self._is_running = False
        self.status_update.emit("Stopping process...")
