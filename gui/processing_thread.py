# gui/processing_thread.py
import os
import time
from PySide6.QtCore import QThread, Signal
from core.dask_engine import DaskEngine
from utils.uvtools import UVTools
from utils.config import Config

class ProcessingThread(QThread):
    status_update = Signal(str)
    progress_update = Signal(int)
    finished_signal = Signal()
    error_signal = Signal(str)

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self._is_running = True

    def run(self):
        try:
            self.status_update.emit("Processing started...")

            image_paths = self.get_image_paths()
            if not image_paths:
                self.error_signal.emit("No image files found.")
                return

            dask_engine = DaskEngine()
            dask_array = dask_engine.load_images_to_dask_array(image_paths)

            self.status_update.emit("Dask array created. Shape: {}, Chunks: {}".format(dask_array.shape, dask_array.chunksize))

            # Placeholder for processing logic
            # Here, we will just compute the mean of the array to trigger computation
            self.status_update.emit("Performing placeholder computation...")
            mean_value = dask_array.mean().compute()
            self.status_update.emit(f"Mean value of the dask array: {mean_value}")

            # Simulate a longer process with progress updates
            for i in range(101):
                if not self._is_running:
                    self.status_update.emit("Processing stopped by user.")
                    break
                time.sleep(0.05) # Simulate work
                self.progress_update.emit(i)

            if self._is_running:
                self.status_update.emit("Processing complete!")

        except Exception as e:
            self.error_signal.emit(str(e))
        finally:
            self.finished_signal.emit()

    def get_image_paths(self):
        if self.config.input_mode == "uvtools":
            self.status_update.emit("Using UVTools to extract images...")
            uvtools = UVTools(self.config.uvtools_path)
            temp_folder = os.path.join(self.config.uvtools_temp_folder, "extracted_images")
            uvtools.extract(self.config.uvtools_input_file, temp_folder)
            image_paths = [os.path.join(temp_folder, f) for f in os.listdir(temp_folder) if f.lower().endswith('.png')]
        else:
            self.status_update.emit("Reading images from folder...")
            image_paths = [os.path.join(self.config.input_folder, f) for f in os.listdir(self.config.input_folder) if f.lower().endswith('.png')]

        return sorted(image_paths)

    def stop(self):
        self._is_running = False
