from PySide6.QtCore import QThread, Signal
import os
import time
import imageio.v2 as imageio
import numpy as np
import dask

from dask_engine import load_image_stack, get_xy_plane, get_xz_plane, get_yz_plane
from uvtools import extract_layers

class ProcessingThread(QThread):
    status_update = Signal(str)
    progress_update = Signal(int)
    error_signal = Signal(str)
    finished_signal = Signal()

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._is_running = True

    def run(self):
        try:
            self.status_update.emit("Starting processing...")
            self.progress_update.emit(0)

            with dask.config.set(scheduler='threads', num_workers=self.config['num_threads']):
                input_folder = self.config["input_folder"]
                output_folder = self.config["output_folder"]
                chunk_size = self.config["chunk_size"]

                if self.config["mode"] == "uvtools":
                    self.status_update.emit("Extracting layers with UVTools...")
                    if not os.path.exists(self.config["uvtools_temp_folder"]):
                        os.makedirs(self.config["uvtools_temp_folder"])

                    input_folder = extract_layers(
                        self.config["uvtools_path"],
                        self.config["uvtools_input_file"],
                        self.config["uvtools_temp_folder"]
                    )
                    self.status_update.emit(f"Layers extracted to {input_folder}")

                self.progress_update.emit(10)

                if not self._is_running:
                    self.status_update.emit("Processing stopped.")
                    return

                self.status_update.emit("Loading image stack with Dask...")
                dask_stack = load_image_stack(input_folder, chunk_size=chunk_size)
                self.status_update.emit(f"Dask array created with shape {dask_stack.shape} and chunk size {dask_stack.chunksize}")
                self.progress_update.emit(25)

                if not self._is_running:
                    self.status_update.emit("Processing stopped.")
                    return

                z_mid = dask_stack.shape[0] // 2
                y_mid = dask_stack.shape[1] // 2
                x_mid = dask_stack.shape[2] // 2

                self.status_update.emit(f"Extracting and saving XY plane at Z={z_mid}...")
                xy_plane = get_xy_plane(dask_stack, z_mid).compute()
                imageio.imwrite(os.path.join(output_folder, "plane_xy.png"), xy_plane)
                self.progress_update.emit(50)

                if not self._is_running:
                    self.status_update.emit("Processing stopped.")
                    return

                self.status_update.emit(f"Extracting and saving XZ plane at Y={y_mid}...")
                xz_plane = get_xz_plane(dask_stack, y_mid).compute()
                imageio.imwrite(os.path.join(output_folder, "plane_xz.png"), xz_plane)
                self.progress_update.emit(75)

                if not self._is_running:
                    self.status_update.emit("Processing stopped.")
                    return

                self.status_update.emit(f"Extracting and saving YZ plane at X={x_mid}...")
                yz_plane = get_yz_plane(dask_stack, x_mid).compute()
                imageio.imwrite(os.path.join(output_folder, "plane_yz.png"), yz_plane)
                self.progress_update.emit(90)

                self.status_update.emit("Processing complete.")
                self.progress_update.emit(100)

        except Exception as e:
            import traceback
            self.error_signal.emit(f"An error occurred: {e}\n{traceback.format_exc()}")
        finally:
            self.finished_signal.emit()

    def stop(self):
        self.status_update.emit("Stopping process...")
        self._is_running = False
