# core/processing_thread.py

import os
import shutil
import time
import traceback
from PySide6.QtCore import QThread, Signal
from dask.distributed import Client, LocalCluster
import dask.array as da

from core import engine
from utils import uvtools
from config import app_config, OperationPlane

class ProcessingThread(QThread):
    """
    Manages the Dask processing in a separate thread.
    Reads its configuration from the global `app_config` object.
    """
    status_update = Signal(str)
    progress_update = Signal(int) # Will be used for overall pipeline progress
    error_signal = Signal(str)
    finished_signal = Signal()

    def __init__(self):
        super().__init__()
        self._is_running = True
        self.session_temp_folder = ""

    def run(self):
        """The main entry point for the processing thread."""
        client = None
        input_image_folder = ""
        is_uvtools_mode = app_config.input_mode == "uvtools"

        try:
            self.status_update.emit("Initializing Dask client...")
            cluster = LocalCluster(n_workers=app_config.worker_count, threads_per_worker=1)
            client = Client(cluster)
            self.status_update.emit(f"Dask client ready. Dashboard at: {client.dashboard_link}")

            if is_uvtools_mode:
                self.session_temp_folder = os.path.join(app_config.uvtools_temp_folder, f"dask_proc_{int(time.time())}")
                self.status_update.emit(f"UVTools Mode: Extracting layers to {self.session_temp_folder}...")
                input_image_folder = uvtools.extract_layers(app_config.uvtools_path, app_config.uvtools_input_file, self.session_temp_folder)
                self.status_update.emit("Layer extraction complete.")
            else:
                input_image_folder = app_config.input_folder

            if not self._is_running: return

            self.status_update.emit("Creating Dask array from images...")
            dask_stack = engine.create_dask_stack(input_image_folder)

            # --- Pipeline Execution ---
            num_ops = len(app_config.pipeline)
            for i, op in enumerate(app_config.pipeline):
                if not self._is_running: break

                self.status_update.emit(f"Pipeline Step {i+1}/{num_ops}: Applying {op.type} on {op.plane} plane...")

                current_stack = dask_stack

                # --- Enhanced EDT ---
                if op.type == "Enhanced EDT":
                    plane_to_axis = { "XY": 0, "XZ": 1, "YZ": 2 }
                    axis = plane_to_axis.get(op.plane, 0)
                    dask_stack = engine.apply_enhanced_edt(current_stack, op, axis=axis)

                # --- Gaussian Blur ---
                elif op.type == "Gaussian Blur":
                    # For 2D filters, we transpose the data so the operation is always on the last two axes
                    if op.plane == "XY": # (Z, Y, X) -> No change needed
                        dask_stack = engine.apply_gaussian_blur(current_stack, op)
                    elif op.plane == "XZ": # (Z, Y, X) -> (Y, Z, X)
                        transposed_stack = current_stack.transpose((1, 0, 2))
                        processed_stack = engine.apply_gaussian_blur(transposed_stack, op)
                        dask_stack = processed_stack.transpose((1, 0, 2)) # Transpose back
                    elif op.plane == "YZ": # (Z, Y, X) -> (X, Y, Z)
                        transposed_stack = current_stack.transpose((2, 1, 0))
                        processed_stack = engine.apply_gaussian_blur(transposed_stack, op)
                        dask_stack = processed_stack.transpose((2, 1, 0)) # Transpose back

                # --- Gaussian Blur ---
                elif op.type == "Gaussian Blur":
                    # For 2D filters, we transpose the data so the operation is always on the last two axes
                    if op.plane == "XY": # (Z, Y, X) -> No change needed
                        dask_stack = engine.apply_gaussian_blur(current_stack, op)
                    elif op.plane == "XZ": # (Z, Y, X) -> (Y, Z, X)
                        transposed_stack = current_stack.transpose((1, 0, 2))
                        processed_stack = engine.apply_gaussian_blur(transposed_stack, op)
                        dask_stack = processed_stack.transpose((1, 0, 2)) # Transpose back
                    elif op.plane == "YZ": # (Z, Y, X) -> (X, Y, Z)
                        transposed_stack = current_stack.transpose((2, 1, 0))
                        processed_stack = engine.apply_gaussian_blur(transposed_stack, op)
                        dask_stack = processed_stack.transpose((2, 1, 0)) # Transpose back

                # --- Blend Modes ---
                elif op.type in ["Multiply", "Screen", "Overlay"]:
                    # These are element-wise, no transposition needed regardless of plane
                    dask_stack = engine.apply_blend_mode(current_stack, op)

                # --- LUT ---
                elif op.type == "Apply LUT":
                    dask_stack = engine.apply_lut(current_stack, op)

                else:
                    self.status_update.emit(f"Warning: Operation type '{op.type}' not yet implemented. Skipping.")

                # Persist the result to memory/disk to avoid recomputing the entire graph
                # This is a key optimization for multi-step pipelines.
                dask_stack = dask_stack.persist()
                self.progress_update.emit(int(((i + 1) / num_ops) * 100))

            if not self._is_running:
                self.status_update.emit("Processing stopped by user.")
            else:
                self.status_update.emit("Pipeline complete. Saving final output...")
                # Here you would typically save the final dask_stack to files
                # For now, we just log completion.
                # Example: da.to_png(dask_stack, os.path.join(app_config.output_folder, 'frame-*.png'))
                self.status_update.emit("Processing finished successfully.")

        except Exception as e:
            error_info = f"An error occurred: {e}\n{traceback.format_exc()}"
            self.error_signal.emit(error_info)
        finally:
            if client:
                self.status_update.emit("Shutting down Dask client...")
                client.close()
                self.status_update.emit("Dask client shut down.")

            if is_uvtools_mode and app_config.uvtools_cleanup and self.session_temp_folder and os.path.isdir(self.session_temp_folder):
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

    def was_stopped(self):
        return not self._is_running
