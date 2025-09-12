import traceback
import shutil
from PySide6.QtCore import QThread, Signal

from utils.config import Config
from utils import uvtools
from core import dask_engine

class ProcessingThread(QThread):
    """
    Manages the Dask processing pipeline in a separate thread to keep the GUI responsive.
    """
    # Signal to update the status label in the main window
    status_update = Signal(str)
    # Signal to update the progress bar (current_value, max_value)
    progress_update = Signal(int, int)
    # Signal to report a processing error
    error_signal = Signal(str)
    # Signal indicating the thread has finished its work
    finished_signal = Signal()

    def __init__(self, app_config: Config, axis_to_extract: int):
        """
        Args:
            app_config: The application configuration dataclass.
            axis_to_extract: The axis to slice along (0=XY, 1=XZ, 2=YZ).
        """
        super().__init__()
        self.app_config = app_config
        self.axis_to_extract = axis_to_extract
        self._is_running = True

    def run(self):
        """
        The main processing loop executed in the background thread.
        """
        self.status_update.emit("Processing started...")
        temp_extraction_folder = None

        try:
            # --- Step 1: Determine Input Folder ---
            if self.app_config.input_mode == "uvtools":
                self.status_update.emit("Starting UVTools slice extraction...")
                temp_extraction_folder = uvtools.extract_layers_with_uvtools(
                    uvtools_exe_path=self.app_config.uvtools_path,
                    input_slice_file=self.app_config.uvtools_input_file,
                    temp_base_folder=self.app_config.uvtools_temp_folder
                )
                input_path = temp_extraction_folder
                self.status_update.emit("UVTools extraction complete.")
            else:
                input_path = self.app_config.input_folder

            if not self._is_running: return

            # --- Step 2: Load Image Stack with Dask ---
            self.status_update.emit("Loading image stack into Dask array...")
            dask_array = dask_engine.load_image_stack(input_path)
            self.status_update.emit("Dask array created successfully.")

            if not self._is_running: return

            # --- Step 3: Extract and Save Orthogonal Planes ---
            axis_names = {0: "XY", 1: "XZ", 2: "YZ"}
            plane_name = axis_names[self.axis_to_extract]
            self.status_update.emit(f"Extracting {plane_name} planes...")

            # Create a dedicated output folder for this extraction
            output_subfolder = f"{self.app_config.output_file_prefix}{plane_name}"
            final_output_path = f"{self.app_config.output_folder}/{output_subfolder}"

            # Define a progress callback to link Dask engine progress to the GUI
            def progress_callback(current, total):
                if not self._is_running:
                    # This is a soft stop; it won't kill the current dask compute
                    # but will prevent the next one from starting.
                    raise InterruptedError("Processing stopped by user.")
                self.progress_update.emit(current, total)

            dask_engine.save_orthogonal_planes(
                dask_array=dask_array,
                output_folder=final_output_path,
                axis=self.axis_to_extract,
                progress_callback=progress_callback
            )
            self.status_update.emit(f"Successfully saved planes to: {final_output_path}")

        except InterruptedError as e:
            self.status_update.emit(str(e))
        except Exception as e:
            # Format a detailed error message with traceback
            error_info = f"An error occurred in the processing thread:\n\n{str(e)}\n\n{traceback.format_exc()}"
            self.error_signal.emit(error_info)
        finally:
            # --- Step 4: Cleanup ---
            if temp_extraction_folder and self.app_config.uvtools_delete_temp_on_completion:
                self.status_update.emit(f"Deleting temporary folder: {temp_extraction_folder}")
                try:
                    shutil.rmtree(temp_extraction_folder)
                    self.status_update.emit("Temporary files deleted.")
                except Exception as e:
                    self.error_signal.emit(f"Could not delete temp folder: {e}")

            if self._is_running:
                 self.status_update.emit("Processing complete!")

            self.finished_signal.emit()

    def stop(self):
        """
        Signals the processing thread to stop. The stop is cooperative,
        meaning the thread will finish its current task before exiting the loop.
        """
        self.status_update.emit("Stopping process...")
        self._is_running = False
