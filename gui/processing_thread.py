# -*- coding: utf-8 -*-
"""
Module: processing_thread.py
Author: Gemini
Description: A QThread subclass for running the RawGL controller in the
             background to prevent the GUI from freezing.
"""

from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal
from typing import Dict, Any

# --- Core Engine Imports ---
from core.slice_loader import SliceLoader
from core.rawgl_controller import RawGLController

class ProcessingThread(QThread):
    """
    Runs the RawGL processing pipeline in a separate thread.
    """
    progress_update = pyqtSignal(int, int)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, slice_loader: SliceLoader, config: Dict[str, Any], output_path: str, save_debug: bool):
        super().__init__()
        self.slice_loader = slice_loader
        self.config = config
        self.output_path = Path(output_path)
        # save_debug is not used by RawGL in this implementation
        self.save_debug = save_debug

    def run(self):
        """The main work of the thread is done here."""
        try:
            print("Processing thread started (RawGL Pipeline).")

            # --- Get settings from the config dictionary ---
            rawgl_path = self.config.get("rawgl_path")
            if not rawgl_path or not Path(rawgl_path).exists():
                raise ValueError("RawGL executable path is not set or is invalid.")

            output_channels = self.config.get("output_channels", 1)
            output_bits = self.config.get("output_bits", 8)

            if not self.config.get("steps"):
                raise ValueError("Processing pipeline has no steps.")

            # --- Main processing loop ---
            # For each step in the pipeline, run all slices through it.
            # The output of one step becomes the input for the next.

            input_paths = self.slice_loader.get_slice_list()
            num_steps = len(self.config["steps"])
            total_images_to_process = len(input_paths) * num_steps
            images_processed = 0

            for i, step in enumerate(self.config["steps"]):
                is_last_step = (i == num_steps - 1)

                shader_name = step.get("operation")
                if not shader_name:
                    print(f"Skipping step {i+1} due to missing 'operation'.")
                    continue

                shader_path = Path(f"shaders/{shader_name}.comp")
                if not shader_path.exists():
                    raise FileNotFoundError(f"Shader file not found: {shader_path}")

                # Determine the output directory for this step
                if is_last_step:
                    current_output_dir = self.output_path
                else:
                    # Create a temporary directory for intermediate files
                    current_output_dir = self.output_path / f"temp_{i}"
                current_output_dir.mkdir(exist_ok=True)

                print(f"\n--- Running Pipeline Step {i+1}/{num_steps}: Shader '{shader_name}' ---")

                controller = RawGLController(rawgl_path=rawgl_path)

                # Define a progress callback for this step
                def step_progress_callback(completed, total):
                    nonlocal images_processed
                    images_processed += 1
                    self.progress_update.emit(images_processed, total_images_to_process)

                # The process_batch method handles the multi-threading
                controller.process_batch(
                    image_paths=input_paths,
                    output_dir=current_output_dir,
                    shader_path=shader_path,
                    channels=output_channels,
                    bits=output_bits,
                    progress_callback=step_progress_callback
                )

                # The output of this step is the input for the next step
                input_paths = sorted(list(current_output_dir.glob("*.png")))

            self.finished.emit()
            print("\nProcessing thread finished successfully.")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"An error occurred in the processing thread: {e}")
            self.error.emit(str(e))
        finally:
            # Cleanup temporary directories
            for i in range(num_steps - 1):
                temp_dir = self.output_path / f"temp_{i}"
                if temp_dir.exists():
                    try:
                        for f in temp_dir.glob("*"):
                            f.unlink()
                        temp_dir.rmdir()
                        print(f"Cleaned up temporary directory: {temp_dir}")
                    except Exception as e:
                        print(f"Error cleaning up temp directory {temp_dir}: {e}")
