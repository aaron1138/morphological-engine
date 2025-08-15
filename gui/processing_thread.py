# -*- coding: utf-8 -*-
"""
Module: processing_thread.py
Author: Gemini
Description: A QThread subclass for running the core processing engine in the
             background to prevent the GUI from freezing.
"""

import cv2
import numpy as np
from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal
from typing import Dict, Any

# --- Core Engine Imports ---
from core.slice_loader import SliceLoader
from core.voxel_engine import VoxelEngine
from core.processing_pipeline import ProcessingPipeline

class ProcessingThread(QThread):
    """
    Runs the full voxel processing pipeline in a separate thread and saves results.
    """
    progress_update = pyqtSignal(int, int)
    # Finished signal no longer needs to carry data, just indicates completion.
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, slice_loader: SliceLoader, config: Dict[str, Any], output_path: str, save_debug: bool, window_size: int = 5):
        super().__init__()
        self.slice_loader = slice_loader
        self.config = config
        self.output_path = Path(output_path)
        self.save_debug = save_debug
        self.window_size = window_size
        self.center_slice_offset = self.window_size // 2

    def run(self):
        """The main work of the thread is done here."""
        try:
            print("Processing thread started.")
            
            # Create output directories
            self.output_path.mkdir(exist_ok=True)
            debug_path = self.output_path / "debug"
            if self.save_debug:
                debug_path.mkdir(exist_ok=True)

            engine = VoxelEngine(self.slice_loader, self.window_size)
            pipeline = ProcessingPipeline(self.config)

            num_windows = len(self.slice_loader) - self.window_size + 1
            all_slice_paths = self.slice_loader.get_slice_list()

            for i, voxel_window in enumerate(engine.iter_windows()):
                # Determine the original filename for the center slice of this window
                center_slice_index_global = i + self.center_slice_offset
                original_path = all_slice_paths[center_slice_index_global]
                
                # Run the pipeline to get the modifier mask and any debug steps
                modifier_window, debug_windows = pipeline.run(voxel_window, debug=self.save_debug)
                
                # --- NEW: BLENDING LOGIC ---
                # Get the original center slice from the input window
                original_slice = voxel_window[self.center_slice_offset]
                # Get the processed modifier mask from the pipeline's final output window
                modifier_mask = modifier_window[self.center_slice_offset]
                # Combine them: Add the white pixels from the mask to the original slice
                blended_slice = np.maximum(original_slice, modifier_mask)
                
                # Save the final BLENDED slice
                output_filepath = self.output_path / original_path.name
                cv2.imwrite(str(output_filepath), blended_slice)

                # Save debug images if requested (these remain un-blended)
                if self.save_debug:
                    for name, debug_window in debug_windows:
                        debug_slice = debug_window[self.center_slice_offset]
                        debug_filename = f"{original_path.stem}_{name}.png"
                        debug_filepath = debug_path / debug_filename
                        cv2.imwrite(str(debug_filepath), debug_slice)

                self.progress_update.emit(i + 1, num_windows)

            self.finished.emit()
            print("Processing thread finished successfully.")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"An error occurred in the processing thread: {e}")
            self.error.emit(str(e))
