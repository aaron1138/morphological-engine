# -*- coding: utf-8 -*-
"""
Module: processing_thread.py
Author: Jules (Refactored for GPU Pipeline)
Description: A QThread subclass for running the core processing engine in the
             background to prevent the GUI from freezing. This version uses
             the new GPU-based processing pipeline.
"""

import cv2
import numpy as np
import openvdb
from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal
from typing import Dict, Any

# --- Core Engine Imports ---
from core.slice_loader import SliceLoader
from core.voxel_engine import VoxelEngine
from core.gpu_processor import GpuProcessor
from core.gpu_pipeline import GpuPipeline
from utils.app_settings import AppSettings

class ProcessingThread(QThread):
    """
    Runs the full GPU-based voxel processing pipeline in a separate thread.
    """
    progress_update = pyqtSignal(int, int)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, slice_loader: SliceLoader, config: Dict[str, Any],
                 output_path: str, app_settings: AppSettings,
                 save_debug: bool, window_size: int = 5):
        super().__init__()
        self.slice_loader = slice_loader
        self.config = config
        self.output_path = Path(output_path)
        self.app_settings = app_settings
        self.save_debug = save_debug
        self.window_size = window_size
        self.center_slice_offset = self.window_size // 2
        self.gpu_processor = None

    def run(self):
        """The main work of the thread is done here."""
        try:
            print("GPU processing thread started.")

            self.output_path.mkdir(exist_ok=True)
            # Debug path creation is removed for now, as debug handling needs redesign.

            # --- Initialize GPU and Voxel Engine ---
            self.gpu_processor = GpuProcessor(self.app_settings)
            engine = VoxelEngine(self.slice_loader, self.window_size)
            pipeline = GpuPipeline(self.config, self.gpu_processor)

            num_windows = len(self.slice_loader) - self.window_size + 1
            all_slice_paths = self.slice_loader.get_slice_list()

            for i, voxel_grid in enumerate(engine.iter_windows()):
                center_slice_index_global = i + self.center_slice_offset
                original_path = all_slice_paths[center_slice_index_global]

                # --- Run the GPU pipeline ---
                # The GPU pipeline directly transforms the data. The old concept of a
                # "modifier mask" is no longer applicable.
                processed_grid, _ = pipeline.run(voxel_grid, debug=self.save_debug)

                # --- Convert result back to image slice for saving ---
                # Create a NumPy array to copy the grid data into
                processed_np = np.zeros((self.window_size, engine.height, engine.width), dtype=np.float32)
                processed_grid.copyToArray(processed_np)

                # Extract the center slice
                output_slice_float = processed_np[self.center_slice_offset]

                # Normalize from [0.0, 1.0] float to [0, 255] uint8 for saving
                output_slice_uint8 = np.clip(output_slice_float * 255.0, 0, 255).astype(np.uint8)

                # Save the final slice
                output_filepath = self.output_path / original_path.name
                cv2.imwrite(str(output_filepath), output_slice_uint8)

                self.progress_update.emit(i + 1, num_windows)

            self.finished.emit()
            print("Processing thread finished successfully.")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"An error occurred in the processing thread: {e}")
            self.error.emit(str(e))
        finally:
            # --- CRITICAL: Release GPU resources ---
            if self.gpu_processor:
                self.gpu_processor.destroy()
                print("GPU resources released.")
