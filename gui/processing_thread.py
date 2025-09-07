# -*- coding: utf-8 -*-
"""
Module: processing_thread.py
Author: Gemini
Description: A QThread subclass for running the core processing engine in the
             background to prevent the GUI from freezing. This version is
             updated to use the GPU processing pipeline.
"""

import cv2
import numpy as np
from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal
from typing import Dict, Any

# --- Core Engine Imports ---
from core.slice_loader import SliceLoader
from core.voxel_engine import VoxelEngine
from gpu.gpu_manager import GpuManager
from gpu.gpu_processing_pipeline import GpuProcessingPipeline

# Try to import pyopenvdb and handle the error gracefully
try:
    import pyopenvdb as vdb
    PYOPENVDB_INSTALLED = True
except ImportError:
    PYOPENVDB_INSTALLED = False
    vdb = None

class ProcessingThread(QThread):
    """
    Runs the full voxel processing pipeline in a separate thread and saves results.
    """
    progress_update = pyqtSignal(int, int)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, slice_loader: SliceLoader, config: Dict[str, Any], output_path: str, save_debug: bool, window_size: int = 5):
        super().__init__()
        self.slice_loader = slice_loader
        self.config = config
        self.output_path = Path(output_path)
        self.save_debug = save_debug # Note: GPU pipeline does not yet support debug steps
        self.window_size = window_size
        self.center_slice_offset = self.window_size // 2

    def run(self):
        """The main work of the thread is done here."""
        gpu_manager = None # Define here for finally block
        try:
            print("Processing thread started (GPU Pipeline).")

            if not PYOPENVDB_INSTALLED:
                raise ImportError("pyopenvdb is not installed. Cannot run the processing pipeline.")

            # --- Initialize GPU resources ---
            gpu_manager = GpuManager()
            if not gpu_manager.ctx:
                raise RuntimeError("Failed to initialize GPU Manager. Cannot run processing.")

            pipeline = GpuProcessingPipeline(gpu_manager)

            # --- Setup output directories ---
            self.output_path.mkdir(exist_ok=True)
            if self.save_debug:
                print("Warning: Debug step saving is not yet implemented for the GPU pipeline.")

            # --- Main processing loop ---
            engine = VoxelEngine(self.slice_loader, self.window_size)
            num_windows = len(self.slice_loader) - self.window_size + 1
            all_slice_paths = self.slice_loader.get_slice_list()

            # --- Simplification: Use the first step in the config to select the shader ---
            if not self.config.get("steps"):
                raise ValueError("Processing configuration has no steps.")
            shader_name = self.config["steps"][0].get("operation", "passthrough")
            shader_path = f"shaders/{shader_name}.comp"
            print(f"Using shader: {shader_path}")

            for i, voxel_grid in enumerate(engine.iter_windows()):
                center_slice_index_global = i + self.center_slice_offset
                original_path = all_slice_paths[center_slice_index_global]

                # Run the GPU pipeline
                # For now, this is a placeholder that returns the input grid.
                processed_grid = pipeline.run(voxel_grid, shader_path)

                if processed_grid is None:
                    # In the future, the pipeline will do real work. For now, we just use the input.
                    print("Pipeline returned None, using original grid for output.")
                    processed_grid = voxel_grid

                # --- Convert result back to numpy for saving ---
                # The output of the pipeline is an OpenVDB grid. We need the center slice as a numpy array.
                # 1. Create a dense numpy array to hold the grid data
                dims = processed_grid.eval_active_voxel_dim()
                dense_array = np.zeros((self.window_size, dims[1], dims[0]), dtype=np.float32)
                processed_grid.copy_to_dense(dense_array)

                # 2. Extract the center slice
                processed_slice_float = dense_array[self.center_slice_offset]

                # 3. Denormalize from [0, 1] float to [0, 255] uint8
                processed_slice_uint8 = (processed_slice_float * 255).astype(np.uint8)

                # --- Blending Logic (simplified) ---
                # For now, we just save the processed slice directly without blending.
                # Blending would require loading the original slice again.
                output_filepath = self.output_path / original_path.name
                cv2.imwrite(str(output_filepath), processed_slice_uint8)

                self.progress_update.emit(i + 1, num_windows)

            self.finished.emit()
            print("Processing thread finished successfully.")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"An error occurred in the processing thread: {e}")
            self.error.emit(str(e))
        finally:
            # --- Cleanup GPU resources ---
            if gpu_manager and gpu_manager.ctx:
                print("Cleaning up GPU resources from processing thread.")
                gpu_manager.cleanup()
