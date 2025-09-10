# -*- coding: utf-8 -*-
"""
Module: processing_thread.py
Author: Jules (Refactored for Dual Pipelines)
Description: A QThread subclass that acts as a dispatcher for running different
             processing pipelines (e.g., in-process GPU or external tool)
             in the background.
"""

import cv2
import numpy as np
import openvdb
from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# --- Core Engine Imports ---
from core.slice_loader import SliceLoader
from core.voxel_engine import VoxelEngine
from core.gpu_processor import GpuProcessor
from core.gpu_pipeline import GpuPipeline
from core.rawgl_pipeline import RawGlPipeline
from utils.app_settings import AppSettings

class ProcessingThread(QThread):
    """
    Runs the selected processing pipeline in a separate thread.
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

    def run(self):
        """The main work of the thread is done here."""
        try:
            pipeline_mode = self.config.get("pipeline_mode", "ModernGL (In-Process)")
            print(f"Processing thread started with mode: {pipeline_mode}")
            self.output_path.mkdir(exist_ok=True)

            if "ModernGL" in pipeline_mode:
                self._run_moderngl_pipeline()
            elif "RawGL" in pipeline_mode:
                self._run_rawgl_pipeline()
            else:
                raise ValueError(f"Unknown pipeline mode: {pipeline_mode}")

            self.finished.emit()
            print("Processing thread finished successfully.")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

    def _run_moderngl_pipeline(self):
        """Executes the in-process GPU pipeline with OpenVDB."""
        gpu_processor = None
        try:
            gpu_processor = GpuProcessor(self.app_settings)
            engine = VoxelEngine(self.slice_loader, self.window_size)
            pipeline = GpuPipeline(self.config, gpu_processor)
            num_windows = len(self.slice_loader) - self.window_size + 1
            all_slice_paths = self.slice_loader.get_slice_list()

            for i, voxel_grid in enumerate(engine.iter_windows()):
                center_slice_index_global = i + self.center_slice_offset
                original_path = all_slice_paths[center_slice_index_global]
                processed_grid, _ = pipeline.run(voxel_grid, debug=self.save_debug)
                processed_np = np.zeros((self.window_size, engine.height, engine.width), dtype=np.float32)
                processed_grid.copyToArray(processed_np)
                output_slice_float = processed_np[self.center_slice_offset]
                output_slice_uint8 = np.clip(output_slice_float * 255.0, 0, 255).astype(np.uint8)
                output_filepath = self.output_path / original_path.name
                cv2.imwrite(str(output_filepath), output_slice_uint8)
                self.progress_update.emit(i + 1, num_windows)
        finally:
            if gpu_processor:
                gpu_processor.destroy()
                print("GPU resources released.")

    def _run_rawgl_pipeline(self):
        """Executes the external RawGL pipeline using multiple threads."""
        rawgl_path = self.app_settings.get("rawgl_executable_path")
        if not rawgl_path:
            raise ValueError("RawGL executable path is not set in settings.")

        pipeline = RawGlPipeline(self.config, rawgl_path)
        all_slice_paths = self.slice_loader.get_slice_list()
        total_slices = len(all_slice_paths)

        # Use a thread pool to run RawGL processes in parallel
        # Default to the number of CPU cores for thread count
        max_workers = os.cpu_count() or 4
        print(f"Starting RawGL processing with up to {max_workers} parallel threads.")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for slice_path in all_slice_paths:
                input_path = str(self.slice_loader.directory / slice_path.name)
                output_path = str(self.output_path / slice_path.name)
                # Submit the pipeline run to the thread pool
                future = executor.submit(pipeline.run, input_path, output_path)
                futures.append(future)

            # Update progress as tasks complete
            completed_count = 0
            for future in as_completed(futures):
                try:
                    # Retrieve result to raise any exceptions that occurred in the thread
                    future.result()
                except Exception as e:
                    # Log the error but continue processing other images
                    print(f"A RawGL thread failed: {e}")
                    # We can also emit an error signal if we want partial errors reported
                completed_count += 1
                self.progress_update.emit(completed_count, total_slices)
