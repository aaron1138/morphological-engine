# -*- coding: utf-8 -*-
"""
Module: processing_thread.py
Author: Jules (Refactored for Dask)
Description: A QThread subclass that uses a Dask cluster to run the RawGL
             pipeline in parallel on orthogonal slices of a 3D volume.
"""

from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal
from typing import Dict, Any
from concurrent.futures import as_completed
import dask
from dask.distributed import Client, LocalCluster

# --- Core Engine Imports ---
from core.slice_loader import SliceLoader
from core.dask_volume import DaskVolume
from core.rawgl_pipeline import RawGlPipeline
from utils.app_settings import AppSettings

class ProcessingThread(QThread):
    """
    Uses a Dask cluster to run the RawGL pipeline in parallel.
    """
    progress_update = pyqtSignal(int, int)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, slice_loader: SliceLoader, config: Dict[str, Any],
                 output_path: str, app_settings: AppSettings):
        super().__init__()
        self.slice_loader = slice_loader
        self.config = config
        self.output_path = Path(output_path)
        self.app_settings = app_settings

    def run(self):
        """The main work of the thread is done here."""
        cluster = None
        client = None
        try:
            worker_count = self.config.get("dask_worker_count", 4)
            print(f"Initializing Dask cluster with {worker_count} workers.")

            # Set up a local Dask cluster
            cluster = LocalCluster(n_workers=worker_count, threads_per_worker=1)
            client = Client(cluster)

            print(f"Dask dashboard available at: {client.dashboard_link}")

            # --- Prepare Pipelines and Data ---
            rawgl_path = self.app_settings.get("rawgl_executable_path")
            pipeline = RawGlPipeline(self.config, rawgl_path)

            dask_volume = DaskVolume(self.slice_loader)

            # --- Define Tasks for Orthogonal Slices ---
            tasks = []

            # Create output directories for the different views
            xz_output_dir = self.output_path / "xz_slices"
            yz_output_dir = self.output_path / "yz_slices"
            xz_output_dir.mkdir(exist_ok=True)
            yz_output_dir.mkdir(exist_ok=True)

            # Generate tasks for XZ slices (iterating through height)
            for y in range(dask_volume.shape[1]):
                slice_data = dask_volume.get_xz_slice(y)
                output_path = xz_output_dir / f"xz_slice_{y:04d}.png"
                tasks.append((slice_data, str(output_path)))

            # Generate tasks for YZ slices (iterating through width)
            for x in range(dask_volume.shape[2]):
                slice_data = dask_volume.get_yz_slice(x)
                output_path = yz_output_dir / f"yz_slice_{x:04d}.png"
                tasks.append((slice_data, str(output_path)))

            if not tasks:
                raise ValueError("No processing tasks were generated.")

            print(f"Submitting {len(tasks)} tasks to Dask cluster...")

            # --- Execute Tasks in Parallel ---
            # Map the pipeline's run function over the tasks
            # Dask will compute the slice_data for each task before passing it
            # to the pipeline.run function.
            futures = client.map(pipeline.run, *zip(*tasks))

            # --- Update Progress ---
            completed_count = 0
            total_tasks = len(tasks)
            for future in as_completed(futures):
                try:
                    future.result() # Check for exceptions from the task
                except Exception as e:
                    print(f"A Dask worker failed: {e}")
                completed_count += 1
                self.progress_update.emit(completed_count, total_tasks)

            self.finished.emit()
            print("Dask processing thread finished successfully.")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))
        finally:
            # --- CRITICAL: Shut down the Dask cluster ---
            if client:
                client.close()
            if cluster:
                cluster.close()
            print("Dask cluster shut down.")
