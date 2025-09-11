import subprocess
import tempfile
import imageio
from pathlib import Path
from typing import Dict, Any

import dask
from dask.distributed import Client, LocalCluster
from PySide6.QtCore import QObject, QThread, Signal

from src.core.dask_grid import DaskGrid

def process_slice_with_rawgl(
    dask_grid: DaskGrid,
    plane: str,
    slice_index: int,
    shader_pass: Dict[str, Any],
    rawgl_executable: str,
    output_dir: Path
) -> str:
    """
    A function designed to be run by a Dask worker.
    It extracts one slice, saves it, runs RawGL, and returns the output path.
    """
    # 1. Get the slice and compute it into a numpy array
    slice_array = dask_grid.get_slice(plane, slice_index).compute()

    # 2. Save the slice to a temporary input file for RawGL
    temp_input_path = output_dir / f"input_{plane}_{slice_index}.png"
    imageio.imwrite(temp_input_path, slice_array)

    # 3. Prepare the RawGL command
    # Use a copy of the shader pass to avoid modifying the original dict
    pass_config = shader_pass.copy()
    pass_config['in'] = {'Texture0': str(temp_input_path)}

    # Define the final output path for this slice
    final_output_path = output_dir / f"output_{plane}_{slice_index}.png"
    pass_config['out'] = {'OutColor': str(final_output_path)}

    # Build command
    command = [rawgl_executable]
    for key, value in pass_config.items():
        arg_key = f"--{key}"
        if isinstance(value, dict):
            for sub_key, sub_val in value.items():
                command.extend([arg_key, sub_key, str(sub_val)])
        else:
            command.extend([arg_key, str(value)])

    # 4. Run the RawGL subprocess
    subprocess.run(command, capture_output=True, text=True, check=True)

    # 5. Return the path to the final output file
    return str(final_output_path)


class RawGLDaskWorker(QObject):
    """
    Worker object that uses a Dask client to run the processing pipeline.
    """
    finished = Signal(bool, str)
    progress_update = Signal(int, int) # current slice, total slices
    log_message = Signal(str)

    def __init__(self, dask_grid: DaskGrid, config: Dict[str, Any]):
        super().__init__()
        self.dask_grid = dask_grid
        self.config = config
        self.is_running = True

    def run(self):
        """
        Sets up a Dask client and executes the processing graph.
        """
        try:
            num_workers = self.config.get('num_workers', 1)
            plane = self.config.get('plane', 'XY')
            slice_range = self.config.get('slice_range', (0, self.dask_grid.shape[0]))
            shader_pass = self.config.get('shader_pass', {})
            rawgl_exec = self.config.get('rawgl_executable', 'rawgl')

            self.log_message.emit(f"Setting up Dask cluster with {num_workers} workers.")
            # Using a LocalCluster is safer for threads and subprocesses
            with LocalCluster(n_workers=num_workers, threads_per_worker=1) as cluster, Client(cluster) as client:
                self.log_message.emit(f"Dask dashboard available at: {client.dashboard_link}")

                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)

                    tasks = []
                    for i in range(slice_range[0], slice_range[1]):
                        if not self.is_running:
                            self.finished.emit(False, "Pipeline cancelled.")
                            return

                        # Create a delayed task for each slice
                        task = dask.delayed(process_slice_with_rawgl)(
                            self.dask_grid, plane, i, shader_pass, rawgl_exec, temp_path
                        )
                        tasks.append(task)

                    total_slices = len(tasks)
                    self.log_message.emit(f"Computing {total_slices} slices across plane {plane}...")

                    # Use dask.compute to execute the entire graph.
                    # This is simpler than as_completed for batch jobs.
                    # Progress can be inferred from the number of completed tasks.
                    self.log_message.emit(f"Computing {len(tasks)} tasks...")
                    results = dask.compute(*tasks)

                    # For now, we signal progress at the end.
                    # A more advanced version could use Dask's event callbacks.
                    for i, result_path in enumerate(results):
                         self.log_message.emit(f"Completed task. Output at: {result_path}")
                         self.progress_update.emit(i + 1, len(tasks))

        except Exception as e:
            error_message = f"An error occurred in the Dask pipeline: {e}"
            self.log_message.emit(error_message)
            self.finished.emit(False, error_message)
            return

        self.finished.emit(True, "Dask pipeline completed successfully.")

    def stop(self):
        self.is_running = False

class RawGLController(QObject):
    """
    Controller to manage the Dask-based RawGL processing thread.
    """
    finished = Signal(bool, str)
    progress_update = Signal(int, int)
    log_message = Signal(str)

    def __init__(self, dask_grid: DaskGrid, config: Dict[str, Any]):
        super().__init__()
        self._dask_grid = dask_grid
        self._config = config
        self._thread = None
        self._worker = None

    def run(self):
        if self._thread and self._thread.isRunning():
            return

        self._thread = QThread()
        self._worker = RawGLDaskWorker(self._dask_grid, self._config)
        self._worker.moveToThread(self._thread)

        self._worker.finished.connect(self.finished)
        self._worker.progress_update.connect(self.progress_update)
        self._worker.log_message.connect(self.log_message)

        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

        self._thread.started.connect(self._worker.run)
        self._thread.start()

    def stop(self):
        if self._worker:
            self._worker.stop()
