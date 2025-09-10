import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Any

from PySide6.QtCore import QObject, QThread, Signal

class RawGLWorker(QObject):
    """
    Worker object that runs the RawGL processing pipeline in a separate thread.
    """
    finished = Signal(bool, str)  # Emits success (bool) and final message (str)
    progress_update = Signal(int, int)  # Emits current step and total steps
    log_message = Signal(str)  # Emits a log message for each step

    def __init__(self, pipeline: List[Dict[str, Any]], rawgl_executable: str = "rawgl"):
        super().__init__()
        self.pipeline = pipeline
        self.rawgl_executable = rawgl_executable
        self.is_running = True

    def run(self):
        """
        Executes the entire RawGL pipeline.
        """
        total_steps = len(self.pipeline)
        self.log_message.emit(f"Starting RawGL pipeline with {total_steps} steps...")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # This mapping stores the output path of a pass's output uniform
            # so it can be used as an input in a subsequent pass.
            # Key: (pass_index, uniform_name), Value: file_path
            output_map = {}

            for i, step_config in enumerate(self.pipeline):
                if not self.is_running:
                    self.log_message.emit("Pipeline execution cancelled.")
                    self.finished.emit(False, "Pipeline cancelled by user.")
                    return

                self.progress_update.emit(i + 1, total_steps)
                self.log_message.emit(f"--- Step {i+1}/{total_steps} ---")

                try:
                    # Resolve inputs from previous steps
                    step_config = self._resolve_inputs(step_config, output_map)

                    # Prepare outputs for this step
                    step_config, output_paths = self._prepare_outputs(step_config, temp_path, i)

                    # Build and run the command
                    command = self._build_command(step_config)
                    self.log_message.emit(f"Executing command: {' '.join(command)}")

                    result = subprocess.run(command, capture_output=True, text=True, check=True)
                    self.log_message.emit(f"RawGL stdout:\n{result.stdout}")
                    if result.stderr:
                        self.log_message.emit(f"RawGL stderr:\n{result.stderr}")

                    # Update the output map for the next iteration
                    for uniform_name, path in output_paths.items():
                        output_map[(i, uniform_name)] = path

                except (subprocess.CalledProcessError, FileNotFoundError, Exception) as e:
                    error_message = f"Error at step {i+1}: {e}"
                    if isinstance(e, subprocess.CalledProcessError):
                        error_message += f"\nRawGL stderr:\n{e.stderr}"
                    self.log_message.emit(error_message)
                    self.finished.emit(False, error_message)
                    return

        self.log_message.emit("--- Pipeline finished successfully ---")
        self.finished.emit(True, "Pipeline completed successfully.")

    def _resolve_inputs(self, config: Dict, output_map: Dict) -> Dict:
        """Resolves input paths that reference outputs of previous passes."""
        if 'in' in config and isinstance(config['in'], dict):
            for uniform, value in config['in'].items():
                # Input format is (pass_index, uniform_name) tuple
                if isinstance(value, tuple) and len(value) == 2:
                    if value in output_map:
                        config['in'][uniform] = str(output_map[value])
                    else:
                        raise ValueError(f"Could not resolve input for '{uniform}': Output from pass {value[0]} ('{value[1]}') not found.")
        return config

    def _prepare_outputs(self, config: Dict, temp_path: Path, step_index: int) -> (Dict, Dict):
        """Prepares output paths, using temp files for intermediate steps."""
        output_paths = {}
        if 'out' in config and isinstance(config['out'], dict):
            for uniform, value in config['out'].items():
                if value == 'TEMP':
                    # Create a temporary path for this intermediate output
                    temp_file = temp_path / f"step_{step_index}_{uniform}.png"
                    config['out'][uniform] = str(temp_file)
                    output_paths[uniform] = temp_file
                else:
                    # This is a final, user-specified output
                    final_path = Path(value)
                    final_path.parent.mkdir(parents=True, exist_ok=True)
                    output_paths[uniform] = final_path
        return config, output_paths

    def _build_command(self, config: Dict) -> List[str]:
        """Constructs the command list from a step configuration dictionary."""
        command = [self.rawgl_executable]
        for key, value in config.items():
            if value is None: continue

            arg_key = f"--{key}"
            if isinstance(value, list):
                command.append(arg_key)
                command.extend(map(str, value))
            elif isinstance(value, dict):
                # For dicts like --in and --out
                for sub_key, sub_val in value.items():
                    command.append(arg_key)
                    command.append(sub_key)
                    command.append(str(sub_val))
            else:
                command.append(arg_key)
                command.append(str(value))
        return command

    def stop(self):
        self.is_running = False

class RawGLController(QObject):
    """
    Controller to manage the RawGL processing thread.
    """
    # Expose signals from the worker
    finished = Signal(bool, str)
    progress_update = Signal(int, int)
    log_message = Signal(str)

    def __init__(self, pipeline: List[Dict[str, Any]], rawgl_executable: str = "rawgl"):
        super().__init__()
        self._pipeline = pipeline
        self._rawgl_executable = rawgl_executable

        self._thread = None
        self._worker = None

    def run(self):
        """
        Starts the pipeline execution in a background thread.
        """
        if self._thread and self._thread.isRunning():
            print("Warning: Pipeline is already running.")
            return

        self._thread = QThread()
        self._worker = RawGLWorker(self._pipeline, self._rawgl_executable)
        self._worker.moveToThread(self._thread)

        # Connect signals
        self._worker.finished.connect(self.finished)
        self._worker.progress_update.connect(self.progress_update)
        self._worker.log_message.connect(self.log_message)

        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

        self._thread.started.connect(self._worker.run)
        self._thread.start()

    def stop(self):
        """Stops the currently running pipeline."""
        if self._worker:
            self._worker.stop()
