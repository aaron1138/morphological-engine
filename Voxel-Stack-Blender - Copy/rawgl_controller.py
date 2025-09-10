import subprocess
import sys
import os
from dataclasses import dataclass, field
from PySide6.QtCore import QObject, Signal, Slot, QRunnable, QThreadPool
import uuid

@dataclass
class RawGLTask:
    """
    A dataclass to hold information for a single RawGL processing task.
    """
    input_path: str
    output_path: str
    shader_path: str
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: str = "Pending"
    log: str = ""

class RawGLWorkerSignals(QObject):
    """
    Defines the signals available from a running RawGLWorker.
    """
    finished = Signal(str)  # task_id
    error = Signal(str, str)  # task_id, error_message
    log_message = Signal(str, str) # task_id, message
    status_changed = Signal(str, str) # task_id, status

class RawGLWorker(QRunnable):
    """
    Worker thread for running a single RawGL command.
    Inherits from QRunnable to be used with QThreadPool.
    """
    def __init__(self, task: RawGLTask, rawgl_executable="rawgl"):
        super().__init__()
        self.task = task
        self.signals = RawGLWorkerSignals()
        self.rawgl_executable = rawgl_executable

    @Slot()
    def run(self):
        """
        Execute the RawGL task.
        """
        self.signals.status_changed.emit(self.task.task_id, "Processing")

        # Base command for single-channel 8-bit grayscale PNG output
        command = [
            self.rawgl_executable,
            '--pass_vertfrag', self.task.shader_path,
            '--pass_size', 'Texture0::0', 'Texture0::1', # Use input texture size
            '--in', 'Texture0', self.task.input_path,
            '--out', 'OutColor', self.task.output_path,
            '--out_format', 'r8',     # 8-bit single channel
            '--out_channels', '1',    # 1 channel
            '--out_bits', '8'         # 8 bits
        ]

        self.signals.log_message.emit(self.task.task_id, f"Executing command: {' '.join(command)}")

        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace'
            )

            # Read output line by line
            while True:
                line = process.stdout.readline()
                if not line:
                    break
                self.signals.log_message.emit(self.task.task_id, line.strip())

            process.wait()

            if process.returncode == 0:
                self.signals.status_changed.emit(self.task.task_id, "Completed")
                self.signals.finished.emit(self.task.task_id)
            else:
                error_msg = f"RawGL process exited with error code {process.returncode}"
                self.signals.status_changed.emit(self.task.task_id, "Failed")
                self.signals.error.emit(self.task.task_id, error_msg)

        except FileNotFoundError:
            error_msg = f"Error: '{self.rawgl_executable}' not found. Please ensure it is in your system's PATH."
            self.signals.status_changed.emit(self.task.task_id, "Failed")
            self.signals.error.emit(self.task.task_id, error_msg)
        except Exception as e:
            self.signals.status_changed.emit(self.task.task_id, "Failed")
            self.signals.error.emit(self.task.task_id, str(e))


class RawGLController(QObject):
    """
    Manages a thread pool for running multiple RawGL processing tasks concurrently.
    """
    # Signals to communicate with the UI
    task_added = Signal(object) # RawGLTask object
    task_status_changed = Signal(str, str) # task_id, status
    task_log_message = Signal(str, str) # task_id, message
    all_tasks_finished = Signal()

    def __init__(self, max_threads=None):
        super().__init__()
        if max_threads is None:
            max_threads = QThreadPool.globalInstance().maxThreadCount()
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(max_threads)
        self.active_tasks = 0
        self.tasks = {}

    def submit_tasks(self, tasks_to_run: list[RawGLTask]):
        """
        Submits a list of RawGLTasks to the thread pool for processing.
        """
        if not tasks_to_run:
            return

        self.active_tasks = len(tasks_to_run)

        for task_data in tasks_to_run:
            self.tasks[task_data.task_id] = task_data
            self.task_added.emit(task_data)

            worker = RawGLWorker(task=task_data)

            # Connect worker signals to controller signals/slots
            worker.signals.finished.connect(self._handle_task_finished)
            worker.signals.error.connect(self._handle_task_error)
            worker.signals.status_changed.connect(self.task_status_changed)
            worker.signals.log_message.connect(self.task_log_message)

            self.thread_pool.start(worker)

    @Slot(str, str)
    def _handle_task_error(self, task_id, message):
        self.tasks[task_id].status = "Failed"
        self.tasks[task_id].log += message + "\n"
        self.task_log_message.emit(task_id, f"ERROR: {message}")
        self._handle_task_finished(task_id)


    @Slot(str)
    def _handle_task_finished(self, task_id):
        self.active_tasks -= 1
        if self.active_tasks == 0:
            self.all_tasks_finished.emit()

    def get_task(self, task_id):
        return self.tasks.get(task_id)
