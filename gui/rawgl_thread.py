# -*- coding: utf-8 -*-
"""
Module: rawgl_thread.py
Author: Jules
Description: A QThread subclass for running the RawGL pipeline in the background.
"""

from PyQt6.QtCore import QThread, pyqtSignal
from core.rawgl_controller import RawGLController

class RawGLThread(QThread):
    """
    Runs a sequence of RawGL passes in a background thread.
    """
    # Signals to communicate with the main thread
    progress_update = pyqtSignal(int, int)  # current_step, total_steps
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, rawgl_exec_path, pipeline_configs):
        """
        Args:
            rawgl_exec_path (str): The file path to the rawgl executable.
            pipeline_configs (list): A list of configuration dictionaries for each pass.
        """
        super().__init__()
        self.rawgl_exec_path = rawgl_exec_path
        self.pipeline_configs = pipeline_configs

    def run(self):
        """
        The main entry point for the thread. Executes the RawGL pipeline.
        """
        try:
            total_steps = len(self.pipeline_configs)
            controller = RawGLController(self.rawgl_exec_path)

            for i, config in enumerate(self.pipeline_configs):
                self.progress_update.emit(i + 1, total_steps)

                # A mechanism to link passes, e.g., output of pass 0 is input of pass 1
                # For now, we assume paths are absolute and pre-configured.
                # A future improvement would be to handle temporary files for chaining.

                success, log = controller.run_pass(config)

                if not success:
                    error_message = f"Failed on pass {i + 1}.\nLog:\n{log}"
                    self.error.emit(error_message)
                    return # Stop processing on failure

            self.finished.emit()

        except Exception as e:
            self.error.emit(f"An unexpected error occurred in the processing thread: {e}")
