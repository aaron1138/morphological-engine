# -*- coding: utf-8 -*-
"""
Module: rawgl_controller.py
Author: Jules
Description: A PyQt6 widget for configuring and running RawGL processing pipelines.
"""

import subprocess
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QPushButton,
    QLineEdit, QFileDialog, QTextEdit, QMessageBox
)
from PyQt6.QtCore import QThread, pyqtSignal

class RawGLProcessingThread(QThread):
    """
    A QThread that runs the RawGL executable in a separate process.
    """
    progress = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, command: list):
        super().__init__()
        self.command = command

    def run(self):
        """
        Executes the RawGL command and streams its output.
        """
        try:
            self.progress.emit(f"Running command: {' '.join(self.command)}")
            process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, # Redirect stderr to stdout
                text=True,
                encoding='utf-8',
                errors='replace' # Handle potential encoding errors
            )

            # Read and emit output line by line
            while True:
                line = process.stdout.readline()
                if not line:
                    break
                self.progress.emit(line.strip())

            process.wait() # Wait for the process to complete

            if process.returncode != 0:
                self.error.emit(f"RawGL process exited with error code {process.returncode}")

        except FileNotFoundError:
            self.error.emit(f"Error: The executable '{self.command[0]}' was not found.")
        except Exception as e:
            self.error.emit(f"An unexpected error occurred: {e}")
        finally:
            self.finished.emit()


class RawGLController(QFrame):
    """
    A widget for controlling the RawGL external executable.
    """
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)

        main_layout = QVBoxLayout(self)

        title_label = QLabel("RawGL External Processor")
        title_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        main_layout.addWidget(title_label)

        # --- UI for selecting paths ---
        self.rawgl_path = self._create_path_selector("RawGL Executable:")
        main_layout.addLayout(self.rawgl_path['layout'])

        self.shader_path = self._create_path_selector("Shader File:")
        main_layout.addLayout(self.shader_path['layout'])

        self.input_path = self._create_path_selector("Input Image:")
        main_layout.addLayout(self.input_path['layout'])

        self.output_path = self._create_path_selector("Output Image:", save_file=True)
        main_layout.addLayout(self.output_path['layout'])

        # --- Run Button ---
        self.run_button = QPushButton("Run RawGL")
        self.run_button.setFixedHeight(30)
        self.run_button.setStyleSheet("font-size: 12pt;")
        main_layout.addWidget(self.run_button)

        # --- Log Output ---
        log_label = QLabel("RawGL Output:")
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setFontFamily("Courier")
        self.log_output.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        main_layout.addWidget(log_label)
        main_layout.addWidget(self.log_output)

        # Connect signals to slots
        self.rawgl_path['button'].clicked.connect(lambda: self._get_path(self.rawgl_path['line_edit']))
        self.shader_path['button'].clicked.connect(lambda: self._get_path(self.shader_path['line_edit']))
        self.input_path['button'].clicked.connect(lambda: self._get_path(self.input_path['line_edit']))
        self.output_path['button'].clicked.connect(lambda: self._get_path(self.output_path['line_edit'], save_file=True))

        self.run_button.clicked.connect(self.run_processing)
        self.processing_thread = None

    def _create_path_selector(self, label_text: str, save_file: bool = False):
        """Helper method to create a labeled line edit with a browse button."""
        layout = QHBoxLayout()
        label = QLabel(label_text)
        label.setFixedWidth(120)
        line_edit = QLineEdit()
        button = QPushButton("Browse...")

        layout.addWidget(label)
        layout.addWidget(line_edit)
        layout.addWidget(button)

        return {"layout": layout, "line_edit": line_edit, "button": button}

    def _get_path(self, line_edit: QLineEdit, save_file: bool = False):
        """Opens a file dialog and sets the path in the line edit."""
        if save_file:
            path, _ = QFileDialog.getSaveFileName(self, "Select Output File", "", "PNG Files (*.png)")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select File", "")

        if path:
            line_edit.setText(path)

    def run_processing(self):
        """
        Validates inputs, constructs the RawGL command, and starts the processing thread.
        """
        paths = {
            "rawgl": self.rawgl_path['line_edit'].text(),
            "shader": self.shader_path['line_edit'].text(),
            "input": self.input_path['line_edit'].text(),
            "output": self.output_path['line_edit'].text()
        }

        for name, path in paths.items():
            if not path:
                QMessageBox.warning(self, "Missing Input", f"Please provide the path for '{name}'.")
                return

        self.log_output.clear()
        self.run_button.setEnabled(False)
        self.log_output.append("--- Starting RawGL process... ---")

        # Assemble the command for 8-bit single-channel grayscale PNG
        command = [
            paths['rawgl'],
            '--pass_vertfrag', paths['shader'],
            '--in', 'Texture0', paths['input'],
            '--out', 'OutColor', paths['output'],
            '--out_format', 'r8',
            '--out_channels', '1',
            '--out_bits', '8'
        ]

        self.processing_thread = RawGLProcessingThread(command)
        self.processing_thread.progress.connect(self.append_log)
        self.processing_thread.error.connect(self.on_processing_error)
        self.processing_thread.finished.connect(self.on_processing_finished)
        self.processing_thread.start()

    def append_log(self, text: str):
        """Appends a line of text to the log output."""
        self.log_output.append(text)

    def on_processing_error(self, message: str):
        """Handles errors reported by the processing thread."""
        self.log_output.append(f"\nERROR: {message}\n")

    def on_processing_finished(self):
        """Called when the processing thread is finished."""
        self.log_output.append("\n--- RawGL process finished. ---")
        self.run_button.setEnabled(True)
        self.processing_thread = None
