import sys
import os
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QHBoxLayout, QVBoxLayout, QLabel, QFrame,
    QMenuBar, QPushButton
)
from PySide6.QtGui import QAction
from PySide6.QtCore import Qt

class MainWindow(QMainWindow):
    """
    The main window for the Voxel Slice Pre-Processor application.
    It sets up the main UI layout, including placeholders for the file list,
    3D viewport, and settings panels, based on common 3D editor layouts.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voxel Slice Pre-Processor")
        self.setGeometry(100, 100, 1600, 900)

        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # --- Left Panel (File List / Project Explorer) ---
        left_panel = QFrame()
        left_panel.setFrameShape(QFrame.Shape.StyledPanel)
        left_panel.setMinimumWidth(250)
        left_panel.setMaximumWidth(400)
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(QLabel("<b>Project Explorer</b>"))
        # In a real app, this would be a QTreeView or QListWidget
        left_layout.addStretch()
        main_layout.addWidget(left_panel)

        # --- Center Panel (3D Viewport) ---
        center_panel = QFrame()
        center_panel.setFrameShape(QFrame.Shape.StyledPanel)
        center_layout = QVBoxLayout(center_panel)
        viewport_label = QLabel("3D Viewport")
        viewport_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center_layout.addWidget(viewport_label)
        # This is where the ModernGL/PySide6 integration widget would go
        main_layout.addWidget(center_panel, stretch=1)

        # --- Right Panel (RawGL Pipeline) ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_widget.setMinimumWidth(500)

        # Import and add the settings panel
        from .rawgl_pipeline_panel import RawGLSettingsPanel
        from src.core.dask_grid import DaskGrid
        from PySide6.QtWidgets import QSpinBox, QFileDialog, QHBoxLayout, QLabel

        self.settings_panel = RawGLSettingsPanel()
        right_layout.addWidget(self.settings_panel)

        # Add Thread Count and Run Button
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel("Dask Workers (Threads):"))
        self.thread_count_spin = QSpinBox()
        self.thread_count_spin.setRange(1, os.cpu_count() or 1)
        self.thread_count_spin.setValue(os.cpu_count() or 1)
        controls_layout.addWidget(self.thread_count_spin)
        controls_layout.addStretch()
        self.run_button = QPushButton("Run Processing")
        self.run_button.setFixedHeight(40)
        self.run_button.clicked.connect(self._run_dask_pipeline)
        controls_layout.addWidget(self.run_button)
        right_layout.addLayout(controls_layout)

        main_layout.addWidget(right_widget)

        self._create_menu_bar()
        self._create_status_bar()

        # To hold the controller and grid instances
        self.dask_grid = None
        self.dask_controller = None

        print("Main window scaffold created and RawGL panel integrated.")

    def _load_image_stack(self):
        """Opens a dialog to select a directory of images and loads it into a DaskGrid."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Image Stack Directory", ".")
        if not dir_path:
            return

        try:
            self.statusBar().showMessage(f"Loading image stack from: {dir_path}...")
            self.dask_grid = DaskGrid(dir_path)
            self.statusBar().showMessage(f"Loaded grid with shape: {self.dask_grid.shape}")
            # Update slice range spinners
            self.settings_panel.end_slice_spin.setValue(self.dask_grid.shape[0])
        except (NotADirectoryError, FileNotFoundError) as e:
            self.statusBar().showMessage(f"Error: {e}", 5000)
            self.dask_grid = None

    def _run_dask_pipeline(self):
        """Initiates the Dask-based RawGL processing pipeline."""
        if self.dask_grid is None:
            self.statusBar().showMessage("Please load an image stack first.", 5000)
            return

        job_config = self.settings_panel.get_processing_config()
        job_config['num_workers'] = self.thread_count_spin.value()
        job_config['rawgl_executable'] = 'rawgl' # Should be configurable in a real app

        from src.processing.rawgl_controller import RawGLController
        self.dask_controller = RawGLController(self.dask_grid, job_config)

        # Connect signals to handlers
        self.dask_controller.progress_update.connect(self._handle_pipeline_progress)
        self.dask_controller.log_message.connect(self._handle_pipeline_log)
        self.dask_controller.finished.connect(self._handle_pipeline_finished)

        self.dask_controller.run()
        # Disable the run button while processing
        self.run_button.setEnabled(False)

    def _handle_pipeline_progress(self, step, total):
        """Updates the status bar with the current progress."""
        self.statusBar().showMessage(f"Processing step {step} of {total}...")

    def _handle_pipeline_log(self, message):
        """Prints log messages from the controller."""
        # In a real app, this would go to a logging widget.
        print(message)

    def _handle_pipeline_finished(self, success, message):
        """Handles the completion of the pipeline."""
        print(f"Pipeline finished. Success: {success}. Message: {message}")
        self.statusBar().showMessage(message, 10000) # Show message for 10 seconds

        # Re-enable the run button
        self.run_button.setEnabled(True)

        self.rawgl_controller = None # Release the controller

    def _create_menu_bar(self):
        menu_bar = self.menuBar()

        # File Menu
        file_menu = menu_bar.addMenu("&File")
        open_action = QAction("Load Image Stack...", self)
        open_action.triggered.connect(self._load_image_stack)
        file_menu.addAction(open_action)
        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

        # Edit Menu
        edit_menu = menu_bar.addMenu("&Edit")
        undo_action = QAction("Undo", self)
        redo_action = QAction("Redo", self)
        edit_menu.addAction(undo_action)
        edit_menu.addAction(redo_action)

        # View Menu
        view_menu = menu_bar.addMenu("&View")

        # Help Menu
        help_menu = menu_bar.addMenu("&Help")
        about_action = QAction("About", self)
        help_menu.addAction(about_action)

    def _create_status_bar(self):
        self.statusBar().showMessage("Ready", 3000)

def launch_gui():
    """Entry point function to launch the application's GUI."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    launch_gui()
