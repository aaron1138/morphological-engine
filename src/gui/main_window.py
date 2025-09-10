import sys
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

        # Import and add the pipeline panel
        from .rawgl_pipeline_panel import RawGLPipelinePanel
        self.pipeline_panel = RawGLPipelinePanel()
        right_layout.addWidget(self.pipeline_panel)

        # Add a "Run" button
        self.run_button = QPushButton("Run Processing Pipeline")
        self.run_button.setFixedHeight(40)
        self.run_button.clicked.connect(self._run_rawgl_pipeline)
        right_layout.addWidget(self.run_button)

        main_layout.addWidget(right_widget)

        self._create_menu_bar()
        self._create_status_bar()

        # To hold the controller instance while it's running
        self.rawgl_controller = None

        print("Main window scaffold created and RawGL panel integrated.")

    def _run_rawgl_pipeline(self):
        """Initiates the RawGL processing pipeline."""
        pipeline_def = self.pipeline_panel.get_pipeline()
        if not pipeline_def:
            print("Pipeline is empty. Nothing to run.")
            return

        # Assuming 'rawgl' is in the system PATH or in the same directory
        # In a real app, this path would be configurable.
        rawgl_executable = "rawgl"

        from src.processing.rawgl_controller import RawGLController
        self.rawgl_controller = RawGLController(pipeline_def, rawgl_executable)

        # Connect signals to handlers
        self.rawgl_controller.progress_update.connect(self._handle_pipeline_progress)
        self.rawgl_controller.log_message.connect(self._handle_pipeline_log)
        self.rawgl_controller.finished.connect(self._handle_pipeline_finished)

        self.rawgl_controller.run()
        # Disable the run button while processing
        self.sender().setEnabled(False)

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
        open_action = QAction("Open...", self)
        save_action = QAction("Save As...", self)
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(open_action)
        file_menu.addAction(save_action)
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
