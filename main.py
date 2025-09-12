import sys
from PySide6.QtWidgets import QApplication

# Import the main application window from our GUI module
from gui.main_window import DaskEngineApp

def main():
    """
    The main entry point for the Dask-Based 3D Print Plane Extractor application.
    """
    # Create the Qt Application
    app = QApplication(sys.argv)

    # Create and show the main window
    window = DaskEngineApp()
    # Set a reasonable default size for the application window
    window.resize(650, 550)
    window.show()

    # Start the application's event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
