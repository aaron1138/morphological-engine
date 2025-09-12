import sys
from PySide6.QtWidgets import QApplication
from dask_slice_engine.gui import DaskEngineApp

def main():
    """
    The main entry point for the application.
    """
    app = QApplication(sys.argv)
    window = DaskEngineApp()
    window.resize(640, 480)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
