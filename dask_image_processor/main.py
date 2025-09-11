# main.py - Dask Image Processor

import sys
from PySide6.QtWidgets import QApplication
from ui.app import DaskProcessorApp

def main():
    """
    The main entry point for the application.
    """
    app = QApplication(sys.argv)
    window = DaskProcessorApp()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
