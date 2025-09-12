import sys
from PySide6.QtWidgets import QApplication
from gui import DaskImageViewerApp

def main():
    """
    The main entry point for the application.
    """
    app = QApplication(sys.argv)
    window = DaskImageViewerApp()
    window.resize(600, 400)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
