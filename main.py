import sys
from PySide6.QtWidgets import QApplication
from gui import DaskProcessorApp

def main():
    """
    The main entry point for the application.
    """
    app = QApplication(sys.argv)
    window = DaskProcessorApp()
    window.resize(700, 500)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
