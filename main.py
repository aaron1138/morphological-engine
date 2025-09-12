# main.py
import sys
from PySide6.QtWidgets import QApplication
from ui import ImageProcessorApp

def main():
    """
    The main entry point for the application.
    """
    app = QApplication(sys.argv)
    window = ImageProcessorApp()
    window.resize(960, 900) # Set a reasonable default size
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
