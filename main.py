# main.py

import sys
from PySide6.QtWidgets import QApplication
from ui import ModularApp

def main():
    """
    The main entry point for the application.
    """
    app = QApplication(sys.argv)
    window = ModularApp()
    window.resize(600, 400)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
