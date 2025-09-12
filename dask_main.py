"""
Copyright (c) 2025 Jules

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import sys
from PySide6.QtWidgets import QApplication
from dask_ui import DaskProcessorApp

def main():
    """
    The main entry point for the application.
    """
    app = QApplication(sys.argv)
    window = DaskProcessorApp()
    window.resize(960, 900)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
