import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QProgressBar, QFileDialog, QMessageBox, QCheckBox,
    QTabWidget, QGroupBox, QRadioButton, QButtonGroup, QStackedWidget,
    QGridLayout, QFrame
)
from PySide6.QtCore import Slot, QSettings
from PySide6.QtGui import QIntValidator

class DaskImageViewerApp(QWidget):
    """The main application window for the Dask Image Viewer."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dask Image Viewer")
        self.settings = QSettings("MyCompany", "DaskImageViewer")
        self.init_ui()
        self._autodetect_uvtools()
        self.load_settings()
        self._connect_signals()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        self.main_tab = QWidget()
        main_tab_layout = QVBoxLayout(self.main_tab)
        self.tab_widget.addTab(self.main_tab, "Main")

        # --- I/O Section ---
        io_group = QGroupBox("I/O")
        io_layout = QVBoxLayout(io_group)

        input_mode_layout = QHBoxLayout()
        self.input_mode_group = QButtonGroup(self)
        self.folder_mode_radio = QRadioButton("Folder Input Mode")
        self.folder_mode_radio.setChecked(True)
        self.uvtools_mode_radio = QRadioButton("Use UVTools")
        self.input_mode_group.addButton(self.folder_mode_radio, 0)
        self.input_mode_group.addButton(self.uvtools_mode_radio, 1)
        input_mode_layout.addWidget(self.folder_mode_radio)
        input_mode_layout.addWidget(self.uvtools_mode_radio)
        input_mode_layout.addStretch(1)
        io_layout.addLayout(input_mode_layout)

        self.io_stacked_widget = QStackedWidget()
        io_layout.addWidget(self.io_stacked_widget)

        # --- Folder Mode Widget ---
        folder_mode_widget = QWidget()
        folder_mode_layout = QGridLayout(folder_mode_widget)
        folder_mode_layout.addWidget(QLabel("Input Folder:"), 0, 0)
        self.input_folder_edit = QLineEdit()
        folder_mode_layout.addWidget(self.input_folder_edit, 0, 1)
        self.input_folder_button = QPushButton("Browse...")
        folder_mode_layout.addWidget(self.input_folder_button, 0, 2)

        folder_mode_layout.addWidget(QLabel("Output Folder:"), 1, 0)
        self.output_folder_edit = QLineEdit()
        folder_mode_layout.addWidget(self.output_folder_edit, 1, 1)
        self.output_folder_button = QPushButton("Browse...")
        folder_mode_layout.addWidget(self.output_folder_button, 1, 2)
        self.io_stacked_widget.addWidget(folder_mode_widget)

        # --- UVTools Mode Widget ---
        uvtools_mode_widget = QWidget()
        uvtools_mode_layout = QGridLayout(uvtools_mode_widget)
        uvtools_mode_layout.addWidget(QLabel("Path to UVToolsCmd.exe:"), 0, 0)
        self.uvtools_path_edit = QLineEdit()
        uvtools_mode_layout.addWidget(self.uvtools_path_edit, 0, 1)
        self.uvtools_path_button = QPushButton("Browse...")
        uvtools_mode_layout.addWidget(self.uvtools_path_button, 0, 2)

        uvtools_mode_layout.addWidget(QLabel("Working Temp Folder:"), 1, 0)
        self.uvtools_temp_folder_edit = QLineEdit()
        uvtools_mode_layout.addWidget(self.uvtools_temp_folder_edit, 1, 1)
        self.uvtools_temp_folder_button = QPushButton("Browse...")
        uvtools_mode_layout.addWidget(self.uvtools_temp_folder_button, 1, 2)

        uvtools_mode_layout.addWidget(QLabel("Input Slice File:"), 2, 0)
        self.uvtools_input_file_edit = QLineEdit()
        uvtools_mode_layout.addWidget(self.uvtools_input_file_edit, 2, 1)
        self.uvtools_input_file_button = QPushButton("Browse...")
        uvtools_mode_layout.addWidget(self.uvtools_input_file_button, 2, 2)

        self.io_stacked_widget.addWidget(uvtools_mode_widget)
        main_tab_layout.addWidget(io_group)

        # --- General Settings Section ---
        general_group = QGroupBox("General")
        general_layout = QGridLayout(general_group)

        general_layout.addWidget(QLabel("Dask Thread Count:"), 0, 0)
        self.thread_count_edit = QLineEdit(str(os.cpu_count()))
        self.thread_count_edit.setValidator(QIntValidator(1, 128, self))
        general_layout.addWidget(self.thread_count_edit, 0, 1)

        main_tab_layout.addWidget(general_group)
        main_tab_layout.addStretch(1)

        # --- Processing Controls ---
        self.start_button = QPushButton("Start Processing")
        self.start_button.setMinimumHeight(40)
        main_layout.addWidget(self.start_button)
        self.progress_bar = QProgressBar()
        main_layout.addWidget(self.progress_bar)
        self.status_label = QLabel("Status: Ready")
        main_layout.addWidget(self.status_label)

    def _connect_signals(self):
        self.input_folder_button.clicked.connect(lambda: self.browse_folder(self.input_folder_edit))
        self.output_folder_button.clicked.connect(lambda: self.browse_folder(self.output_folder_edit))
        self.uvtools_path_button.clicked.connect(lambda: self.browse_file(self.uvtools_path_edit, "Select UVToolsCmd.exe", "Executable Files (*.exe)"))
        self.uvtools_temp_folder_button.clicked.connect(lambda: self.browse_folder(self.uvtools_temp_folder_edit))
        self.uvtools_input_file_button.clicked.connect(lambda: self.browse_file(self.uvtools_input_file_edit, "Select Input Slice File"))
        self.input_mode_group.idClicked.connect(self.on_input_mode_changed)
        self.start_button.clicked.connect(self.start_processing)

    def on_input_mode_changed(self, stack_index):
        self.io_stacked_widget.setCurrentIndex(stack_index)

    def browse_folder(self, line_edit):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", line_edit.text())
        if folder:
            line_edit.setText(folder)

    def browse_file(self, line_edit, caption, file_filter="All Files (*)"):
        file, _ = QFileDialog.getOpenFileName(self, caption, line_edit.text(), file_filter)
        if file:
            line_edit.setText(file)

    def _autodetect_uvtools(self):
        """Checks for UVTools in the default location and populates the path if found."""
        default_path = "C:\\Program Files\\UVTools\\UVToolsCmd.exe"
        if os.path.exists(default_path):
            if not self.uvtools_path_edit.text():
                self.uvtools_path_edit.setText(default_path)

    def load_settings(self):
        """Loads settings from QSettings."""
        self.resize(self.settings.value("window_size", self.size()))
        self.move(self.settings.value("window_position", self.pos()))
        self.input_folder_edit.setText(self.settings.value("input_folder", ""))
        self.output_folder_edit.setText(self.settings.value("output_folder", ""))
        self.uvtools_path_edit.setText(self.settings.value("uvtools_path", ""))
        self.uvtools_temp_folder_edit.setText(self.settings.value("uvtools_temp_folder", ""))
        self.uvtools_input_file_edit.setText(self.settings.value("uvtools_input_file", ""))
        self.thread_count_edit.setText(self.settings.value("thread_count", str(os.cpu_count())))

        input_mode = self.settings.value("input_mode", "folder")
        if input_mode == "uvtools":
            self.uvtools_mode_radio.setChecked(True)
            self.io_stacked_widget.setCurrentIndex(1)
        else:
            self.folder_mode_radio.setChecked(True)
            self.io_stacked_widget.setCurrentIndex(0)

    def save_settings(self):
        """Saves current UI settings to QSettings."""
        self.settings.setValue("window_size", self.size())
        self.settings.setValue("window_position", self.pos())
        self.settings.setValue("input_folder", self.input_folder_edit.text())
        self.settings.setValue("output_folder", self.output_folder_edit.text())
        self.settings.setValue("uvtools_path", self.uvtools_path_edit.text())
        self.settings.setValue("uvtools_temp_folder", self.uvtools_temp_folder_edit.text())
        self.settings.setValue("uvtools_input_file", self.uvtools_input_file_edit.text())
        self.settings.setValue("thread_count", self.thread_count_edit.text())

        if self.folder_mode_radio.isChecked():
            self.settings.setValue("input_mode", "folder")
        else:
            self.settings.setValue("input_mode", "uvtools")

    def closeEvent(self, event):
        self.save_settings()
        event.accept()

    def start_processing(self):
        self.save_settings()

        input_folder = ""

        if self.uvtools_mode_radio.isChecked():
            uvtools_path = self.uvtools_path_edit.text()
            input_file = self.uvtools_input_file_edit.text()
            temp_folder = self.uvtools_temp_folder_edit.text()

            if not all([uvtools_path, input_file, temp_folder]):
                self.show_error_message("Input Error", "Please fill in all UVTools fields.")
                return

            try:
                from uvtools_wrapper import extract_layers
                self.status_label.setText("Status: Extracting layers with UVTools...")
                input_folder = extract_layers(uvtools_path, input_file, temp_folder)
                self.status_label.setText(f"Status: Layers extracted to {input_folder}")
            except Exception as e:
                self.show_error_message("UVTools Error", str(e))
                self.status_label.setText("Status: Error during UVTools extraction.")
                return

        else: # Folder mode
            input_folder = self.input_folder_edit.text()
            output_folder = self.output_folder_edit.text()

            if not all([input_folder, output_folder]):
                self.show_error_message("Input Error", "Please fill in all folder fields.")
                return

        try:
            from dask_engine import DaskEngine
            thread_count = int(self.thread_count_edit.text())
            engine = DaskEngine(input_folder, thread_count)

            self.status_label.setText("Status: Loading images with Dask...")
            dask_array = engine.load_images()
            self.status_label.setText(f"Status: Dask array created with shape {dask_array.shape}")

        except Exception as e:
            self.show_error_message("Dask Engine Error", str(e))
            self.status_label.setText("Status: Error creating Dask array.")

    def show_error_message(self, title, text):
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle(title)
        msg_box.setText(text)
        msg_box.exec()
