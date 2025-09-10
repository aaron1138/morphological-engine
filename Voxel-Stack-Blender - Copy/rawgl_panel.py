import os
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLineEdit,
    QFileDialog,
    QTableWidget,
    QTableWidgetItem,
    QPlainTextEdit,
    QHeaderView,
    QLabel,
    QFormLayout
)
from PySide6.QtCore import Slot, Qt
from rawgl_controller import RawGLController, RawGLTask

class RawGLPanel(QWidget):
    """
    A UI panel for controlling and monitoring RawGL processing tasks.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("RawGL Processing Pipeline")

        # --- Backend Controller ---
        self.controller = RawGLController()

        # --- UI Widgets ---
        # File selection
        self.input_files_edit = QLineEdit()
        self.input_files_edit.setPlaceholderText("Select one or more image files")
        self.btn_select_inputs = QPushButton("Select Inputs...")

        self.shader_file_edit = QLineEdit()
        self.shader_file_edit.setPlaceholderText("Select a .glsl or .frag shader file")
        self.btn_select_shader = QPushButton("Select Shader...")

        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Select a directory to save output files")
        self.btn_select_output = QPushButton("Select Output Dir...")

        # Action button
        self.btn_start_processing = QPushButton("Start Processing")

        # Progress table
        self.task_table = QTableWidget()
        self.task_table.setColumnCount(3)
        self.task_table.setHorizontalHeaderLabels(["Input File", "Status", "Output File"])
        self.task_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.task_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)

        # Log window
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumBlockCount(1000)

        # --- Layout ---
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        form_layout.addRow("Input Image(s):", self._create_file_selection_layout(self.input_files_edit, self.btn_select_inputs))
        form_layout.addRow("Shader File:", self._create_file_selection_layout(self.shader_file_edit, self.btn_select_shader))
        form_layout.addRow("Output Directory:", self._create_file_selection_layout(self.output_dir_edit, self.btn_select_output))

        layout.addLayout(form_layout)
        layout.addWidget(self.btn_start_processing)
        layout.addWidget(QLabel("Processing Tasks:"))
        layout.addWidget(self.task_table)
        layout.addWidget(QLabel("Logs:"))
        layout.addWidget(self.log_edit)

        # --- Connections ---
        self.btn_select_inputs.clicked.connect(self._select_input_files)
        self.btn_select_shader.clicked.connect(self._select_shader_file)
        self.btn_select_output.clicked.connect(self._select_output_dir)
        self.btn_start_processing.clicked.connect(self._start_processing)

        self.controller.task_added.connect(self._add_task_to_table)
        self.controller.task_status_changed.connect(self._update_task_status)
        self.controller.task_log_message.connect(self._append_log_message)

        self.task_rows = {} # To map task_id to table row index

    def _create_file_selection_layout(self, line_edit, button):
        layout = QHBoxLayout()
        layout.addWidget(line_edit)
        layout.addWidget(button)
        return layout

    @Slot()
    def _select_input_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Input Images", "", "Image Files (*.png *.jpg *.bmp *.tif)")
        if files:
            self.input_files_edit.setText(";".join(files))

    @Slot()
    def _select_shader_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Shader File", "", "Shader Files (*.glsl *.frag *.vert)")
        if file:
            self.shader_file_edit.setText(file)

    @Slot()
    def _select_output_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_dir_edit.setText(directory)

    @Slot()
    def _start_processing(self):
        input_files = self.input_files_edit.text().split(';')
        shader_file = self.shader_file_edit.text()
        output_dir = self.output_dir_edit.text()

        if not all([input_files, shader_file, output_dir]):
            self.log_edit.appendPlainText("ERROR: Please select input files, a shader, and an output directory.")
            return

        self.task_table.setRowCount(0)
        self.log_edit.clear()

        tasks = []
        for input_path in input_files:
            if not input_path: continue
            base_name = os.path.basename(input_path)
            output_path = os.path.join(output_dir, base_name)
            tasks.append(RawGLTask(input_path=input_path, output_path=output_path, shader_path=shader_file))

        self.controller.submit_tasks(tasks)

    @Slot(object)
    def _add_task_to_table(self, task: RawGLTask):
        row_position = self.task_table.rowCount()
        self.task_table.insertRow(row_position)

        self.task_table.setItem(row_position, 0, QTableWidgetItem(os.path.basename(task.input_path)))
        self.task_table.setItem(row_position, 1, QTableWidgetItem(task.status))
        self.task_table.setItem(row_position, 2, QTableWidgetItem(os.path.basename(task.output_path)))

        self.task_rows[task.task_id] = row_position

    @Slot(str, str)
    def _update_task_status(self, task_id, status):
        if task_id in self.task_rows:
            row = self.task_rows[task_id]
            self.task_table.item(row, 1).setText(status)

    @Slot(str, str)
    def _append_log_message(self, task_id, message):
        # Prepend with the task's filename for clarity
        task = self.controller.get_task(task_id)
        if task:
            log_line = f"[{os.path.basename(task.input_path)}] {message}"
            self.log_edit.appendPlainText(log_line)
        else:
            self.log_edit.appendPlainText(message)
