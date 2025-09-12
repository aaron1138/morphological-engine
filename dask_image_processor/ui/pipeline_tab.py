# ui/pipeline_tab.py

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QGroupBox, QStackedWidget, QLabel, QAbstractItemView,
    QHeaderView
)
from PySide6.QtCore import Slot

from config import app_config, PipelineOperation
from ui.add_operation_dialog import AddOperationDialog
from ui.lut_editor import LutEditorWidget

class PipelineTab(QWidget):
    """A widget for managing the processing pipeline."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self._connect_signals()
        self.load_pipeline()

    def init_ui(self):
        main_layout = QHBoxLayout(self)

        # --- Left Panel: Pipeline List ---
        left_panel = QGroupBox("Processing Pipeline")
        left_layout = QVBoxLayout(left_panel)

        self.pipeline_table = QTableWidget(0, 2)
        self.pipeline_table.setHorizontalHeaderLabels(["Operation", "Plane"])
        self.pipeline_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.pipeline_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.pipeline_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        left_layout.addWidget(self.pipeline_table)

        button_layout = QHBoxLayout()
        self.add_button = QPushButton("Add...")
        self.remove_button = QPushButton("Remove")
        self.move_up_button = QPushButton("Move Up")
        self.move_down_button = QPushButton("Move Down")
        button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.remove_button)
        button_layout.addStretch()
        button_layout.addWidget(self.move_up_button)
        button_layout.addWidget(self.move_down_button)
        left_layout.addLayout(button_layout)

        main_layout.addWidget(left_panel, 1)

        # --- Right Panel: Operation Details ---
        details_panel = QGroupBox("Operation Details")
        details_layout = QVBoxLayout(details_panel)
        self.details_stack = QStackedWidget()

        # Placeholder widgets for different operations
        self.placeholder_widget = QWidget()
        self.placeholder_widget.setLayout(QVBoxLayout())
        self.placeholder_widget.layout().addWidget(QLabel("Select an operation to see details."))

        self.edt_params_widget = self._create_edt_params_widget()
        self.shader_params_widget = self._create_shader_params_widget()
        self.blur_params_widget = self._create_blur_params_widget()
        self.blend_params_widget = self._create_blend_params_widget()

        self.details_stack.addWidget(self.placeholder_widget)
        self.details_stack.addWidget(self.edt_params_widget)
        self.details_stack.addWidget(self.shader_params_widget)
        self.details_stack.addWidget(self.blur_params_widget)
        self.details_stack.addWidget(self.blend_params_widget)

        self.lut_editor_widget = LutEditorWidget()
        self.details_stack.addWidget(self.lut_editor_widget)

        details_layout.addWidget(self.details_stack)
        main_layout.addWidget(details_panel, 2)

    def _create_edt_params_widget(self):
        """Creates the parameter editor for the Enhanced EDT operation."""
        widget = QWidget()
        layout = QGridLayout(widget)
        layout.addWidget(QLabel("Look Forward:"), 0, 0)
        self.edt_look_forward_edit = QLineEdit()
        layout.addWidget(self.edt_look_forward_edit, 0, 1)
        layout.addWidget(QLabel("Look Backward:"), 1, 0)
        self.edt_look_backward_edit = QLineEdit()
        layout.addWidget(self.edt_look_backward_edit, 1, 1)
        layout.addWidget(QLabel("Fade Distance Limit:"), 2, 0)
        self.edt_fade_dist_edit = QLineEdit()
        layout.addWidget(self.edt_fade_dist_edit, 2, 1)
        layout.setRowStretch(3, 1)

        self.edt_look_forward_edit.editingFinished.connect(self._update_current_op_params)
        self.edt_look_backward_edit.editingFinished.connect(self._update_current_op_params)
        self.edt_fade_dist_edit.editingFinished.connect(self._update_current_op_params)

        return widget

    def _create_shader_params_widget(self):
        """Creates the parameter editor for the GPU Shader operation."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.addWidget(QLabel("Shader File:"))
        self.shader_file_edit = QLineEdit()
        self.shader_file_edit.setReadOnly(True)
        layout.addWidget(self.shader_file_edit)
        self.browse_shader_button = QPushButton("Browse...")
        layout.addWidget(self.browse_shader_button)

        self.browse_shader_button.clicked.connect(self._browse_for_shader)
        return widget

    def _create_blur_params_widget(self):
        """Creates the parameter editor for the Gaussian Blur operation."""
        widget = QWidget()
        layout = QGridLayout(widget)
        layout.addWidget(QLabel("Kernel Size:"), 0, 0)
        self.blur_ksize_edit = QLineEdit()
        self.blur_ksize_edit.editingFinished.connect(self._update_current_op_params)
        layout.addWidget(self.blur_ksize_edit, 0, 1)
        layout.setRowStretch(1, 1)
        return widget

    def _create_blend_params_widget(self):
        """Creates the parameter editor for blend modes."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(QLabel("No parameters for this blend mode."))
        layout.addStretch(1)
        return widget

    def _connect_signals(self):
        self.add_button.clicked.connect(self.add_operation)
        self.remove_button.clicked.connect(self.remove_operation)
        self.move_up_button.clicked.connect(self.move_up)
        self.move_down_button.clicked.connect(self.move_down)
        self.pipeline_table.itemSelectionChanged.connect(self.update_details_panel)
        self.lut_editor_widget.lut_params_changed.connect(self._update_current_op_params)

    def load_pipeline(self):
        """Populates the table with operations from the global config."""
        self.pipeline_table.setRowCount(0)
        for op in app_config.pipeline:
            self._add_row_to_table(op)
        self.update_details_panel()

    def _add_row_to_table(self, op: PipelineOperation):
        row_position = self.pipeline_table.rowCount()
        self.pipeline_table.insertRow(row_position)
        self.pipeline_table.setItem(row_position, 0, QTableWidgetItem(op.type))
        self.pipeline_table.setItem(row_position, 1, QTableWidgetItem(op.plane))

    @Slot()
    def add_operation(self):
        dialog = AddOperationDialog(self)
        if dialog.exec():
            op_type, op_plane = dialog.get_selection()
            new_op = PipelineOperation(type=op_type, plane=op_plane)
            app_config.pipeline.append(new_op)
            self._add_row_to_table(new_op)
            self.pipeline_table.selectRow(self.pipeline_table.rowCount() - 1)

    @Slot()
    def remove_operation(self):
        current_row = self.pipeline_table.currentRow()
        if current_row >= 0:
            self.pipeline_table.removeRow(current_row)
            del app_config.pipeline[current_row]

    @Slot()
    def move_up(self):
        self._move_operation(-1)

    @Slot()
    def move_down(self):
        self._move_operation(1)

    def _move_operation(self, direction):
        current_row = self.pipeline_table.currentRow()
        if current_row < 0: return

        new_row = current_row + direction
        if 0 <= new_row < self.pipeline_table.rowCount():
            # Swap items in the config list
            p = app_config.pipeline
            p[current_row], p[new_row] = p[new_row], p[current_row]

            # Visually swap rows in the table
            for col in range(self.pipeline_table.columnCount()):
                item1 = self.pipeline_table.takeItem(current_row, col)
                item2 = self.pipeline_table.takeItem(new_row, col)
                self.pipeline_table.setItem(current_row, col, item2)
                self.pipeline_table.setItem(new_row, col, item1)

            self.pipeline_table.selectRow(new_row)

    @Slot()
    def update_details_panel(self):
        """Shows the correct parameter editor and populates it with data."""
        current_row = self.pipeline_table.currentRow()
        if current_row < 0 or current_row >= len(app_config.pipeline):
            self.details_stack.setCurrentWidget(self.placeholder_widget)
            return

        op = app_config.pipeline[current_row]

        # Block signals to prevent feedback loops while populating widgets
        all_param_widgets = [
            self.edt_look_forward_edit, self.edt_look_backward_edit, self.edt_fade_dist_edit,
            self.shader_file_edit, self.blur_ksize_edit, self.lut_editor_widget
        ]
        for widget in all_param_widgets:
            widget.blockSignals(True)

        if op.type == "Enhanced EDT":
            self.edt_look_forward_edit.setText(str(op.look_forward))
            self.edt_look_backward_edit.setText(str(op.look_backward))
            self.edt_fade_dist_edit.setText(str(op.fade_distance_limit))
            self.details_stack.setCurrentWidget(self.edt_params_widget)
        elif op.type == "GPU Shader":
            self.shader_file_edit.setText(op.shader_file)
            self.details_stack.setCurrentWidget(self.shader_params_widget)
        elif op.type == "Gaussian Blur":
            self.blur_ksize_edit.setText(str(op.gaussian_ksize_x))
            self.details_stack.setCurrentWidget(self.blur_params_widget)
        elif op.type == "Apply LUT":
            self.lut_editor_widget.set_lut_params(op.lut_params)
            self.details_stack.setCurrentWidget(self.lut_editor_widget)
        elif op.type in ["Multiply", "Screen", "Overlay"]:
            self.details_stack.setCurrentWidget(self.blend_params_widget)
        else:
            self.details_stack.setCurrentWidget(self.placeholder_widget)

        # Unblock signals
        for widget in all_param_widgets:
            widget.blockSignals(False)

    @Slot()
    def _update_current_op_params(self):
        """Saves the current state of the parameter editors to the config."""
        current_row = self.pipeline_table.currentRow()
        if current_row < 0: return

        op = app_config.pipeline[current_row]

        try:
            if op.type == "Enhanced EDT":
                op.look_forward = int(self.edt_look_forward_edit.text())
                op.look_backward = int(self.edt_look_backward_edit.text())
                op.fade_distance_limit = float(self.edt_fade_dist_edit.text())
            elif op.type == "GPU Shader":
                op.shader_file = self.shader_file_edit.text()
            elif op.type == "Gaussian Blur":
                ksize = int(self.blur_ksize_edit.text())
                op.gaussian_ksize_x = ksize if ksize % 2 != 0 else ksize + 1 # Ensure odd
            elif op.type in ["Multiply", "Screen", "Overlay"]:
                op.blend_mode = op.type.lower()
            elif op.type == "Apply LUT":
                # The lut_editor already updated its internal params object,
                # which is the same object in our config. So, nothing to do here.
                pass
        except (ValueError, TypeError) as e:
            print(f"Warning: Invalid value in parameter fields. {e}")
            # Optionally, show a visual error indication on the widget

    @Slot()
    def _browse_for_shader(self):
        current_row = self.pipeline_table.currentRow()
        if current_row < 0: return
        op = app_config.pipeline[current_row]

        filepath, _ = QFileDialog.getOpenFileName(self, "Select Shader File", op.shader_file, "Shader Files (*.glsl *.fs *.vs)")
        if filepath:
            self.shader_file_edit.setText(filepath)
            self._update_current_op_params() # Update config immediately
