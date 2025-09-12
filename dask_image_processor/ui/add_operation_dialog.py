# ui/add_operation_dialog.py

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QComboBox, QDialogButtonBox
)
from config import OperationType, OperationPlane

class AddOperationDialog(QDialog):
    """A dialog to add a new operation to the pipeline."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Pipeline Operation")

        layout = QVBoxLayout(self)

        self.type_combo = QComboBox()
        self.type_combo.addItems([e.value for e in OperationType])
        layout.addWidget(self.type_combo)

        self.plane_combo = QComboBox()
        self.plane_combo.addItems([e.value for e in OperationPlane])
        layout.addWidget(self.plane_combo)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def get_selection(self):
        """Returns the selected operation type and plane."""
        return self.type_combo.currentText(), self.plane_combo.currentText()
