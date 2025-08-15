# -*- coding: utf-8 -*-
"""
Module: parameter_panel.py
Author: Gemini
Description: A PyQt6 widget for dynamically configuring the processing pipeline.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QComboBox,
    QPushButton, QListWidget, QListWidgetItem, QSpinBox,
)
from PyQt6.QtCore import Qt, pyqtSignal
from typing import Dict, Any, List

class OperationWidget(QWidget):
    """
    A widget representing a single operation in the processing pipeline.
    It contains controls for the operation's parameters (e.g., iterations).
    """
    def __init__(self, op_name: str):
        super().__init__()
        self.op_name = op_name
        
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        self.label = QLabel(f"<b>{op_name.replace('_', ' ').title()}</b>")
        layout.addWidget(self.label)
        
        # Add parameter controls based on operation type
        self.param_widgets = {}
        if op_name in ["erode", "dilate", "open", "close"]:
            self.iterations_label = QLabel("Iterations:")
            self.iterations_spinbox = QSpinBox()
            self.iterations_spinbox.setRange(1, 99)
            self.iterations_spinbox.setValue(1)
            layout.addWidget(self.iterations_label)
            layout.addWidget(self.iterations_spinbox)
            self.param_widgets["iterations"] = self.iterations_spinbox
        
        self.setLayout(layout)

    def get_params(self) -> Dict[str, Any]:
        """Returns the parameters for this operation as a dictionary."""
        params = {"operation": self.op_name}
        for key, widget in self.param_widgets.items():
            params[key] = widget.value()
        return params

class ParameterPanel(QFrame):
    """
    The main panel for building and managing the processing pipeline steps.
    """
    # Signal to emit when the configuration might change RAM usage
    config_changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        
        main_layout = QVBoxLayout(self)
        
        # --- Title ---
        title_label = QLabel("Processing Pipeline")
        title_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        main_layout.addWidget(title_label)
        
        # --- List of Operations ---
        self.ops_list_widget = QListWidget()
        self.ops_list_widget.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.ops_list_widget.setStyleSheet("QListWidget::item { border-bottom: 1px solid #ccc; }")
        main_layout.addWidget(self.ops_list_widget)
        
        # --- Add Operation Controls ---
        add_op_layout = QHBoxLayout()
        self.op_combo = QComboBox()
        self.op_combo.addItems([
            "gradient_sobel", "gradient_scharr", "erode", 
            "dilate", "open", "close"
        ])
        
        self.add_op_button = QPushButton("Add Step")
        self.add_op_button.clicked.connect(self.add_operation)
        
        add_op_layout.addWidget(self.op_combo, 1)
        add_op_layout.addWidget(self.add_op_button)
        main_layout.addLayout(add_op_layout)
        
        # --- Remove Operation Button ---
        self.remove_op_button = QPushButton("Remove Selected Step")
        self.remove_op_button.clicked.connect(self.remove_operation)
        main_layout.addWidget(self.remove_op_button)

    def add_operation(self):
        """Adds a new operation to the pipeline list."""
        op_name = self.op_combo.currentText()
        op_widget = OperationWidget(op_name)
        
        list_item = QListWidgetItem(self.ops_list_widget)
        # Set a hint for the size, especially important for custom widgets
        list_item.setSizeHint(op_widget.sizeHint())
        
        self.ops_list_widget.addItem(list_item)
        self.ops_list_widget.setItemWidget(list_item, op_widget)
        self.config_changed.emit()

    def remove_operation(self):
        """Removes the currently selected operation from the list."""
        current_row = self.ops_list_widget.currentRow()
        if current_row >= 0:
            self.ops_list_widget.takeItem(current_row)
            self.config_changed.emit()

    def get_config(self) -> Dict[str, Any]:
        """
        Builds and returns the configuration dictionary from the UI widgets.
        """
        steps = []
        for i in range(self.ops_list_widget.count()):
            item = self.ops_list_widget.item(i)
            op_widget = self.ops_list_widget.itemWidget(item)
            if isinstance(op_widget, OperationWidget):
                steps.append(op_widget.get_params())
        
        return {"steps": steps}

    def set_config(self, config: Dict[str, Any]):
        """
        Populates the UI from a loaded configuration dictionary.
        """
        self.ops_list_widget.clear()
        if "steps" in config:
            for step in config["steps"]:
                op_name = step.get("operation")
                if op_name:
                    op_widget = OperationWidget(op_name)
                    # Set parameters from config
                    for key, value in step.items():
                        if key in op_widget.param_widgets:
                            op_widget.param_widgets[key].setValue(value)
                    
                    list_item = QListWidgetItem(self.ops_list_widget)
                    list_item.setSizeHint(op_widget.sizeHint())
                    self.ops_list_widget.addItem(list_item)
                    self.ops_list_widget.setItemWidget(list_item, op_widget)
        self.config_changed.emit()
