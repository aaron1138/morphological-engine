# -*- coding: utf-8 -*-
"""
Module: processing_pipeline.py
Author: Gemini
Description: Orchestrates the entire 3D processing workflow. It takes a
             configuration object and a voxel window, then applies a sequence
             of operations (e.g., gradient, morphology) to it.
"""

import numpy as np
from typing import Dict, Any, List, Tuple

# --- Import our core processing modules ---
try:
    from core import operators_3d
    from core import morphology
except ImportError:
    print("Warning: Could not import core modules. Standalone testing may fail.")
    class operators_3d:
        def apply_3d_gradient(*args, **kwargs): return np.zeros((1,1,1))
    class morphology:
        def apply_3d_erosion(*args, **kwargs): return np.zeros((1,1,1))
        def apply_3d_dilation(*args, **kwargs): return np.zeros((1,1,1))
        def apply_3d_opening(*args, **kwargs): return np.zeros((1,1,1))
        def apply_3d_closing(*args, **kwargs): return np.zeros((1,1,1))

class ProcessingPipeline:
    """
    Manages and executes a configurable pipeline of 3D image processing steps.
    """
    OPERATION_MAP = {
        "gradient_sobel": operators_3d.apply_3d_gradient,
        "gradient_scharr": operators_3d.apply_3d_gradient,
        "erode": morphology.apply_3d_erosion,
        "dilate": morphology.apply_3d_dilation,
        "open": morphology.apply_3d_opening,
        "close": morphology.apply_3d_closing,
    }

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._validate_config()
        
    def _validate_config(self):
        if "steps" not in self.config or not isinstance(self.config["steps"], list):
            raise ValueError("Configuration must contain a 'steps' list.")
        for i, step in enumerate(self.config["steps"]):
            if "operation" not in step:
                raise ValueError(f"Step {i} is missing the 'operation' key.")
            if step["operation"] not in self.OPERATION_MAP:
                raise ValueError(f"Unknown operation in step {i}: {step['operation']}")

    def run(self, voxel_window: np.ndarray, debug: bool = False) -> Tuple[np.ndarray, List[Tuple[str, np.ndarray]]]:
        """
        Executes the full pipeline on a given voxel window.

        Args:
            voxel_window (np.ndarray): The input 3D data block.
            debug (bool): If True, returns intermediate steps for debugging.

        Returns:
            A tuple containing:
            - np.ndarray: The final processed 3D data block.
            - List[Tuple[str, np.ndarray]]: A list of (name, data) for each debug step.
        """
        processed_data = voxel_window.copy()
        debug_steps = []

        # print("\n--- Running Processing Pipeline ---") # Commented out for cleaner GUI output
        for i, step in enumerate(self.config["steps"]):
            op_name = step["operation"]
            func = self.OPERATION_MAP[op_name]
            params = step.copy()
            
            if op_name == "gradient_sobel": params["operator"] = "sobel"
            elif op_name == "gradient_scharr": params["operator"] = "scharr"
            if "operation" in params: del params["operation"]

            # print(f"Step {i+1}: Applying '{op_name}' with params: {params}")
            processed_data = func(voxel_window=processed_data, **params)
            
            if debug:
                debug_name = f"{i+1:02d}_{op_name}"
                debug_steps.append((debug_name, processed_data.copy()))
            
        # print("--- Pipeline Finished ---")
        return processed_data, debug_steps
