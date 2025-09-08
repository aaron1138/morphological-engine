# Agent Development Guide: Voxel Slice Pre-Processor

## 1. Project Status & Goal

This is an **existing Python application** built with PySide6. The goal is not to build a new framework from scratch, but to **refactor and enhance** the current codebase by integrating high-performance libraries.

The end-goal is to replace the current CPU-based, NumPy-powered processing pipeline with a new GPU-accelerated pipeline built on OpenVDB, ModernGL, and NanoVDB.

## 2. Technology Stack

### Existing Technologies:
- **GUI:** PySide6
- **Core Logic:** Python 3.10+
- **Data Structures:** NumPy
- **Image I/O:** OpenCV-Python (`cv2`)

### Technologies to Integrate:
- **Voxel Engine:** OpenVDB 12.1.0
- **GPU Compute:** ModernGL
- **GPU Voxel Structure:** NanoVDB
- **CPU Acceleration:** Numba
- **Testing:** Pytest

## 3. Codebase Architecture

The project is structured into three main directories:

- `core/`: Contains the backend data processing logic.
- `gui/`: Contains all PySide6 UI components and the processing thread that connects the UI to the core.
- `utils/`: Contains helper modules for configuration and file management.

### Execution Flow:
1.  `main.py` starts the PySide6 `QApplication` and `MainWindow`.
2.  The user interacts with the GUI (`gui/`) to load a sequence of 2D images (`core/slice_loader`).
3.  The user configures a processing pipeline via `gui/parameter_panel.py`.
4.  On "Run", `gui/processing_thread.py` is started.
5.  The thread instantiates `core/voxel_engine.py`, which iterates through the 2D slices and creates 3D "voxel windows" as NumPy arrays.
6.  For each window, `core/processing_pipeline.py` is called to perform a series of CPU-based morphological operations.
7.  The result is saved back to disk.

## 4. Integration Strategy

The refactoring will proceed as follows:

1.  **`core/voxel_engine.py`:** This module will be refactored to use `openvdb` grids instead of `numpy` arrays. The `iter_windows()` generator will be modified to yield `openvdb.FloatGrid` objects.
2.  **`core/gpu_processor.py` (New Module):** A new module will be created to handle all low-level GPU interactions. It will manage the ModernGL context, shaders, and the conversion of OpenVDB grids to NanoVDB buffers on the GPU.
3.  **`core/gpu_pipeline.py` (New Module):** A new pipeline class will be created to replace the existing `ProcessingPipeline`. It will take an `openvdb.FloatGrid`, use the `gpu_processor` to perform operations on the GPU via GLSL shaders, and return the result.
4.  **`gui/processing_thread.py`:** This will be updated to use the new `GpuPipeline`.
5.  **`gui/parameter_panel.py`:** The UI will be modified to allow users to define a pipeline of GLSL shaders instead of the old, hardcoded CPU operations.
6.  **Configuration & Logging:** New utilities will be added to manage application-wide settings (`config.json`) and log processing runs (`runs.json`).
