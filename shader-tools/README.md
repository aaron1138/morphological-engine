# Python Shader Toolchain

This project provides a command-line backend and a graphical user interface for applying RetroArch `.slangp` shader presets to static images.

## Features

- **Backend (`shader_tool.py`):** A powerful command-line tool for applying shader presets.
- **Frontend (`shader_gui.py`):** A user-friendly GUI for single and batch image processing.
- **High-Performance:** Uses ModernGL for headless OpenGL processing.
- **Broad Image Support:** Uses Pillow to support a wide variety of image formats.

## Setup

1.  **Install `slangc`:** The `slangc` compiler is part of RetroArch. You must install RetroArch and ensure the `slangc` executable is in your system's PATH.
2.  **Install Python Dependencies:** Navigate to this directory in your terminal and run:
    ```bash
    pip install -r requirements.txt
    ```
    **Note:** The `moderngl[headless]` dependency is used to ensure that the necessary libraries for offscreen GPU processing are installed correctly. If you are having issues, you may need to uninstall existing versions first: `pip uninstall moderngl glcontext` before running the install command again.

## Usage

### Backend (`shader_tool.py`)

The backend script is used for processing a single image.

**Command:**
```bash
python shader_tool.py <input_image> <shader_preset> <output_image>
```

**Arguments:**
- `input_image`: Path to the input image file.
- `shader_preset`: Path to the `.slangp` shader preset file.
- `output_image`: Path to save the processed output PNG file.

### Frontend (`shader_gui.py`)

The frontend provides a graphical interface for easier use, including batch processing.

**Command:**
```bash
python shader_gui.py
```

**Interface:**
1.  **Select Input Image / Input Folder:** Choose a single image or a folder of images to process.
2.  **Select Shader Preset:** Choose the `.slangp` file.
3.  **Select Output Folder:** Choose where to save the processed images.
4.  **Start Processing:** Begin the operation. Progress will be displayed in the log window.
