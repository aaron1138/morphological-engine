# Environment Setup Instructions

This document details the steps required to set up the development environment for the voxel processing engine.

## Dependencies

The following primary libraries are required:
- Python 3.9+
- OpenVDB 12.1.0
- NanoVDB
- ModernGL
- Numba

## Installation Steps

### 1. Installable Packages

The following packages can be installed directly using pip:

```bash
pip install moderngl numba opencv-python PyQt6
```

### 2. Manual Installation Required

**IMPORTANT:** The core libraries `OpenVDB` and `NanoVDB` cannot be installed via pip in a standard Python environment. They are complex C++/CUDA libraries that must be built from source.

- **`pyopenvdb` (for OpenVDB):** You must follow the official OpenVDB documentation to build and install the library and its Python bindings from source. Ensure you are targeting version 12.1.0. This will require a C++ compiler (g++, clang) and dependencies like Boost and TBB.

- **`py-nanovdb` (for NanoVDB):** Similarly, the NanoVDB Python bindings must be built from source. Please refer to the official NVIDIA NanoVDB repository for instructions.

The code in this repository is written *assuming* that these two libraries have been successfully installed in your environment.
