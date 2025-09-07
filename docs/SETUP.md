# Environment Setup for Voxel Engine

This document outlines the steps required to set up the development environment for the Python-based voxel processing engine. The setup involves compiling OpenVDB from source with specific flags and installing several Python packages.

## Core Components

*   **Python 3.9+**
*   **OpenVDB 12.1.0**: For voxel data structures.
*   **NanoVDB**: For GPU-accelerated voxel structures.
*   **ModernGL**: For OpenGL rendering and GPGPU tasks.
*   **Numba**: for JIT-compilation of Python code.
*   **NumPy**: For numerical operations.
*   **nanobind**: For C++/Python bindings, required by OpenVDB's Python module.

## Platform

These instructions are tailored for a Debian-based Linux distribution (e.g., Ubuntu 22.04). For instructions on other platforms like macOS or Windows, please refer to the [official OpenVDB build documentation](https://www.openvdb.org/documentation/doxygen/build.html).

---

### Step 1: Install System Dependencies

First, install the necessary build tools and libraries required by OpenVDB.

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    libboost-iostreams-dev \
    libtbb-dev \
    libblosc-dev \
    libglfw3-dev \
    libglew-dev
```

### Step 2: Install Python Dependencies

Install the required Python packages using `pip`. It is highly recommended to do this within a virtual environment.

```bash
pip install numpy
pip install nanobind
pip install moderngl
pip install numba
pip install Pillow
```

### Step 3: Build and Install OpenVDB from Source

This is the most critical step. We will clone the OpenVDB repository, check out the correct version (12.1.0), and build it from source using CMake. The flags provided to CMake are essential for enabling the Python and NanoVDB modules.

```bash
# Clone the repository
git clone https://github.com/AcademySoftwareFoundation/openvdb.git
cd openvdb

# Check out the specified version
git fetch --tags
git checkout tags/v12.1.0 -b v12.1.0

# Create a build directory
mkdir build
cd build

# Configure the build with CMake
# This command enables the Python and NanoVDB modules.
# It also points to the nanobind installation if needed, though pip should handle this.
cmake .. \
    -D CMAKE_BUILD_TYPE=Release \
    -D OPENVDB_BUILD_PYTHON_MODULE=ON \
    -D OPENVDB_BUILD_NANOVDB=ON \
    -D USE_NANOVDB=ON

# Compile and install OpenVDB
# The -j flag specifies the number of parallel jobs (set to number of CPU cores)
make -j$(nproc)

# Install the compiled libraries and Python module
# This may require sudo if installing to a system directory.
# To install to a custom location, use -D CMAKE_INSTALL_PREFIX=/path/to/install
sudo make install
```

### Step 4: Verify the Installation

After the installation is complete, you should be able to import the OpenVDB and ModernGL modules in Python without errors.

```python
import pyopenvdb as vdb
import moderngl
import numba
import numpy

print("Successfully imported OpenVDB version:", vdb.getVersionString())
print("ModernGL, Numba, and NumPy are also available.")
```

If the `pyopenvdb` module cannot be found, you may need to add the installation path to your `PYTHONPATH` environment variable. If you used the default install prefix (`/usr/local`), the path would be `/usr/local/lib/python3.x/site-packages` (replace `3.x` with your Python version).

Example:
```bash
export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.10/site-packages
```
