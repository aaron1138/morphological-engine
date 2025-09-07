# Project Setup Guide

This document provides step-by-step instructions for setting up the development environment for this project. The core of the project relies on a specific version of OpenVDB built from source with Python and NanoVDB support, along with several Python packages for GPU programming and numerical computation.

## 1. System Prerequisites

Before you begin, you need a C++17 compatible compiler, Git, and CMake.

**On Debian/Ubuntu:**
```bash
sudo apt-get update
sudo apt-get install -y build-essential g++ git cmake
```

**On Windows:**
- Install [Visual Studio](https://visualstudio.microsoft.com/downloads/) with the "Desktop development with C++" workload.
- Install [Git](https://git-scm.com/download/win).
- Install [CMake](https://cmake.org/download/).
- Install [vcpkg](https://github.com/microsoft/vcpkg) and set the `VCPKG_DEFAULT_TRIPLET` environment variable to `x64-windows`.

---

## 2. Python Environment

It is highly recommended to use a Python virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate  # On Linux/macOS
# .\.venv\Scripts\activate  # On Windows
```

Install the required Python packages. These are necessary for both running the application and for building the `pyopenvdb` module from source.

```bash
pip install moderngl numba numpy nanobind Pillow opencv-python
```

- **ModernGL**: For GPU context and shader management.
- **Numba**: For JIT-compilation of Python math functions.
- **NumPy**: A dependency for `pyopenvdb`.
- **nanobind**: A dependency for `pyopenvdb` Python bindings.
- **Pillow**: For reading/writing image files (e.g., PNGs).

---

## 3. Building OpenVDB 12.1.0 from Source

This is the most critical part of the setup. We need to build OpenVDB from source to enable the specific modules we need (Python bindings and NanoVDB).

### 3.1. Install C++ Dependencies

**On Debian/Ubuntu:**
These libraries are required to build OpenVDB.
```bash
sudo apt-get install -y libboost-iostreams-dev libtbb-dev libblosc-dev zlib1g-dev
```

**On Windows (using vcpkg):**
```bash
vcpkg install boost-iostreams:x64-windows tbb:x64-windows blosc:x64-windows zlib:x64-windows
```

### 3.2. Clone and Configure OpenVDB

1.  **Clone the repository and check out the correct version tag:**
    ```bash
    git clone https://github.com/AcademySoftwareFoundation/openvdb.git
    cd openvdb
    git checkout v12.1.0
    ```

2.  **Create a build directory:**
    ```bash
    mkdir build
    cd build
    ```

3.  **Configure the build using CMake.** The following command enables the Python module and NanoVDB support.

    **On Linux/macOS:**
    Make sure your virtual environment is activated so CMake can find the correct Python installation.
    ```bash
    cmake .. \
      -D CMAKE_BUILD_TYPE=Release \
      -D OPENVDB_BUILD_PYTHON_MODULE=ON \
      -D OPENVDB_BUILD_NANOVDB=ON \
      -D NANOVDB_BUILD_TOOLS=ON \
      -D USE_NANOVDB=ON
    ```

    **On Windows:**
    You need to point CMake to the `vcpkg` toolchain file.
    ```bash
    cmake .. ^
      -G "Visual Studio 17 2022" -A x64 ^
      -D CMAKE_TOOLCHAIN_FILE="C:/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake" ^
      -D CMAKE_BUILD_TYPE=Release ^
      -D OPENVDB_BUILD_PYTHON_MODULE=ON ^
      -D OPENVDB_BUILD_NANOVDB=ON ^
      -D NANOVDB_BUILD_TOOLS=ON ^
      -D USE_NANOVDB=ON
    ```
    *(Adjust the path to your `vcpkg` installation and your Visual Studio version accordingly)*

### 3.3. Compile and Install

After CMake has successfully configured the project, compile and install it.

**On Linux/macOS:**
The `-j` flag specifies the number of parallel jobs. Adjust it based on your CPU cores.
```bash
make -j$(nproc)
sudo make install
```
This will install the OpenVDB libraries to a system-wide location (like `/usr/local/lib`) and the `pyopenvdb` Python module into your activated virtual environment's `site-packages`.

**On Windows:**
```bash
cmake --build . --config Release --parallel
cmake --build . --config Release --target install
```

---

## 4. Verification

After installation, you should be able to import `pyopenvdb` and `ModernGL` in Python without errors.

```python
import pyopenvdb as vdb
import ModernGL as mgl
import numba
import numpy

print("Successfully imported pyopenvdb, ModernGL, Numba, and NumPy!")
print(f"OpenVDB version: {vdb.openvdb_version_string()}")
```

If the script runs without errors and prints the version string, your environment is set up correctly.
