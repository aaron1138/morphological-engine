# Development Environment Setup on Windows

This guide details the process for setting up the development environment from a clean Windows 10/11 (64-bit) installation. This project relies on several C++ libraries that must be compiled from source.

## Table of Contents
1.  [Prerequisites](#prerequisites)
2.  [Step 1: Install Python](#step-1-install-python)
3.  [Step 2: Create a Virtual Environment](#step-2-create-a-virtual-environment)
4.  [Step 3: Set Up Vcpkg (C++ Package Manager)](#step-3-set-up-vcpkg-c-package-manager)
5.  [Step 4: Install C++ Dependencies](#step-4-install-c-dependencies)
6.  [Step 5: Compile and Install OpenVDB](#step-5-compile-and-install-openvdb)
7.  [Step 6: Install Python Dependencies](#step-6-install-python-dependencies)
8.  [Step 7: Verification](#step-7-verification)

---

### Prerequisites

Before you begin, ensure you have the following software installed:

1.  **Git:** Download and install from [git-scm.com](https://git-scm.com/download/win).
2.  **Visual Studio 2019 or later:** Download the Community version from the [Visual Studio website](https://visualstudio.microsoft.com/downloads/). During installation, you **must** select the **"Desktop development with C++"** workload.
3.  **CMake:** Download and install the latest version from [cmake.org](https://cmake.org/download/). Make sure to select the option "Add CMake to the system PATH for all users" during installation.

---

### Step 1: Install Python

This project requires Python 3.10 or newer.

1.  Download the Python 3.10 installer from the [official Python website](https://www.python.org/downloads/windows/).
2.  Run the installer.
3.  **Crucially, check the box that says "Add Python to PATH"** on the first screen of the installer.
4.  Complete the installation with the default settings.

---

### Step 2: Create a Virtual Environment

Using a virtual environment is essential to isolate project dependencies.

1.  Open a new Command Prompt (`cmd.exe`) or PowerShell.
2.  Navigate to your project's root directory.
3.  Create the virtual environment:
    ```bash
    python -m venv .venv
    ```
4.  Activate the virtual environment. This must be done every time you open a new terminal to work on the project:
    ```bash
    .\.venv\Scripts\activate
    ```
    Your terminal prompt should now be prefixed with `(.venv)`.

---

### Step 3: Set Up Vcpkg (C++ Package Manager)

Vcpkg simplifies acquiring and building C++ libraries on Windows.

1.  Choose a location to install `vcpkg` (e.g., `C:\dev\vcpkg`).
2.  Open a new terminal and clone the repository:
    ```bash
    git clone https://github.com/microsoft/vcpkg.git
    cd vcpkg
    ```
3.  Run the bootstrap script:
    ```bash
    .\bootstrap-vcpkg.bat
    ```
4.  Set the following environment variable to ensure `vcpkg` builds 64-bit libraries by default. You can set this permanently in your system settings or run it in your active terminal session.
    ```bash
    set VCPKG_DEFAULT_TRIPLET=x64-windows
    ```

---

### Step 4: Install C++ Dependencies

With `vcpkg` ready, install the libraries required by OpenVDB. This process can take a significant amount of time.

Run the following commands from within the `vcpkg` directory:

```bash
.\vcpkg install zlib:x64-windows
.\vcpkg install blosc:x64-windows
.\vcpkg install tbb:x64-windows
.\vcpkg install boost-iostreams:x64-windows
.\vcpkg install boost-any:x64-windows
.\vcpkg install boost-algorithm:x64-windows
.\vcpkg install boost-interprocess:x64-windows
```

---

### Step 5: Compile and Install OpenVDB

Now we will compile OpenVDB 12.1.0 and its Python bindings.

1.  Clone the OpenVDB repository. Navigate to a suitable directory (e.g., `C:\dev`) and run:
    ```bash
    git clone https://github.com/AcademySoftwareFoundation/openvdb.git -b v12.1.0 --recursive
    cd openvdb
    ```
2.  Create a build directory:
    ```bash
    mkdir build
    cd build
    ```
3.  Run CMake to configure the build. **Replace `<PATH_TO_VCPKG>` with the actual path** to your `vcpkg` installation (e.g., `C:\dev\vcpkg`).

    ```bash
    cmake .. -G "Visual Studio 17 2022" -A x64 \
             -DCMAKE_TOOLCHAIN_FILE="<PATH_TO_VCPKG>\scripts\buildsystems\vcpkg.cmake" \
             -DVCPKG_TARGET_TRIPLET=x64-windows \
             -DOPENVDB_BUILD_PYTHON_MODULE=ON \
             -DUSE_NUMPY=ON \
             -DUSE_NANOVDB=ON \
             -DPYTHON_EXECUTABLE=".\.venv\Scripts\python.exe" \
             -DCMAKE_INSTALL_PREFIX=".\install"
    ```
    *Note: Adjust the Visual Studio generator (`-G`) if you are using a different version.*

4.  Build and install OpenVDB. This will compile the libraries and the Python module.
    ```bash
    cmake --build . --config Release --target install --parallel %NUMBER_OF_PROCESSORS%
    ```
5.  The compiled Python wheel (`.whl`) will be located in the `build\python\dist` directory. Install it using pip (ensure your virtual environment is still active):
    ```bash
    pip install .\python\dist\pyopenvdb-*.whl
    ```

---

### Step 6: Install Python Dependencies

The project requires several other Python packages.

1.  Make sure you are in the project's root directory and your virtual environment is active.
2.  Create a file named `requirements.txt` with the following content:
    ```
    PySide6
    ModernGL
    numba
    pytest
    numpy
    opencv-python
    pyopengl
    ```
3.  Install the packages:
    ```bash
    pip install -r requirements.txt
    ```

---

### Step 7: Verification

To ensure all components are installed correctly, run this simple Python script.

1.  Save the following as `verify_install.py` in your project root:
    ```python
    try:
        import pyopenvdb
        import ModernGL
        import numba
        import cv2
        import OpenGL.GL as gl
        from PySide6.QtWidgets import QApplication
        print("✅ All major libraries imported successfully!")
    except ImportError as e:
        print(f"❌ Failed to import a library: {e}")
    ```
2.  Run the script from your activated virtual environment:
    ```bash
    python verify_install.py
    ```
You should see the success message. If not, re-check the previous steps for errors. You are now ready to run the application.
