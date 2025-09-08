# Voxel Slice Pre-Processor: Development Environment Setup

This guide provides detailed instructions for setting up the development environment for the Voxel Slice Pre-Processor project on a clean **Windows 10/11** installation.

## Introduction

The project relies on a specific toolchain involving C++ libraries (OpenVDB, NanoVDB) and Python packages. Following these steps carefully is crucial for a successful build. The most complex part is compiling OpenVDB from source with the correct options. We will use the `vcpkg` C++ package manager to simplify the acquisition of dependencies.

---

## Part 1: Core Development Environment

These are the essential tools required to build the project.

### Step 1: Install Visual Studio 2022
Visual Studio is required for its C++ compiler (MSVC).

1.  Download the **Visual Studio 2022 Community Edition** installer from the [official Microsoft website](https://visualstudio.microsoft.com/downloads/).
2.  Run the installer.
3.  In the "Workloads" tab, select **"Desktop development with C++"**. This will install the necessary C++ toolchain, CMake, and MSVC compiler.
4.  Proceed with the installation.

### Step 2: Install Python
The project requires Python 3.10 or newer.

1.  Download the **Python 3.10** (or later) installer from the [official Python website](https://www.python.org/downloads/windows/).
2.  Run the installer.
3.  **Important:** On the first page of the installer, check the box that says **"Add Python to PATH"**.
4.  Complete the installation with the default options.

### Step 3: Install Git
Git is required for version control and for cloning the necessary repositories.

1.  Download and install **Git for Windows** from the [official website](https://git-scm.com/download/win).
2.  Use the default options during installation.

---

## Part 2: C++ Dependencies via vcpkg

We use `vcpkg` to download and build the C++ libraries that OpenVDB depends on.

### Step 1: Install vcpkg
1.  Open a new **PowerShell** or **Command Prompt**.
2.  Choose a location to install `vcpkg`. A good choice is `C:\dev\vcpkg` or a similar path without spaces.
3.  Run the following commands:
    ```bash
    git clone https://github.com/Microsoft/vcpkg.git
    cd vcpkg
    ./bootstrap-vcpkg.bat
    ```

### Step 2: Set up vcpkg (User-wide integration)
To make vcpkg libraries automatically available to Visual Studio and CMake, run this command from the `vcpkg` directory:
```bash
./vcpkg integrate install
```

### Step 3: Install Required Libraries
Now, use vcpkg to install the dependencies for OpenVDB. These commands can take a significant amount of time to complete as they are building the libraries from source.

Run these commands from the `vcpkg` directory:
```bash
# Set the default triplet to 64-bit Windows
setx VCPKG_DEFAULT_TRIPLET x64-windows

# Install dependencies
./vcpkg install tbb:x64-windows
./vcpkg install zlib:x64-windows
./vcpkg install blosc:x64-windows
./vcpkg install boost-iostreams:x64-windows
./vcpkg install boost-any:x64-windows
./vcpkg install boost-algorithm:x64-windows
./vcpkg install boost-interprocess:x64-windows
```
**Note:** You may need to restart your terminal for the `VCPKG_DEFAULT_TRIPLET` environment variable to be recognized.

---

## Part 3: Building OpenVDB with Python Bindings

This is the core compilation step.

### Step 1: Clone the OpenVDB Repository
1.  Navigate to your development projects directory (e.g., `C:\dev`).
2.  Clone the OpenVDB repository and check out the target version **12.1.0**:
    ```bash
    git clone https://github.com/AcademySoftwareFoundation/openvdb.git
    cd openvdb
    git checkout v12.1.0
    ```

### Step 2: Configure the Build with CMake
1.  Create a build directory inside the `openvdb` folder:
    ```bash
    mkdir build
    cd build
    ```
2.  Run `cmake` to configure the build. This command points to the `vcpkg` toolchain file and enables the options we need (Python bindings and NanoVDB). Replace `<PATH_TO_VCPKG>` with the actual path to your `vcpkg` installation (e.g., `C:\dev\vcpkg`).

    ```bash
    cmake .. -G "Visual Studio 17 2022" -A x64 \
      -DCMAKE_TOOLCHAIN_FILE=<PATH_TO_VCPKG>/scripts/buildsystems/vcpkg.cmake \
      -DOPENVDB_BUILD_PYTHON_MODULE=ON \
      -DUSE_NANOVDB=ON \
      -DNANOVDB_BUILD_EXAMPLES=OFF \
      -DNANOVDB_BUILD_UNITTESTS=OFF \
      -DOPENVDB_BUILD_UNITTESTS=OFF \
      -DOPENVDB_BUILD_CORE_TESTS=OFF \
      -DOPENVDB_ENABLE_RPATH=OFF
    ```

### Step 3: Compile and Install OpenVDB
1.  From the `build` directory, run the following command to compile the project. This will take a long time.
    ```bash
    cmake --build . --config Release --parallel
    ```
2.  After the build succeeds, the `pyopenvdb` module will be located in a path similar to: `openvdb\build\python\Release\`.

### Step 4: Make pyopenvdb Available to Python
To make the compiled `pyopenvdb` module accessible to your Python environment, you have two options:

1.  **Recommended (System-wide):** Copy the `pyopenvdb.pyd` file and the `openvdb.dll` file from the build output directory into your Python installation's `Lib/site-packages` directory.
2.  **Alternative (Per-project):** Add the output directory (`openvdb\build\python\Release\`) to the `PYTHONPATH` environment variable.

---

## Part 4: Python Environment Setup

### Step 1: Create a Virtual Environment
1.  Navigate to the root of this project repository.
2.  Create a virtual environment:
    ```bash
    python -m venv .venv
    ```
3.  Activate the virtual environment:
    ```bash
    .venv\Scripts\activate
    ```

### Step 2: Install Python Packages
Install all required Python packages using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

---

## Part 5: Final Sanity Check

After completing all the steps, run a final check to ensure all components are correctly installed and accessible.

1.  Activate your virtual environment (`.venv\Scripts\activate`).
2.  Open a Python interpreter by typing `python`.
3.  Run the following commands. If no `ImportError` occurs, the setup is successful.
    ```python
    import openvdb
    import moderngl
    import PySide6
    import numba

    print("Successfully imported all core libraries!")
    print("OpenVDB version:", openvdb.version)
    ```
