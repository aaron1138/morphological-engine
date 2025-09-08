# Setup Guide: Voxel Pre-Processor on Windows 10/11

This guide details the complete process for setting up the development environment from a clean Windows installation. Following these steps will ensure you have all the necessary compilers, libraries, and Python packages to run and develop the application.

## 1. Prerequisites: Core Development Tools

These tools are required to compile the C++ dependencies and the OpenVDB library itself.

### 1.1. Visual Studio 2022

Visual Studio provides the C++ compiler (MSVC) and build tools.

- Download the **Visual Studio 2022 Community Edition** installer from the [official Microsoft website](https://visualstudio.microsoft.com/downloads/).
- During installation, select the **"Desktop development with C++"** workload. This includes the necessary C++ toolchain, Windows SDK, and CMake.
- Ensure that the **"C++ CMake tools for Windows"** component is selected within that workload.

### 1.2. Git for Windows

Git is required for cloning the necessary source code repositories.

- Download and install **Git for Windows** from the [official website](https://git-scm.com/download/win).
- You can leave the default installation options.

## 2. C++ Dependencies via vcpkg

We will use `vcpkg`, a C++ package manager from Microsoft, to install the libraries OpenVDB depends on. This is the most reliable method on Windows.

### 2.1. Install vcpkg

- Open a **PowerShell** or **Command Prompt** terminal.
- Choose a location to install `vcpkg`. A good choice is `C:\dev\vcpkg` or a similar path without spaces.

```bash
# Clone the vcpkg repository
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg

# Run the bootstrap script to build vcpkg
.\bootstrap-vcpkg.bat
```

### 2.2. Install OpenVDB Dependencies

- From the `vcpkg` directory in your terminal, run the following commands to install the required libraries for OpenVDB. This step can take a significant amount of time.
- **Important**: We recommend setting the default triplet to `x64-windows` to avoid issues with 32-bit vs 64-bit builds.

```bash
# Set the environment variable for the current session
$env:VCPKG_DEFAULT_TRIPLET = 'x64-windows'

# Or set it permanently (requires administrator terminal)
setx VCPKG_DEFAULT_TRIPLET "x64-windows"

# Install the dependencies
.\vcpkg install zlib blosc tbb boost-iostreams boost-any boost-algorithm boost-interprocess
```

## 3. Build OpenVDB v12.1.0 from Source

Now we will clone and compile the OpenVDB library itself.

### 3.1. Clone OpenVDB

- Navigate to the directory where you want to store the OpenVDB source code (e.g., `C:\dev`).
- Clone the repository and check out the specified version tag.

```bash
git clone https://github.com/AcademySoftwareFoundation/openvdb.git
cd openvdb
git checkout v12.1.0
```

### 3.2. Configure and Build with CMake

- From the `openvdb` directory, create a build directory.
- Run CMake to configure the build. You must provide the path to the `vcpkg.cmake` toolchain file. Replace `[PATH_TO_VCPKG]` with the actual path to your `vcpkg` installation (e.g., `C:\dev\vcpkg`).

```bash
mkdir build
cd build

# Configure the build with CMake
# Note the three arguments: Toolchain file, VCPKG triplet, and setting the architecture
cmake .. -DCMAKE_TOOLCHAIN_FILE=[PATH_TO_VCPKG]\scripts\buildsystems\vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows -A x64

# Build the project using CMake
# This will compile OpenVDB. It can take a while.
cmake --build . --config Release --parallel
```

### 3.3. Set Environment Variable

For the Python bindings (`pyopenvdb`) to find the compiled OpenVDB libraries, you must set the `OPENVDB_LOCATION` environment variable.

- The path should point to the `install` directory that CMake creates within your build folder (e.g., `C:\dev\openvdb\build\install\Release`).
- You can set this variable through the "Edit the system environment variables" control panel in Windows.
- **Crucially, you must restart your terminal or IDE for this new environment variable to be recognized.**

## 4. Set Up Python Environment

### 4.1. Install Python

- If you don't have it, install **Python 3.10 or newer** from the [official Python website](https://www.python.org/downloads/windows/).
- During installation, make sure to check the box that says **"Add Python to PATH"**.

### 4.2. Create Virtual Environment and Install Packages

- Navigate to the root of this project repository.
- Create and activate a Python virtual environment.

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

- With the virtual environment active and the `OPENVDB_LOCATION` environment variable set, install the required packages.

```bash
pip install -r requirements.txt
```

## 5. Verification

To ensure everything is installed correctly, run the following Python command. If it executes without any errors, your environment is set up correctly.

```bash
python -c "import pyopenvdb; print('pyopenvdb successfully imported!')"
```

You are now ready to run and develop the application.
