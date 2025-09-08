# Development Environment Setup Guide (Windows 10/11)

This guide details the complete process for setting up a development environment for the Voxel Slice Pre-Processor project from a clean Windows installation.

## Prerequisites

- Windows 10 or 11 (64-bit)
- Administrator access

## Step 1: Install Core Development Tools

We will use the Chocolatey package manager to install most of the required command-line tools.

1.  **Install Chocolatey:**
    Open PowerShell as an **Administrator** and run the following command:
    ```powershell
    Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
    ```
    Close and reopen your PowerShell terminal after installation.

2.  **Install Git, Python, and CMake:**
    In a new **Administrator** PowerShell terminal, run:
    ```powershell
    choco install -y git python --version=3.11 cmake --installargs 'ADD_CMAKE_TO_PATH=System'
    ```
    - This installs Git, Python 3.11, and the latest CMake.
    - The CMake argument ensures it's added to the system PATH for all users.

## Step 2: Install C++ Build Environment

Compiling OpenVDB requires the Visual Studio C++ build toolchain.

1.  **Download Visual Studio Installer:**
    Go to the [Visual Studio Downloads page](https://visualstudio.microsoft.com/downloads/) and download the "Build Tools for Visual Studio".

2.  **Install C++ Workload:**
    Run the installer. In the "Workloads" tab, select **"Desktop development with C++"**. No optional components are needed. Click "Install".

## Step 3: Compile and Install OpenVDB using vcpkg

We use `vcpkg` to manage C++ libraries. It simplifies the process of building OpenVDB and its numerous dependencies (TBB, Blosc, etc.).

1.  **Install vcpkg:**
    In a PowerShell terminal (no administrator needed), navigate to a directory where you want to store libraries (e.g., `C:\dev\`).
    ```powershell
    git clone https://github.com/Microsoft/vcpkg.git
    .\vcpkg\bootstrap-vcpkg.bat
    ```

2.  **Install OpenVDB 12.1.0:**
    This step will take a significant amount of time as `vcpkg` will download, configure, and compile OpenVDB and all its dependencies.
    ```powershell
    .\vcpkg\vcpkg.exe install openvdb[tools]:x64-windows --recurse
    ```
    *Note: We install `openvdb[tools]` to get the `vdb_print` utility, which is useful for debugging.*

3.  **Integrate vcpkg with your user profile:**
    This makes the libraries findable by other build systems like CMake.
    ```powershell
    .\vcpkg\vcpkg.exe integrate install
    ```

## Step 4: Set Up Python Environment and Install Dependencies

1.  **Clone the Project Repository:**
    Clone this repository to your local machine.
    ```powershell
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Create and Activate a Virtual Environment:**
    From the project's root directory:
    ```powershell
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    ```
    You will see `(.venv)` appear at the beginning of your command prompt.

3.  **Install Python Packages:**
    This command installs all necessary Python libraries for the project.
    ```powershell
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` will be created and maintained for this project)*. For now, install them manually:
    ```powershell
    pip install pyside6 numpy opencv-python moderngl numba pytest pyopenvdb
    ```
    **Important:** The `pyopenvdb` installed from `pip` is a pre-compiled binary. For development, you may need to compile it against the specific OpenVDB version you built. If you encounter issues, refer to the advanced guide on building `pyopenvdb` from source.

## Step 5: Verification

To ensure everything is installed correctly, run the following command from your activated virtual environment:

```powershell
python -c "import cv2; import numpy; import PySide6; import moderngl; import openvdb; print('All major libraries imported successfully!')"
```

If this command prints the success message without any errors, your environment is ready.

---
*This document should be updated as project dependencies change.*
