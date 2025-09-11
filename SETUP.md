# Voxel Slice Pre-Processor: Development Environment Setup

This guide provides detailed instructions for setting up the development environment for the Voxel Slice Pre-Processor project on a clean **Windows 10/11** installation.

## Introduction

The project relies on a standard Python environment and an external command-line tool (`rawgl.exe`). The previous dependency on compiling C++ libraries has been removed, making the setup process much simpler.

---

## Part 1: Core Development Environment

These are the essential tools required for the project.

### Step 1: Install Python
The project requires Python 3.10 or newer.

1.  Download the **Python 3.10** (or later) installer from the [official Python website](https://www.python.org/downloads/windows/).
2.  Run the installer.
3.  **Important:** On the first page of the installer, check the box that says **"Add Python to PATH"**.
4.  Complete the installation with the default options.

### Step 2: Install Git
Git is required for version control.

1.  Download and install **Git for Windows** from the [official website](https://git-scm.com/download/win).
2.  Use the default options during installation.

### Step 3: (Optional) Install RawGL
The processing pipeline depends on the external tool `rawgl.exe`.

1.  Download RawGL from its [official repository](https://github.com/ssh4net/RawGL/releases).
2.  Extract the archive.
3.  For the application to find it easily, you should add the `bin` directory containing `rawgl.exe` to your system's **PATH** environment variable.

---

## Part 2: Python Environment Setup

### Step 1: Clone the Project Repository
Clone this project to your local machine:
```bash
git clone <repository_url>
cd <repository_name>
```

### Step 2: Create a Virtual Environment
From the root of the project repository, create and activate a Python virtual environment. This keeps the project's dependencies isolated.

```bash
# Create the virtual environment
python -m venv .venv

# Activate the virtual environment
.venv\Scripts\activate
```

### Step 3: Install Python Packages
Install all required Python packages using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

---

## Part 3: Final Sanity Check

After completing all the steps, run a final check to ensure all components are correctly installed and accessible.

1.  Activate your virtual environment (`.venv\Scripts\activate`).
2.  Open a Python interpreter by typing `python`.
3.  Run the following commands. If no `ImportError` occurs, the setup is successful.
    ```python
    import dask
    import dask_image
    import PySide6
    import numba

    print("Successfully imported all core libraries!")
    ```
