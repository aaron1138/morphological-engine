# Environment Setup Guide

This document provides step-by-step instructions for setting up the development environment for this project on a Debian-based Linux distribution (like Ubuntu). The most complex part of the setup is building the OpenVDB library from source with the correct Python and NanoVDB support enabled.

## 1. System Prerequisites

First, you need to install the necessary build tools and libraries. These include a C++17 compliant compiler, CMake, and the core dependencies for OpenVDB.

```bash
# Update package list
sudo apt-get update

# Install build essentials and CMake
sudo apt-get install -y build-essential cmake

# Install OpenVDB dependencies
# Boost, TBB, and Blosc are essential for core functionality.
sudo apt-get install -y libboost-all-dev libtbb-dev libblosc-dev

# Install other optional but recommended dependencies
sudo apt-get install -y zlib1g-dev
```

## 2. Python Environment Setup

We will use a Python virtual environment to keep our project dependencies isolated.

```bash
# 1. Ensure you have python3-venv installed
sudo apt-get install -y python3-venv

# 2. Create a virtual environment in the project root
python3 -m venv .venv

# 3. Activate the virtual environment
source .venv/bin/activate

# 4. Install nanobind, which is required for building the Python bindings
pip install nanobind

# 5. Install the rest of the Python packages from requirements.txt
pip install -r requirements.txt
```
*Note: Every time you start a new terminal session to work on this project, you must reactivate the virtual environment with `source .venv/bin/activate`.*

## 3. Building OpenVDB 12.1.0 from Source

This is the most critical step. We will clone the official OpenVDB repository, check out the correct version tag, and build it using CMake with flags to enable the Python module (`pyopenvdb`) and NanoVDB.

```bash
# 1. Clone the OpenVDB repository
git clone https://github.com/AcademySoftwareFoundation/openvdb.git
cd openvdb

# 2. Check out the 12.1.0 release
git checkout v12.1.0

# 3. Create a build directory
mkdir build
cd build

# 4. Configure the build with CMake.
# This command enables the Python module and NanoVDB support.
# It assumes the dependencies are in standard system paths found by apt.
cmake .. \
    -D CMAKE_INSTALL_PREFIX=../install \
    -D OPENVDB_BUILD_CORE=ON \
    -D OPENVDB_BUILD_PYTHON_MODULE=ON \
    -D OPENVDB_BUILD_NANOVDB=ON \
    -D NANOVDB_BUILD_TOOLS=ON \
    -D USE_NANOVDB=ON

# 5. Compile the source code.
# The -j flag specifies the number of parallel jobs (set it to your number of CPU cores).
make -j$(nproc)

# 6. Install the compiled libraries into the 'install' directory
make install
```

## 4. Post-Installation Verification

After the build and installation are complete, you need to tell your Python interpreter where to find the newly compiled `pyopenvdb` module.

```bash
# The path will be inside the 'install' directory you created in the openvdb folder.
# The exact path might vary slightly based on your Python version.
# From the project root directory (not inside the openvdb directory):
export PYTHONPATH=$(pwd)/openvdb/install/lib/python3.10/site-packages:$PYTHONPATH

# Now, test if you can import the module
python -c "import pyopenvdb; print('pyopenvdb imported successfully!')"
```

If the import is successful, your environment is correctly set up and you are ready to start development. You may want to add the `export PYTHONPATH...` line to your shell's startup script (e.g., `~/.bashrc` or `~/.zshrc`) for convenience.
