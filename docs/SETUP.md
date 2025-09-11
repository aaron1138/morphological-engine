# Environment Setup Guide

This document provides step-by-step instructions for setting up the development environment for this project on a Debian-based Linux distribution (like Ubuntu).

## 1. System Prerequisites

You will need Python 3 and the `venv` module for creating virtual environments.

```bash
# Update package list
sudo apt-get update

# Install python3 and the venv module
sudo apt-get install -y python3 python3-venv
```

## 2. Python Environment Setup

We will use a Python virtual environment to keep our project dependencies isolated.

```bash
# 1. Create a virtual environment in the project root
python3 -m venv .venv

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Install the required Python packages from requirements.txt
pip install -r requirements.txt
```

That's it! The project now uses pure Python dependencies, so no complex compilation is required.

*Note: Every time you start a new terminal session to work on this project, you must reactivate the virtual environment with `source .venv/bin/activate`.*
