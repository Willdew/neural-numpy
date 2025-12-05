# Introduction

Repository for group 26 implementation of a neural network using numpy. Extremely inefficient and in no way practical as it's CPU based.

# Installation

### Clone the project

Clone the project

```
git clone https://github.com/Willdew/neural-numpy.git
cd neural-numpy
```

## uv

This project uses [uv](https://docs.astral.sh/uv/) for dependency management, a bit like using pip, but automatically handles dependencies, venv and such, and is way faster (rust yay). Follow the instructions to installing

### 1\. Install uv

**Windows**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Linux / macOS**
Use your package manager to install uv, or run this curl command:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2\. Install Environment

Run this command to install Python (if needed) and all dependencies:

```bash
uv sync
```

### 3\. Activate

**Windows**

```powershell
.\venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

# Running

To run a python script, simply use

```bash
uv run whateveryourscriptiscalled.py
```

If you want to try to run a single training run on the CIFAR10 dataset you can simply run

```bash
uv run main.py
```

# Jupyter

a jupyter kernel has been included in the venv. Simply type `code .` or `codium .` and select the .venv as the kernel!

# Adding Packages

If you want to add a package, for example the superior sorting algorithm we would use:

```bash
uv add stalin-sort
```
