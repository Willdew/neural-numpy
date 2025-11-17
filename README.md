# Introduction

Repository for group 26 implementation of a neural network using numpy. Extremely inefficient and in no way practical as it's CPU based.

# Installation

This project uses `uv` for dependency management, a bit like using pip, but automatically handles dependencies, venv and such. Follow the instructions to installing uv:

### 1\. Install uv

`I have no idea if this works as i don't use windows, but these are the official instructions`
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
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

# Running

To run main.py, simply use

```bash
uv run
```

If you want to run a specific script, use

```bash
uv run whateveryourscriptiscalled.py
```

# Adding Packages

If you want to add a package, for example the superior sorting algorithm we would use:

```bash
uv add stalin-sort
```
