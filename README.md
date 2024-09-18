# LLM-Syllabi-Evaluator-RAG-

To run the project follow these steps below:

### 1. Conda Environment

Create a new Conda environment named `NAME_ENVIRONMENT` with Python 3.10. Open your terminal or command prompt and run:

```bash
conda create -n NAME_ENVIRONMENT python=3.10
```
Then activate the Conda environment:

```bash
conda activate NAME_ENVIRONMENT
```

### 2. Install Required Python Packages

Install required Python packages listed in the file `packages.txt`:

```bash
pip install -r packages.txt
```

### 3. Ensure Prerequisites are Installed

Ensure the following prerequisites are installed on your system:

- **Make:** Required for building some of the packages. The installation method depends on your operating system:
  - **Linux:** You can typically install `make` via your package manager, for example, `sudo apt-get install make` on Debian/Ubuntu.
  - **macOS:** `make` can be installed with Homebrew using `brew install make`.
  - **Windows:** Consider using a package manager like Chocolatey (`choco install make`) or installing Make for Windows.
- **NVIDIA CUDA Toolkit 11.8:** Required for GPU support and compiling CUDA packages. Installation methods vary by operating system:
  - **Linux:** Follow the [official Linux installation guide](https://developer.nvidia.com/cuda-downloads?target_os=Linux) provided by NVIDIA, selecting the appropriate version and distribution.
  - **macOS:** CUDA support for macOS is limited and may not be available for newer versions. Check the [CUDA Toolkit archive](https://developer.nvidia.com/nvidia-cuda-toolkit-developer-tools-mac-hosts) for compatibility.
  - **Windows:** Use the [official Windows installation guide](https://developer.nvidia.com/cuda-downloads?target_os=Windows) to download and install the toolkit suitable for your system.
