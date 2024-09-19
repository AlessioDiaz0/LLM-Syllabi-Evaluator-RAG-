# LLM-Syllabi-Evaluator-RAG-

To run the project follow these steps below:

### 1. Conda Environment

-To download Conda follow use the [Conda Website](https://conda.io/projects/conda/en/latest/user-guide/install/index.html), and ensure to initialize the program using `conda init`

Create a new Conda environment named `NAME_ENVIRONMENT` with Python 3.10. Open your terminal or command prompt and run:

```bash
conda create -n NAME_ENVIRONMENT python=3.10
```

Then activate the Conda environment:

```bash
conda activate NAME_ENVIRONMENT
```

NOTE: To leave the enviroment use the command `conda deactivate`

### 2. Install Python Packages

Install required Python packages listed in `packages.txt` in your new environment:

```bash
pip install -r packages.txt
```

NOTE: It is important to ensure all the version are kept to the original versions, otherwise problems with mismatched tensors may arise.

### 3. Install `Ollama`

To install Ollama follow the instructions in this repository [Git Ollama](https://github.com/ollama/ollama) or [Ollama Download](https://ollama.com/download).

Alternatively follow the instructions in this link: https://www.gpu-mart.com/blog/how-to-run-llama-3-1-8b-with-ollama.

---

### NVIDIA GPUs ONLY

### 1. Ensure Prerequisites are Installed

Ensure the following prerequisites are installed on your system:

- **NVIDIA CUDA Toolkit 11.8:** Required for GPU support and compiling CUDA packages. Installation methods vary by operating system:
  - **Linux:** Follow the [official Linux installation guide](https://developer.nvidia.com/cuda-downloads?target_os=Linux) provided by NVIDIA, selecting the appropriate version and distribution.
  - **macOS:** CUDA support for macOS is limited and may not be available for newer versions. Check the [CUDA Toolkit archive](https://developer.nvidia.com/nvidia-cuda-toolkit-developer-tools-mac-hosts) for compatibility.
  - **Windows:** Use the [official Windows installation guide](https://developer.nvidia.com/cuda-downloads?target_os=Windows) to download and install the toolkit suitable for your system.

### 2. Install PyTorch with CUDA

It is recommended to install PyTorch with CUDA to utilize GPU acceleration for efficiency. Install [PyTorch](https://pytorch.org/get-started/locally/) by configuring your download to your machine and running the command given by the GUI under the 'Start Locally section'.

---

### Test Run

To ensure that everything was properly installed follow these steps

1. **Delete** the current ChromaDB directory, then ensure that the sample_guideline.pdf & sample_syllabus.pdf files exist in the `documents` folder.

2. **Run** the `document_ingestion.py` file. A new ChromaDB directory with all the documents will be created and it will be ready to be used by the RAG pipeline.
 ```bash
 python document_ingestion.py
 ```

 3. **Run** the `rag_pipeline.py` file. The LLM will generate an output which will be saved under a JSON file in the `documents\generated_output` directory
 ```bash
 python rag_pipeline.py
 ```

---

### First Run Instructions

To run the LLM with your faculty syllabus and university syllabus guideline/requirements

1. Delete any ChromaDB directory that was previously constructed using different documents,

2. Place your files in the corresponding  `sample_guideline` and `sample_syllabus` directories

3. **Run** the `document_ingestion.py` file. A new ChromaDB directory with all the documents will be created and it will be ready to be used by the RAG pipeline.
 ```bash
 python document_ingestion.py
 ```

4. **Run** the `rag_pipeline.py` file. The LLM will generate an output which will be saved under a JSON file in the `documents\generated_output` directory
 ```bash
 python rag_pipeline.py
 ```

---

### Produce More Than one Response

- To generate a new response with the LLM Syllabi Evaluator simply run the `rag_pipeline.py` again. This will generate a new response using the documents previously ingested