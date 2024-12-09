# pyTorchAutoForge
Custom library based on raw PyTorch to automate DNN development, tracking and deployment, tightly integrated with MLflow and Optuna. The package also includes functions to export and load models to/from ONNx format, as well as a MATLAB wrapper class for model evaluation.

# Quick installation (bash)
1) Clone the repository
2) Create a virtual environment using python >= 3.10 (tested with 3.11), using `python -m venv <your_venv_name>`
3) Activate the virtual environment using `source <your_venv_name>/bin/activate` on Linux 
4) Install the requirements using `pip install -r requirements.txt`
5) Install the package using `pip install .` in the root folder of the repository
