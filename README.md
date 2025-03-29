# pyTorchAutoForge

A library based on PyTorch (<https://pytorch.org/>) and designed to automate ML models development, tracking and deployment, integrated with MLflow and Optuna (<https://mlflow.org/>, <https://optuna.org/>). It also supports spiking networks libraries (WIP). Model optimization and deployment can be performed using ONNx, pyTorch facilities or TensorRT (WIP). The library also aims to be compatible with Jetson Orin Nano Jetpack rev6.1. ASeveral other functionalities and utilities for sklearn and pySR (<https://github.com/MilesCranmer/PySR>) are included (see README and documentation).

## Installation using pip

Run in a conda or virtual environment:

```bash
pip install pyTorchAutoForge
```

Dependencies for the core modules should be installed automatically using pip.

## Manual installation (bash)

1) Clone the repository
2) Create a virtual environment using python >= 3.10 (tested with 3.11), using `python -m venv <your_venv_name>`
3) Activate the virtual environment using `source <your_venv_name>/bin/activate` on Linux
4) Install the requirements using `pip install -r requirements.txt`
5) Install the package using `pip install .` in the root folder of the repository
