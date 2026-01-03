# pyTorchAutoForge
**_WARNING: work in progress. Do not hesitate to open issues for improvements or problems!_**

A library based on PyTorch (<https://pytorch.org/>) and designed to automate ML models development, tracking and deployment, integrated with MLflow and Optuna (<https://mlflow.org/>, <https://optuna.org/>). It also supports spiking networks libraries (WIP). Model optimization and deployment can be performed using ONNx, pyTorch facilities or TensorRT (WIP). The library aims to be compatible with Jetson Orin Nano Jetpack rev6.1. Several other functionalities and utilities for sklearn and pySR (<https://github.com/MilesCranmer/PySR>) are included (see README and documentation).

## Installation using pip

The suggested installation method is through pip as the others are mostly intended for development and may not be completely up-to-date with the newest release versions. 
In whatever conda or virtual environment you like (preferably with a sufficiently new torch release, to install from pypi:

```bash
pip install pyTorchAutoForge
```

Or from a local copy of the repository (requires `hatch` module for the build):

```bash
cd pyTorchAutoforge
pip install .
```

An automatic installation script `conda_install.sh` is provided and should work in most cases. Note that it will automatically create a new environment named **autoforge** and makes several assumptions about your environment.
Dependencies for the core modules should be installed automatically using pip. However, this is currently not fully tested. Please open related issues.
