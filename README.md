# pyTorchAutoForge

**_WARNING: work in progress. Do not hesitate to open issues for improvements or problems!_**

A library based on PyTorch (<https://pytorch.org/>) and designed to automate ML models development, tracking and deployment, integrated with MLflow and Optuna (<https://mlflow.org/>, <https://optuna.org/>). It also supports spiking networks libraries (WIP). Model optimization and deployment can be performed using ONNx, pyTorch facilities or TensorRT (WIP). The library aims to be compatible with Jetson Orin Nano Jetpack rev6.1. Several other functionalities and utilities for sklearn and pySR (<https://github.com/MilesCranmer/PySR>) are included (see README and documentation).

## Documentation

Documentation is built with Sphinx, the PyData theme, and auto-generated public API pages. It is published through GitHub Pages:
<https://petercalifano.github.io/pyTorchAutoForge/>.

Local preview:

```bash
python -m pip install -e ".[docs]"
doc/makedoc.sh -a
```

Local strict build:

```bash
doc/makedoc.sh
```

## Some brief usage guides (WIP)

### TensorRT exporter quick usage

```python
from pyTorchAutoForge.api.tensorrt import TRTengineExporter

exporter = TRTengineExporter()
engine_path = exporter.build_engine_from_onnx_path(
    onnx_model_path="/tmp/model.onnx",
    output_engine_path="/tmp/model.engine",
)
print(engine_path)
```

Notes:

- `TRTEXEC` mode requires `trtexec` in `PATH`.
- `PYTHON` mode requires the `tensorrt` Python package.
- Default behavior avoids architecture-specific flags and is suitable for Jetson deployment workflows.

## Installation Using Pip

The package is available on PyPI. In any conda or virtual environment with a suitable PyTorch release:

```bash
python -m pip install pyTorchAutoForge
```

From a local checkout:

```bash
cd pyTorchAutoforge
python -m pip install .
```

An automatic installation script `conda_install.sh` is provided for development installs. By default it uses an existing `autoforge` conda environment and installs only the core package dependencies:

```bash
./conda_install.sh --create-env --editable
```

Optional extras are explicit:

```bash
./conda_install.sh --with-test --with-docs --build-docs
```

Jetson/ARM installs skip x86-only dependencies through package markers. Provide board-specific PyTorch wheels when needed:

```bash
./conda_install.sh --jetson --pytorch-url <wheel-or-url> --torchvision-url <wheel-or-url>
```
