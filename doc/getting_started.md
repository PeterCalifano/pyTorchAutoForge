# Getting Started

## Local Development Environment

Recommended local environment name: `autoforge`.

```bash
conda activate autoforge
python -m pip install -e ".[test,docs]"
```

Scripts default to `autoforge`. Override with `CONDA_ENV` or script-specific flags when using a different local environment name.

## Install From Source

```bash
git clone git@github.com:PeterCalifano/pyTorchAutoForge.git
cd pyTorchAutoForge
python -m pip install -e .
```

## TensorRT Export Smoke Example

```python
from pyTorchAutoForge.api.tensorrt import TRTengineExporter

exporter = TRTengineExporter()
engine_path = exporter.build_engine_from_onnx_path(
    onnx_model_path="/tmp/model.onnx",
    output_engine_path="/tmp/model.engine",
)
print(engine_path)
```

`TRTEXEC` mode requires `trtexec` in `PATH`. `PYTHON` mode requires the `tensorrt` Python package.
