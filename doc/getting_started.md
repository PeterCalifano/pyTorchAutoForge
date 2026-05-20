# Getting Started

This page covers first install, local development setup, and basic validation.

## Install From PyPI

Use this path when you only need the released library:

```bash
python -m pip install pyTorchAutoForge
```

Optional feature groups are explicit:

```bash
python -m pip install "pyTorchAutoForge[explain]"
python -m pip install "pyTorchAutoForge[classical-ml]"
```

`explain` installs SHAP and Captum support. `classical-ml` installs optional XGBoost and PySR wrappers.

## Install From Source

Use this path for local development or when testing unreleased changes:

```bash
git clone git@github.com:PeterCalifano/pyTorchAutoForge.git
cd pyTorchAutoForge
python -m pip install -e .
```

Install development extras:

```bash
python -m pip install -e ".[test,docs]"
```

Install documentation and explainer extras together:

```bash
python -m pip install -e ".[test,docs,explain]"
```

## Conda Development Environment

Recommended local environment name: `autoforge`.

```bash
conda create -n autoforge python=3.12
conda activate autoforge
python -m pip install --upgrade pip
python -m pip install -e ".[test,docs]"
```

Repository scripts default to `autoforge` when no conda environment is active. To use a different active environment:

```bash
conda activate my_ptaf_env
bash doc/makedoc.sh
```

To force a specific environment:

```bash
CONDA_ENV=my_ptaf_env bash doc/makedoc.sh
```

## Verify Install

Run a minimal import check:

```bash
python - <<'PY'
import pyTorchAutoForge

print("pyTorchAutoForge import ok")
PY
```

Expected output:

```text
pyTorchAutoForge import ok
```

Check PyTorch availability separately:

```bash
python - <<'PY'
import torch

print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
PY
```

## Run Tests

Default local test run:

```bash
./run_tests.sh -- -q
```

Use another conda environment:

```bash
./run_tests.sh -e my_ptaf_env -- -q
```

Slow, GPU, and visual tests are opt-in:

```bash
./run_tests.sh -- --run-slow -q
./run_tests.sh -- --run-gpu -q
./run_tests.sh -- --run-visual -q
```

## Build Documentation

Build the Sphinx site:

```bash
bash doc/makedoc.sh
```

Serve locally:

```bash
bash doc/makedoc.sh -a
```

Open:

```text
http://127.0.0.1:8000
```

## Next Steps

- Use the API reference for model-building, datasets, optimization, evaluation, and deployment modules.
- Use examples under `examples/` for runnable workflows.
- Install optional extras only for the backends you need.
