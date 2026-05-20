# Model Explainer

`pyTorchAutoForge.evaluation.explainability` provides typed explainability
requests and dispatches them to SHAP or Captum adapters.

Install optional explainability dependencies first:

```bash
python -m pip install "pyTorchAutoForge[explain]"
```

## Captum Integrated Gradients

Use Captum methods when you have a differentiable Torch model and tensor inputs.

```python
import torch

from pyTorchAutoForge.evaluation.explainability import (
    CaptumIntegratedGradientsConfig,
    ExplainerBackendType,
    ExplainerConfig,
    ExplainerInputSpec,
    ExplainerOutputSpec,
    ExplainerRequest,
    ExplainerTaskType,
    ExplainerTargetSpec,
    ModelExplainer,
)

model = torch.nn.Sequential(
    torch.nn.Linear(3, 8),
    torch.nn.ReLU(),
    torch.nn.Linear(8, 1),
)

inputs = torch.tensor(
    [
        [0.2, 0.4, 0.8],
        [0.5, 0.1, 0.3],
    ],
    dtype=torch.float32,
)

request = ExplainerRequest(
    model=model,
    inputs=inputs,
    config=ExplainerConfig(
        method_config=CaptumIntegratedGradientsConfig(n_steps=16),
        backend=ExplainerBackendType.TORCH,
        device="cpu",
    ),
    input_spec=ExplainerInputSpec(
        feature_names=("mass", "velocity", "temperature"),
    ),
    output_spec=ExplainerOutputSpec(
        task_type=ExplainerTaskType.REGRESSION,
        output_names=("score",),
    ),
    target_spec=ExplainerTargetSpec(target_index=0),
)

result = ModelExplainer().explain(request)

print(result.values.shape)
print(result.method)
```

Expected output shape:

```text
torch.Size([2, 3])
captum.integrated_gradients
```

## SHAP Tabular Explanation

Use SHAP when you need black-box, tabular explanations. Torch models and Python
callables are supported.

```python
import torch

from pyTorchAutoForge.evaluation.explainability import (
    ExplainerBackendType,
    ExplainerConfig,
    ExplainerInputSpec,
    ExplainerOutputSpec,
    ExplainerRequest,
    ExplainerTaskType,
    ModelExplainer,
    ShapExplainerConfig,
)

model = torch.nn.Sequential(
    torch.nn.Linear(3, 1),
)

inputs = torch.tensor(
    [
        [0.2, 0.4, 0.8],
        [0.5, 0.1, 0.3],
        [0.9, 0.6, 0.2],
    ],
    dtype=torch.float32,
)

request = ExplainerRequest(
    model=model,
    inputs=inputs,
    background_data=inputs,
    config=ExplainerConfig(
        method_config=ShapExplainerConfig(max_background_samples=3),
        backend=ExplainerBackendType.TORCH,
    ),
    input_spec=ExplainerInputSpec(
        feature_names=("mass", "velocity", "temperature"),
    ),
    output_spec=ExplainerOutputSpec(
        task_type=ExplainerTaskType.REGRESSION,
        output_names=("score",),
    ),
)

result = ModelExplainer().explain(request)

print(result.values.shape)
print(result.method)
```

Expected output:

```text
(3, 3)
shap
```

## Saving Results

Set `save_outputs=True` to write portable artifacts:

```python
request.config.save_outputs = True
request.config.output_folder = "explainer_outputs"
result = ModelExplainer().explain(request)

print(result.metadata["saved_artifacts"])
```

Saved files:

- `explanation.npz`: arrays such as values, data, predictions, and names.
- `explanation.json`: method, backend, metadata, and raw object type names.

## Public API

- {class}`pyTorchAutoForge.evaluation.explainability.ModelExplainer`
- {class}`pyTorchAutoForge.evaluation.explainability.ExplainerConfig`
- {class}`pyTorchAutoForge.evaluation.explainability.ExplainerRequest`
- {class}`pyTorchAutoForge.evaluation.explainability.ExplainerInputSpec`
- {class}`pyTorchAutoForge.evaluation.explainability.ExplainerOutputSpec`
- {class}`pyTorchAutoForge.evaluation.explainability.ExplainerTargetSpec`
- {class}`pyTorchAutoForge.evaluation.explainability.ShapExplainerConfig`
- {class}`pyTorchAutoForge.evaluation.explainability.CaptumIntegratedGradientsConfig`
- {class}`pyTorchAutoForge.evaluation.explainability.CaptumSaliencyConfig`
- {class}`pyTorchAutoForge.evaluation.explainability.CaptumGradientShapConfig`
- {class}`pyTorchAutoForge.evaluation.explainability.ExplainerResult`
