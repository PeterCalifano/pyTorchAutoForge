# API Reference

Public API pages are generated from source with `sphinx-autoapi`.
Google-style docstrings are parsed through `sphinx.ext.napoleon`.

## Main Entry Points

- {mod}`pyTorchAutoForge.model_building`: neural-network blocks, backbones, model assembly, and model mutation utilities.
- {mod}`pyTorchAutoForge.datasets`: dataset containers, labels, image augmentations, and vector error models.
- {mod}`pyTorchAutoForge.optimization`: training-manager utilities and loss functions.
- {mod}`pyTorchAutoForge.evaluation`: model evaluation, plotting, profiling, and explainability entry points.
- {mod}`pyTorchAutoForge.evaluation.explainability`: typed SHAP/Captum explainer subsystem.
- {mod}`pyTorchAutoForge.api`: ONNX, TensorRT, Torch, MATLAB, TCP, runtime, and MLflow integration surfaces.
- {mod}`pyTorchAutoForge.monitoring`: run logging helpers.
- {mod}`pyTorchAutoForge.utils`: device, conversion, timing, and general utility helpers.

## Explainability Shortcuts

- {class}`pyTorchAutoForge.evaluation.explainability.ModelExplainer`
- {class}`pyTorchAutoForge.evaluation.explainability.ExplainerConfig`
- {class}`pyTorchAutoForge.evaluation.explainability.ExplainerRequest`
- {class}`pyTorchAutoForge.evaluation.explainability.ShapExplainerConfig`
- {class}`pyTorchAutoForge.evaluation.explainability.CaptumIntegratedGradientsConfig`
- {class}`pyTorchAutoForge.evaluation.explainability.ExplainerResult`

See [](../model_explainer.md) for runnable usage.

## Generated API Tree

```{toctree}
:maxdepth: 2

module_index
generated/pyTorchAutoForge/index
```
