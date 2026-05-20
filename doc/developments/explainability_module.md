# PTAF Explainability Module Implementation Plan

Scope: replace the current `ModelExplainer` prototype with a typed, modular explainability subsystem. Current implementation target is CPU-friendly Torch tabular models with SHAP and Captum adapters. Future targets include ONNX Runtime, TensorRT, optimized/quantized engines, recurrent models, spiking models, image inputs, sequence inputs, event inputs, and optional custom C++/CUDA acceleration.

## Stage 0 - Repository, Docs, External Usage Inventory

- [x] Treat `doc/` as canonical documentation folder.
- [x] Confirm `doc/` is the canonical documentation source folder.
- [x] Confirm active documentation config lives in `.github/workflows/docs_pages.yml` and `doc/conf.py`.
- [x] Create this implementation plan at `doc/developments/explainability_module.md`.
- [x] Keep old `doc/developments/model_explainer.md` as historical design context until final cleanup decision.
- [x] Search PTAF for old explainer API references.
- [x] Search local workspace for `mlgears`; no matching checkout was found under checked workspace paths, so no external upgrade can be applied in this implementation pass.
- [x] Re-run API usage search after implementation and before final validation.

## Stage 1 - Typed Core Package

- [x] Create `pyTorchAutoForge/evaluation/explainability/`.
- [x] Add `schemas.py` with:
  - [x] `ExplainerTaskType`
  - [x] `ExplainerBackendType`
  - [x] `ExplainerMethod`
  - [x] `ExplainerInputSpec`
  - [x] `ExplainerOutputSpec`
- [x] Add `method_configs.py` with typed method config dataclasses:
  - [x] `ExplainerBaselineConfig`
  - [x] `ShapExplainerConfig`
  - [x] `CaptumIntegratedGradientsConfig`
  - [x] `CaptumSaliencyConfig`
  - [x] `CaptumGradientShapConfig`
- [x] Add validation for all config objects before backend library calls.
- [x] Add `resolve_method_name` and avoid all free-form `method_kwargs` style APIs.
- [x] Use names that explicitly belong to explainer subsystem and avoid ambiguous public names like `InputSpec`, `OutputSpec`, `TargetSpec`, `BaselineConfig`, and `PredictorProtocol`.

Validation gate:

- [x] `conda run -n autoforge pytest -q tests/evaluation/explainability/test_method_configs.py`

## Stage 2 - Request, Target, Result Objects

- [x] Add `targets.py` with `ExplainerTargetSpec`.
- [x] Add `configs.py` with `ExplainerConfig` and `ExplainerRequest`.
- [x] Add `results.py` with `ExplainerResult`.
- [x] Keep constructors side-effect-light.
- [x] Create output folders only in `ExplainerResult.save`.
- [x] Store arrays in `.npz` and metadata in `.json`.
- [x] Do not serialize native SHAP/Captum raw objects by default.
- [x] Keep `raw` available in memory for native backend objects.

Validation gate:

- [x] `conda run -n autoforge pytest -q tests/evaluation/explainability/test_target_spec.py tests/evaluation/explainability/test_explanation_result.py`

## Stage 3 - Predictors

- [x] Add `predictors.py` with:
  - [x] `ExplainerPredictorProtocol`
  - [x] `TorchExplainerPredictor`
  - [x] `CallableExplainerPredictor`
- [x] Support tensor, tuple/list tensor, and dict tensor inputs for Torch prediction.
- [x] Preserve model training/eval state around prediction.
- [x] Provide `predict_with_grad` for gradient explainers.
- [x] Add TODO for runtime-backed predictors wrapping `ModelRuntimeApi` for future ONNX Runtime and TensorRT black-box explainers.

Validation gate:

- [x] `conda run -n autoforge pytest -q tests/evaluation/explainability/test_torch_predictor.py`

## Stage 4 - Registry And Dispatch

- [x] Add `registry.py` with:
  - [x] `ExplainerAdapterProtocol`
  - [x] `ExplainerRegistry`
  - [x] `DEFAULT_EXPLAINER_REGISTRY`
  - [x] `register_default_explainers`
- [x] Add `explainer.py` with `ModelExplainer.explain`.
- [x] Resolve method from typed method config, not from arbitrary string options.
- [x] Instantiate adapters lazily through registry.
- [x] Leave `auto_plot` as no-op metadata warning for now.
- [x] Save outputs through `ExplainerResult.save` only when `save_outputs=True`.

Validation gate:

- [x] `conda run -n autoforge pytest -q tests/evaluation/explainability/test_explainer_registry.py`

## Stage 5 - SHAP Adapter

- [x] Add `adapters/shap_adapter.py`.
- [x] Lazy import `shap` inside adapter methods.
- [x] Support tabular Torch tensor, NumPy array, and pandas DataFrame inputs.
- [x] Infer DataFrame feature names when explicit `ExplainerInputSpec.feature_names` is absent.
- [x] Use explicit background data when supplied.
- [x] Sample background from inputs when missing, capped by `max_background_samples`.
- [x] Use `TorchExplainerPredictor` or callable predictor.
- [x] Apply `ExplainerTargetSpec` when target selection is simple and clear.
- [x] Return normalized `ExplainerResult`.
- [x] Keep native SHAP explanation in `raw["shap_explanation"]`.
- [x] Metadata includes method, backend, algorithm, link, background size, seed.

Validation gate:

- [x] `conda run -n autoforge pytest -q tests/evaluation/explainability/test_shap_adapter.py`

## Stage 6 - Captum Adapter

- [x] Add `adapters/captum_adapter.py`.
- [x] Lazy import `captum` inside adapter methods.
- [x] Support simple Torch tensor inputs for first implementation.
- [x] Implement:
  - [x] Integrated Gradients
  - [x] Saliency
  - [x] GradientShap
- [x] Implement typed baseline strategies:
  - [x] zero
  - [x] mean
  - [x] constant
  - [x] tensor
  - [x] background
- [x] Use `ExplainerTargetSpec.target_index` where Captum supports it.
- [x] Preserve convergence deltas only for methods that return them.
- [x] Return normalized `ExplainerResult`.

Validation gate:

- [x] `conda run -n autoforge pytest -q tests/evaluation/explainability/test_captum_adapter.py`

## Stage 7 - Exports, Facade, Dependency Metadata

- [x] Update `pyTorchAutoForge/evaluation/explainability/__init__.py`.
- [x] Update `pyTorchAutoForge/evaluation/__init__.py`.
- [x] Replace `pyTorchAutoForge/evaluation/ModelExplainer.py` with lightweight re-export facade.
- [x] Stop exporting old `ModelExplainerHelper`, `CaptumExplainMethods`, and `ShapExplainMethods`.
- [x] Add optional extras in `pyproject.toml`:
  - [x] `explain-shap`
  - [x] `explain-captum`
  - [x] `explain`
- [x] Ensure package import has no import-time dependency on SHAP, Captum, or TensorRT.

Validation gate:

- [x] `conda run -n autoforge pytest -q tests/api/test_import_safety.py tests/evaluation/explainability`

## Stage 8 - Runnable CPU Example

- [x] Add `examples/example_ModelExplainer.py`.
- [x] Use deterministic `DummyTestModel`.
- [x] Run SHAP block only when SHAP is installed.
- [x] Run Captum Integrated Gradients block only when Captum is installed.
- [x] Print result shapes and metadata.
- [x] Keep example CPU-only and short.

Validation gate:

- [x] `conda run -n autoforge python examples/example_ModelExplainer.py`

## Stage 9 - Performance Work For Later Stages

- [ ] Add timing metadata hooks around adapter execution once functional API stabilizes.
- [ ] Profile SHAP perturbation cost by sample count, feature count, and background size.
- [ ] Profile Captum gradient methods by input size, target count, and integration step count.
- [ ] Add batch-size tuning for SHAP model wrapper calls.
- [ ] Add vectorized target selection for multi-output Torch predictions.
- [ ] Add runtime-backed black-box perturbation adapter that wraps `ModelRuntimeApi`.
- [ ] Add ONNX Runtime predictor wrapper before any ONNX-specific attribution claims.
- [ ] Add TensorRT predictor wrapper before any TensorRT-specific attribution claims.
- [ ] Prototype custom C++/CUDA direct kernels only after profiling identifies a hot path that Python/Captum/SHAP cannot handle.
- [ ] Candidate custom acceleration areas:
  - [ ] baseline tensor generation
  - [ ] perturbation mask application
  - [ ] batched feature ablations
  - [ ] aggregation of attribution statistics
  - [ ] event/sequence temporal window selection
- [ ] Keep C++/CUDA extensions optional with CPU fallback and parity tests.

## Stage 10 - Final Validation

- [x] `conda run -n autoforge pytest -q tests/evaluation/explainability`
- [x] `conda run -n autoforge python examples/example_ModelExplainer.py`
- [x] `conda run -n autoforge pytest -q tests/api/test_import_safety.py`
- [x] `conda run -n autoforge pytest -q tests/api/onnx/test_ModelHandlerONNx.py tests/api/tensorrt/test_TRTengineExporter.py`
- [x] Confirm no public `method_kwargs` field exists.
- [x] Confirm no arbitrary method option dictionary exists in public explainer API.
- [x] Confirm method options are typed dataclasses.
- [x] Confirm invalid options fail in validation.
- [x] Confirm old ONNX/TensorRT runtime behavior is not changed.

## Non-Goals In First Implementation

- [ ] Full image explainability.
- [ ] Full sequence explainability.
- [ ] Full event explainability.
- [ ] Full recurrent state policy.
- [ ] Full spiking attribution.
- [ ] Full ONNX Runtime attribution.
- [ ] Full TensorRT attribution.
- [ ] Plot/report generation.
- [ ] MLflow logging.
- [ ] Native SHAP/Captum raw object serialization.
- [ ] Refactors to unrelated modules.
