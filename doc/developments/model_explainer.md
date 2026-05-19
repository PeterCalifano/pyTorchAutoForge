# Model Explainer Implementation Plan

Status date: 2026-05-17

Scope: feature-level model explanation in PTAF for PyTorch models, first for tabular/vector inputs and then image/CNN workflows. Main public entry point remains `ModelExplainerHelper`, with typed configuration and result containers added around the existing helper-style API.

## Current Status

- [x] Existing prototype reviewed for integration risks.
- [x] Heavy optional backend imports moved out of module import path.
- [x] `pyTorchAutoForge.evaluation` namespace converted to lazy exports.
- [x] Captum call contract fixed for target selection.
- [x] Basic SHAP tabular path stabilized with deterministic background selection.
- [x] Focused regression tests added for import laziness, Captum call shape, SHAP background selection, and result serialization.
- [x] Real Captum and SHAP tests added in `autoforge`.
- [x] Full default test suite passes in `autoforge`.

## Stage 1 - API Contract And Integration Baseline

- [x] Keep `ModelExplainerHelper` as backward-compatible public entry point.
- [x] Keep `explain_features()` return shape compatible with old dict consumers.
- [x] Add typed result objects:
  - [x] `FeatureAttributionStats`
  - [x] `CaptumExplainerResult`
  - [x] `ShapExplainerResult`
- [x] Add `ModelExplainerConfig` for new configuration fields.
- [x] Preserve legacy `features_names` argument.
- [x] Add preferred `feature_names` argument.
- [x] Keep `CaptumExplainMethods` and `ShapExplainMethods` exported.
- [ ] Add deprecation warning for `features_names` after downstream callers are checked.
- [ ] Decide whether `task_type` should remain accepted but unused, or become part of backend selection.
- [ ] Add API reference examples in MkDocs page.

## Stage 2 - Import Safety And Optional Dependencies

- [x] Remove top-level `captum`, `shap`, `torch`, `seaborn`, and `matplotlib.pyplot` imports from `ModelExplainer.py`.
- [x] Load `torch` only when helper instance is created.
- [x] Load `captum.attr` only when Captum backend is selected.
- [x] Load `shap` only when SHAP backend is selected.
- [x] Load plot packages only when plotting is requested.
- [x] Make `pyTorchAutoForge.evaluation.__init__` lazy.
- [x] Add import-safety regression tests.
- [ ] Split backend dependency errors by backend:
  - [ ] Captum missing should not mention SHAP.
  - [ ] SHAP missing should not mention Captum.
- [ ] Consider moving `shap` and `captum` to optional extras if package install weight becomes too high.

## Stage 3 - Captum Backend

- [x] Support `IntegratedGradients`.
- [x] Support `Saliency`.
- [x] Support `GradientShap`.
- [x] Pass target index as `target=...`.
- [x] Return convergence delta when backend supports it.
- [x] Add default zero baseline for `GradientShap`.
- [x] Allow configured baseline for Integrated Gradients / Gradient SHAP.
- [x] Convert tensor attributions to numpy result arrays.
- [x] Compute signed mean, absolute mean, standard deviation, quantiles, and min/max.
- [x] Flatten non-batch feature dimensions for stats.
- [x] Add regression tests using fake Captum backend.
- [x] Run real Captum Integrated Gradients analytic test in `autoforge`.
- [x] Run real Captum Saliency analytic test in `autoforge`.
- [x] Add real Captum unit tests gated by installed dependency marker.
- [ ] Add image/CNN Captum attribution mode:
  - [ ] input validation for `NCHW`.
  - [ ] per-channel aggregation.
  - [ ] pixel/heatmap output.
  - [ ] optional layer attribution.
- [ ] Add support for `LayerIntegratedGradients`.
- [ ] Add support for `GuidedBackprop` or equivalent image-oriented method.
- [ ] Add batch-size control for large attribution jobs.

## Stage 4 - SHAP Backend

- [x] Support tabular/vector SHAP path.
- [x] Validate SHAP input as `(num_samples, num_features)`.
- [x] Use deterministic background sample selection.
- [x] Add configurable background fraction.
- [x] Add configurable background max sample count.
- [x] Forward PyTorch model through numpy wrapper under `torch.no_grad()`.
- [x] Normalize scalar/vector outputs to `(num_samples, num_outputs)`.
- [x] Preserve feature and output names.
- [x] Add fake-SHAP regression test for background selection and output shape.
- [x] Add `shap` to project dependencies because backend is public.
- [x] Install/verify SHAP in active development environment.
- [x] Add real SHAP smoke test for vector-to-vector regression.
- [x] Add real SHAP smoke test for scalar regression.
- [ ] Add classifier-logit vs probability mode decision.
- [ ] Add warning if model is in training mode before explainer switches to eval.
- [ ] Add explicit unsupported error for image SHAP until CNN path is designed.

## Stage 5 - Artifacts And Plots

- [x] Replace pandas/PyTables `.h5` write path with compressed `.npz`.
- [x] Save SHAP values, base values, data, feature names, and output names.
- [x] Return artifact path in result object.
- [x] Save Captum feature-importance plot when `auto_plot=True`.
- [x] Use seaborn for Captum bar plot when available.
- [x] Fall back to matplotlib when seaborn is unavailable.
- [x] Fix SHAP multi-output plot bookkeeping so figures from all outputs are retained.
- [x] Close figures by default after saving.
- [ ] Add configurable artifact format enum:
  - [ ] `NPZ`
  - [ ] `HDF5`
  - [ ] `CSV_SUMMARY`
- [ ] Add summary CSV export for Captum stats.
- [ ] Add stable file naming scheme for repeated runs.
- [ ] Add optional MLflow artifact logging hook.
- [ ] Add plot tests that verify files exist without checking visual content.

## Stage 6 - Tests And CI Coverage

- [x] Restore `tests/evaluation/test_ModelExplainer.py` from empty placeholder into real tests.
- [x] Test lazy import of explainer backend dependencies.
- [x] Test lazy import of evaluation namespace.
- [x] Test feature-stat flattening.
- [x] Test Captum `target` keyword behavior.
- [x] Test Captum default baseline behavior for Gradient SHAP.
- [x] Test SHAP deterministic background selection.
- [x] Test `.npz` artifact content for SHAP fake backend.
- [x] Test feature-name validation.
- [x] Run targeted evaluation/import safety suite.
- [x] Run full default pytest suite and record unrelated failures.
- [x] Add dependency-marked real Captum tests.
- [x] Add dependency-marked real SHAP tests.
- [ ] Add no-plot mode test for headless CI.
- [ ] Add plot-save smoke test with matplotlib `Agg`.
- [x] Add shape tests for multi-output regression.
- [x] Add shape tests for single-output regression.
- [ ] Add classification target tests.

## Stage 7 - Examples And Documentation

- [ ] Add runnable example for Captum vector regression.
- [ ] Add runnable example for Captum vector classification.
- [ ] Add runnable example for SHAP vector regression.
- [ ] Add example output snippets in docs.
- [ ] Add explainer API page to MkDocs navigation.
- [ ] Add docs section explaining backend dependency requirements.
- [ ] Add docs section explaining artifact outputs.
- [ ] Add docs section explaining known limitations:
  - [ ] SHAP currently tabular/vector only.
  - [ ] image/CNN attribution planned but not complete.
  - [ ] results are explanation artifacts, not ONNX-exportable runtime code.

## Stage 8 - Advanced Roadmap

- [ ] Add image/CNN attribution support.
- [ ] Add layer-wise Captum explainers.
- [ ] Add segmentation explanation workflow.
- [ ] Add multi-input model support.
- [ ] Add model wrapper support for PTAF runtime APIs.
- [ ] Add MLflow integration for explainer artifacts.
- [ ] Add explainer report generator.
- [ ] Add dataset sampling utility for explanation cohorts.
- [ ] Add feature-group attribution support.
- [ ] Add uncertainty-aware attribution summaries for dropout ensembles.

## Validation Ledger

- [x] `python3 -m py_compile pyTorchAutoForge/evaluation/ModelExplainer.py pyTorchAutoForge/evaluation/__init__.py tests/evaluation/test_ModelExplainer.py`
- [x] `conda run -n autoforge pytest -q tests/evaluation/test_ModelExplainer.py`
- [x] `conda run -n autoforge pytest -q tests/api/test_import_safety.py tests/evaluation/test_ResultsPlotter.py tests/evaluation/test_ModelProfiler.py tests/evaluation/test_ModelExplainer.py`
- [x] Real Captum Integrated Gradients analytic test in `autoforge`.
- [x] Real Captum Saliency analytic test in `autoforge`.
- [x] Real SHAP scalar-regression additivity test in `autoforge`.
- [x] Real SHAP vector-regression additivity test in `autoforge`.
- [x] `conda run -n autoforge pytest -q`

## Known Open Issues

- [ ] `pyproject.toml` now lists `shap`; lock/install scripts may need matching update if this repo treats them as source-of-truth.
- [ ] Current image/CNN explainer tasks remain planned, not implemented.
