import os
import warnings
from datetime import datetime
from typing import Any, Literal

import numpy
import onnx
import torch
from numpy.testing import assert_allclose

from pyTorchAutoForge.model_building.modelBuildingBlocks import AutoForgeModule
from pyTorchAutoForge.utils import numpy_to_torch, timeit_averaged_, torch_to_numpy


class ModelHandlerONNx:
    """Utility class to export, validate, and post-process ONNX models.

    This handler centralizes ONNX export for torch models and exposes one
    canonical API (`export_onnx`) with backend selection. Legacy methods
    (`torch_export`, `torch_dynamo_export`) are kept as compatibility wrappers.

    Notes:
        Export path resolution follows these rules:
            1. If `onnx_export_path` ends with `.onnx`, it is treated as a full
               output file path.
            2. If `onnx_model_name` contains a directory component, it is treated
               as a full output path override.
            3. If `onnx_model_name` is a bare name, it is written into
               `onnx_export_path` (directory mode, backward compatible behavior).
            4. If no model name is provided and directory mode is used, filename is
               auto-generated as `<base>_<YYYYMMDD_HHMM>.onnx`.
        Existing files are overwritten.
    """

    def __init__(self,
                 model: torch.nn.Module | AutoForgeModule | onnx.ModelProto,
                 dummy_input_sample: torch.Tensor | numpy.ndarray | tuple[torch.Tensor, ...],
                 onnx_export_path: str = ".",
                 opset_version: int = 13,
                 run_export_validation: bool = True,
                 generate_report: bool = False,
                 run_onnx_simplify: bool = False,
                 ) -> None:
        """Initialize ONNX handler configuration and model references.

        Args:
            model: Source model, either a torch module or an ONNX proto.
            dummy_input_sample: Default sample used for export and validation when
                no call-time input is provided.
            onnx_export_path: Export directory or explicit `.onnx` file path.
            opset_version: Target ONNX opset for export.
            run_export_validation: Whether to run checker + ORT validation after export.
            generate_report: Whether to enable torch exporter report generation.
            run_onnx_simplify: Whether to simplify the exported ONNX model.

        Raises:
            ValueError: If `model` is not torch or ONNX type.
        """
        self.torch_model: torch.nn.Module | None = None
        self.onnx_model: onnx.ModelProto | None = None

        # Allocate model to the correct attribute based on type
        if isinstance(model, torch.nn.Module):
            self.torch_model = model

        elif isinstance(model, onnx.ModelProto):
            self.onnx_model = model

        else:
            raise ValueError(
                "Model must be of base type torch.nn.Module or onnx.ModelProto")

        # Initialize configuration variables
        self.run_export_validation = run_export_validation
        self.onnx_filepath = ""
        self.dummy_input_sample = dummy_input_sample
        self.onnx_export_path = onnx_export_path
        self.opset_version = opset_version
        self.IO_names = {"input": ["input"], "output": ["output"]}
        self.dynamic_axes = {"input": {0: "batch_size"},
                             "output": {0: "batch_size"}}
        self.generate_report = generate_report
        self.torch_version = torch.__version__
        self.run_onnx_simplify = run_onnx_simplify
        self._runtime_api: Any | None = None

    def _resolve_model_inputs(self,
                              model_inputs: torch.Tensor | numpy.ndarray | tuple[torch.Tensor, ...] | list[torch.Tensor | numpy.ndarray] | None,
                              ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Resolve model forward inputs and normalize tensor-like values.

        Args:
            model_inputs: Optional model inputs supplied at export call time.

        Returns:
            A normalized tensor or tuple of tensors for exporter consumption.

        Raises:
            ValueError: If no model inputs are available from call-time args or defaults.
        """
        resolved_inputs = model_inputs if model_inputs is not None else self.dummy_input_sample
        if resolved_inputs is None:
            raise ValueError(
                "Model inputs must be provided or dummy input sample must be provided when constructing this class."
            )

        if isinstance(resolved_inputs, (tuple, list)):
            return tuple(numpy_to_torch(sample) for sample in resolved_inputs)
        return numpy_to_torch(resolved_inputs)

    def _resolve_additional_export_kwargs(self,
                                          additional_export_kwargs: dict[str, Any] | None,
                                          ) -> dict[str, Any]:
        """Validate optional extra kwargs for `torch.onnx.export`.

        Args:
            additional_export_kwargs: Extra keyword arguments forwarded to
                `torch.onnx.export`.

        Returns:
            A validated kwargs dictionary.

        Raises:
            TypeError: If `additional_export_kwargs` is not a dictionary.
            ValueError: If unsupported/managed keys are provided.
        """
        if additional_export_kwargs is None:
            return {}

        if not isinstance(additional_export_kwargs, dict):
            raise TypeError(
                "additional_export_kwargs must be a dictionary when provided.")

        forbidden_keys = {
            "model",
            "args",
            "f",
            "dynamo",
            "input_names",
            "output_names",
            "dynamic_axes",
            "verbose",
            "report",
            "opset_version",
            "export_params",
            "do_constant_folding",
        }

        forbidden = sorted(
            set(additional_export_kwargs).intersection(forbidden_keys))

        # Prevent user from passing keys that are already exposed as dedicated arguments
        if forbidden:
            raise ValueError(
                f"Unsupported keys in additional_export_kwargs: {forbidden}. "
                "Use dedicated method arguments for these fields."
            )

        return dict(additional_export_kwargs)

    def _resolve_export_target(self, onnx_model_name: str | None = None) -> str:
        """Resolve and store the final ONNX output filepath.

        Args:
            onnx_model_name: Optional name or path override for ONNX output.

        Returns:
            The resolved ONNX output filepath.
        """

        # Helper to ensure a given path ends with .onnx extension, adding it if missing
        def _with_onnx_extension(path_or_name: str) -> str:
            root, ext = os.path.splitext(path_or_name)
            if ext.lower() == ".onnx":
                return path_or_name
            return f"{root if ext else path_or_name}.onnx"

        export_path = str(self.onnx_export_path) if self.onnx_export_path not in (
            None, "") else "."
        export_path_is_file = export_path.lower().endswith(".onnx")

        if onnx_model_name is not None:
            # If a model name is provided, check if it contains a directory component. If it does, treat it as a full path override. If not, write it into the export path directory.
            model_name = str(onnx_model_name)
            model_name_has_dir = os.path.dirname(model_name) != ""

            if model_name_has_dir:
                final_path = _with_onnx_extension(model_name)
            else:
                # If model name is a bare name, write it into the export path directory
                target_dir = os.path.dirname(
                    export_path) if export_path_is_file else export_path
                target_dir = target_dir if target_dir != "" else "."
                final_path = os.path.join(
                    target_dir, _with_onnx_extension(model_name))

        else:
            # If no model name is provided, use export path directly (with extension if it's a file) or generate a timestamped name in the target directory.
            if export_path_is_file:
                final_path = _with_onnx_extension(export_path)
            else:
                target_dir = export_path
                target_dir = target_dir if target_dir != "" else "."
                dir_name = os.path.basename(os.path.normpath(target_dir))
                base_name = "onnx_export" if dir_name in (
                    "", ".", os.path.sep) else dir_name
                timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                final_path = os.path.join(
                    target_dir, f"{base_name}_{timestamp}.onnx")

        # Make sure the parent directory exists before export (if specified)
        parent_dir = os.path.dirname(final_path)
        if parent_dir != "":
            os.makedirs(parent_dir, exist_ok=True)

        self.onnx_filepath = final_path
        return final_path

    def _run_export_shared(self,
                           model_inputs: torch.Tensor | tuple[torch.Tensor, ...],
                           additional_export_kwargs: dict[str, Any],
                           dynamic_axes: dict,
                           IO_names: dict,
                           enable_verbose: bool,
                           use_dynamo: bool,
                           ):
        """Call `torch.onnx.export` using shared arguments for both backends.

        Args:
            model_inputs: Input tensor(s) used for model forward during export.
            additional_export_kwargs: Extra keyword arguments forwarded to
                `torch.onnx.export`.
            dynamic_axes: Dynamic axes mapping for ONNX I/O.
            IO_names: ONNX input/output names.
            enable_verbose: Whether to print verbose exporter logs.
            use_dynamo: Whether to use dynamo export path.

        Returns:
            The return value from `torch.onnx.export` (`ONNXProgram` for dynamo
            exports in supported torch versions, otherwise `None`).

        Raises:
            ValueError: If no torch model is available for export.
        """
        if self.torch_model is None:
            raise ValueError(
                "Torch model is not available. Cannot run ONNX export from ONNX ModelProto."
            )

        # Keep backward-compatible tuple behavior: tuple model inputs are wrapped
        # so they are passed as a single argument to forward().
        model_inputs_for_export = (model_inputs,) if isinstance(
            model_inputs, tuple) else model_inputs

        # Validate additional export kwargs and prevent conflicts with dedicated arguments
        export_kwargs = {
            "export_params": True,
            "opset_version": self.opset_version,
            "do_constant_folding": True,
            "input_names": IO_names["input"],
            "output_names": IO_names["output"],
            "dynamic_axes": dynamic_axes,
            "dynamo": use_dynamo,
            "report": self.generate_report,
            "verbose": enable_verbose,
        }

        export_kwargs.update(additional_export_kwargs)

        # Call torch.onnx.export with the resolved arguments
        return torch.onnx.export(model=self.torch_model,
                                 args=model_inputs_for_export,
                                 f=self.onnx_filepath,
                                 **export_kwargs,
                                 )

    def _run_export_legacy(self,
                           model_inputs: torch.Tensor | tuple[torch.Tensor, ...],
                           additional_export_kwargs: dict[str, Any],
                           dynamic_axes: dict,
                           IO_names: dict,
                           enable_verbose: bool,
                           ) -> None:
        """Export ONNX using TorchScript-based exporter (`dynamo=False`).

        Args:
            model_inputs: Input tensor(s) for tracing/export.
            additional_export_kwargs: Extra keyword arguments forwarded to
                `torch.onnx.export`.
            dynamic_axes: Dynamic axes mapping for ONNX I/O.
            IO_names: Input/output ONNX tensor names.
            enable_verbose: Whether to enable verbose exporter logs.

        Raises:
            ValueError: If no torch model is available for export.
        """

        # Caller for legacy torch_export method (soft-deprecated)
        self._run_export_shared(
            model_inputs=model_inputs,
            additional_export_kwargs=additional_export_kwargs,
            dynamic_axes=dynamic_axes,
            IO_names=IO_names,
            enable_verbose=enable_verbose,
            use_dynamo=False,
        )

    def _run_export_dynamo(self,
                           model_inputs: torch.Tensor | tuple[torch.Tensor, ...],
                           additional_export_kwargs: dict[str, Any],
                           dynamic_axes: dict,
                           IO_names: dict,
                           enable_verbose: bool,
                           ) -> None:
        """Export ONNX using TorchDynamo-based exporter (`dynamo=True`).

        Args:
            model_inputs: Input tensor(s) for tracing/export.
            additional_export_kwargs: Extra keyword arguments forwarded to
                `torch.onnx.export`.
            dynamic_axes: Dynamic axes mapping for ONNX I/O.
            IO_names: Input/output ONNX tensor names.
            enable_verbose: Whether to enable verbose exporter logs.

        Raises:
            ValueError: If no torch model is available for export.
            RuntimeError: If export does not produce an output ONNX file.
        """

        # Get exported onnx program
        onnx_program = self._run_export_shared(model_inputs=model_inputs,
                                               additional_export_kwargs=additional_export_kwargs,
                                               dynamic_axes=dynamic_axes,
                                               IO_names=IO_names,
                                               enable_verbose=enable_verbose,
                                               use_dynamo=True,
                                               )

        # Optimize and save the ONNX program
        if onnx_program is not None and hasattr(onnx_program, "optimize"):
            onnx_program.optimize()

        if onnx_program is not None and hasattr(onnx_program, "save"):
            onnx_program.save(self.onnx_filepath)

        if not os.path.isfile(self.onnx_filepath):
            raise RuntimeError(
                "TorchDynamo ONNX export failed, no model generated.")

    def _post_export_steps(self, test_sample: torch.Tensor | tuple[torch.Tensor, ...]) -> None:
        """Run optional validation and simplification after successful export.

        Args:
            test_sample: Normalized tensor(s) used for ONNX runtime validation.
        """

        # Pre-load model if needed
        if self.run_export_validation or self.run_onnx_simplify:
            if self.onnx_model is None:
                try:
                    self.onnx_model = self.onnx_load(
                        onnx_filepath=self.onnx_filepath)

                except Exception as err:
                    # Exit with failure if validation or simplification is requested but model cannot be loaded
                    raise RuntimeError(
                        "ONNX model loading failed after export. Cannot proceed with validation or simplification: ") from err
            else:
                print(
                    "\033[38;5;208mONNX model already loaded in handler state, skipping reload for post-export steps.\033[0m")

            # Assert model is loaded
            assert self.onnx_model is not None, "ONNX model is not loaded in handler state after export. Cannot proceed with validation or simplification."

            # Run automatic validation
            if self.run_export_validation:

                # Run onnx validate
                self.onnx_validate(onnx_model=self.onnx_model,
                                   test_sample=test_sample)

            # Automatic simplification step
            # TODO add check if onnx-simplify is installed before trying to run it and raise a warning if not available
            if self.run_onnx_simplify:

                self.onnx_filepath, self.onnx_model = self.onnx_simplify(
                    self.onnx_model)

    def _get_runtime_api(self):
        """Return lazy-initialized runtime facade instance."""
        if self._runtime_api is None:
            from pyTorchAutoForge.api.runtime import ModelRuntimeApi

            self._runtime_api = ModelRuntimeApi()
        return self._runtime_api

    def export_onnx(self,
                    model_inputs: torch.Tensor | numpy.ndarray | tuple[
                        torch.Tensor, ...] | list[torch.Tensor] | None = None,
                    input_tensor: torch.Tensor | numpy.ndarray | tuple[
                        torch.Tensor, ...] | list[torch.Tensor] | None = None,
                    additional_export_kwargs: dict[str, Any] | None = None,
                    onnx_model_name: str | None = None,
                    dynamic_axes: dict | None = None,
                    IO_names: dict | None = None,
                    enable_verbose: bool = False,
                    backend: Literal["auto", "dynamo", "legacy"] = "auto",
                    fallback_to_legacy: bool | None = None, ) -> str:
        """Export torch model to ONNX using the selected backend.

        Args:
            model_inputs: Input tensor(s). If `None`, class dummy sample is used.
            input_tensor: Deprecated alias for `model_inputs`.
            additional_export_kwargs: Optional extra kwargs forwarded to
                `torch.onnx.export` (for parameters not already exposed).
            onnx_model_name: Optional output name/path override.
            dynamic_axes: Dynamic axes mapping. Defaults to class configuration.
            IO_names: ONNX input/output names. Defaults to class configuration.
            enable_verbose: Whether to enable verbose exporter logs.
            backend: Export backend selector: `auto`, `dynamo`, or `legacy`.
            fallback_to_legacy: Controls fallback after dynamo failure. If `None`,
                fallback is enabled for `auto` and `dynamo`, disabled for `legacy`.

        Returns:
            The exported ONNX filepath.

        Raises:
            ValueError: If backend selection is invalid.
            RuntimeError: If export fails with no successful fallback.
        """

        if model_inputs is not None and input_tensor is not None:
            raise ValueError(
                "Please provide only one of: model_inputs or input_tensor.")
        if model_inputs is None and input_tensor is not None:
            warnings.warn(
                "input_tensor is deprecated. Use model_inputs instead.",
                FutureWarning,
                stacklevel=2,
            )
            model_inputs = input_tensor

        # Resolve and validate model inputs, additional export kwargs, and export target path before attempting export
        resolved_model_inputs = self._resolve_model_inputs(model_inputs)
        resolved_additional_export_kwargs = self._resolve_additional_export_kwargs(
            additional_export_kwargs)

        self._resolve_export_target(onnx_model_name=onnx_model_name)

        if dynamic_axes is None:
            dynamic_axes = self.dynamic_axes
        if IO_names is None:
            IO_names = self.IO_names

        # Normalize and select backend
        backend = backend.lower()
        if backend not in ("auto", "dynamo", "legacy"):
            raise ValueError(
                "Invalid backend. Please select one of: 'auto', 'dynamo', 'legacy'.")

        if fallback_to_legacy is None:
            fallback_to_legacy = backend in ("auto", "dynamo")

        used_backend = backend

        try:
            # Try export with the selected backend, applying fallback logic if enabled
            if backend == "legacy":
                self._run_export_legacy(
                    resolved_model_inputs,
                    resolved_additional_export_kwargs,
                    dynamic_axes,
                    IO_names,
                    enable_verbose,
                )
                used_backend = "legacy"

            else:
                try:
                    self._run_export_dynamo(
                        resolved_model_inputs,
                        resolved_additional_export_kwargs,
                        dynamic_axes,
                        IO_names,
                        enable_verbose,
                    )
                    used_backend = "dynamo"

                except Exception as dyn_exc:
                    if not fallback_to_legacy:
                        raise

                    warnings.warn(
                        f"TorchDynamo export failed ({dyn_exc}). Falling back to legacy exporter.",
                        RuntimeWarning,
                        stacklevel=2,
                    )

                    self._run_export_legacy(
                        resolved_model_inputs,
                        resolved_additional_export_kwargs,
                        dynamic_axes,
                        IO_names,
                        enable_verbose,
                    )
                    used_backend = "legacy"

        except Exception as export_exc:
            raise RuntimeError(
                f"ONNX export failed using backend '{backend}'.") from export_exc

        print(
            f"\033[92mModel exported to ONNx format: {self.onnx_filepath} (backend: {used_backend})\033[0m")
        self._post_export_steps(test_sample=resolved_model_inputs)
        return self.onnx_filepath

    def torch_export(self,
                     model_inputs: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
                     input_tensor: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
                     additional_export_kwargs: dict[str, Any] | None = None,
                     onnx_model_name: str | None = None,
                     dynamic_axes: dict | None = None,
                     IO_names: dict | None = None,
                     enable_verbose: bool = False,
                     ) -> str:
        """Export ONNX using the legacy backend.

        Deprecated:
            Use `export_onnx(backend="legacy")` instead.

        Args:
            model_inputs: Input tensor(s) for export.
            input_tensor: Deprecated alias for `model_inputs`.
            additional_export_kwargs: Optional extra kwargs forwarded to
                `torch.onnx.export`.
            onnx_model_name: Optional output name/path override.
            dynamic_axes: Dynamic axes mapping.
            IO_names: ONNX input/output names.
            enable_verbose: Whether to enable verbose exporter logs.

        Returns:
            The exported ONNX filepath.
        """
        warnings.warn(
            "torch_export is deprecated and will be removed in a future release. "
            "Use export_onnx(..., backend='legacy') instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self.export_onnx(model_inputs=model_inputs,
                                input_tensor=input_tensor,
                                additional_export_kwargs=additional_export_kwargs,
                                onnx_model_name=onnx_model_name,
                                dynamic_axes=dynamic_axes,
                                IO_names=IO_names,
                                enable_verbose=enable_verbose,
                                backend="legacy",
                                fallback_to_legacy=False,
                                )

    def torch_dynamo_export(self,
                            model_inputs: torch.Tensor | None = None,
                            input_tensor: torch.Tensor | None = None,
                            additional_export_kwargs: dict[str,
                                                           Any] | None = None,
                            onnx_model_name: str = "onnx_dynamo_export",
                            dynamic_axes: dict | None = None,
                            IO_names: dict | None = None,
                            enable_verbose: bool = False,
                            ) -> str:
        """Export ONNX using the dynamo-first backend with legacy fallback.

        Deprecated:
            Use `export_onnx(backend="dynamo")` instead.

        Args:
            model_inputs: Input tensor(s) for export.
            input_tensor: Deprecated alias for `model_inputs`.
            additional_export_kwargs: Optional extra kwargs forwarded to
                `torch.onnx.export`.
            onnx_model_name: Optional output name/path override.
            dynamic_axes: Dynamic axes mapping.
            IO_names: ONNX input/output names.
            enable_verbose: Whether to enable verbose exporter logs.

        Returns:
            The exported ONNX filepath.
        """
        warnings.warn(
            "torch_dynamo_export is deprecated and will be removed in a future release. "
            "Use export_onnx(..., backend='dynamo') instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self.export_onnx(
            model_inputs=model_inputs,
            input_tensor=input_tensor,
            additional_export_kwargs=additional_export_kwargs,
            onnx_model_name=onnx_model_name,
            dynamic_axes=dynamic_axes,
            IO_names=IO_names,
            enable_verbose=enable_verbose,
            backend="dynamo",
            fallback_to_legacy=True,
        )

    def clear_onnx_session_cache(self) -> None:
        """Clear cached ONNX Runtime sessions owned by this handler.

        Returns:
            ``None``.
        """
        runtime_api = self._get_runtime_api()
        runtime_api.clear_session_cache(backend="onnx")

    def onnx_forward(self,
                     model_inputs: (torch.Tensor
                                    | numpy.ndarray
                                    | tuple[torch.Tensor | numpy.ndarray, ...]
                                    | list[torch.Tensor | numpy.ndarray]
                                    | dict[str, torch.Tensor | numpy.ndarray]
                                    | None
                                    ) = None,
                     input_tensor: (torch.Tensor
                                    | numpy.ndarray
                                    | tuple[torch.Tensor | numpy.ndarray, ...]
                                    | list[torch.Tensor | numpy.ndarray]
                                    | None
                                    ) = None,
                     providers: list[str] | tuple[str, ...] | None = None,
                     provider_options: dict[str, dict[str, Any]] | None = None,
                     force_new_session: bool = False,
                     session_options: Any | None = None,
                     ) -> dict[str, numpy.ndarray]:
        """Run ONNX Runtime inference through a cached runtime helper.

        Args:
            model_inputs: Runtime model inputs in tensor, tuple/list, or dict form.
            input_tensor: Deprecated alias for `model_inputs`.
            providers: Ordered execution providers preference list.
            provider_options: Optional provider-specific options.
            force_new_session: Whether to bypass and refresh session cache.
            session_options: Optional ORT session options object.

        Returns:
            ONNX output dictionary keyed by graph output names.

        Raises:
            ValueError: If both `model_inputs` and `input_tensor` are provided,
                or if no ONNX source is available.
            TypeError: If provider options or input value types are invalid.

        Notes:
            ONNX source selection order is:
            1. `self.onnx_model` (in-memory model).
            2. `self.onnx_filepath` (existing model file on disk).
            Provider selection is delegated to `ModelRuntimeApi` and
            `OnnxRuntimeApi`, which keep requested provider order, skip
            unavailable providers with warning, and raise only when no provider
            remains usable.
        """
        if model_inputs is not None and input_tensor is not None:
            raise ValueError(
                "Please provide only one of: model_inputs or input_tensor.")

        if model_inputs is None and input_tensor is not None:
            warnings.warn(
                "input_tensor is deprecated. Use model_inputs instead.",
                FutureWarning,
                stacklevel=2,
            )
            model_inputs = input_tensor

        if model_inputs is None and self.dummy_input_sample is not None:
            model_inputs = self.dummy_input_sample

        # Get runtime api model
        runtime_api = self._get_runtime_api()

        # Forward request
        return runtime_api.onnx_forward(model_inputs=model_inputs,
                                        providers=providers,
                                        provider_options=provider_options,
                                        force_new_session=force_new_session,
                                        session_options=session_options,
                                        onnx_model=self.onnx_model,
                                        onnx_filepath=self.onnx_filepath if self.onnx_filepath != "" else None,
                                        )

    # Conversion utility to convert ONNX model to a different opset version, with error handling
    def convert_to_onnx_opset(self,
                              onnx_model: onnx.ModelProto | None = None,
                              onnx_opset_version: int | None = None,
                              ) -> onnx.ModelProto | None:
        """Convert an ONNX model to a target opset version.

        Args:
            onnx_model: ONNX model to convert. Uses stored model when `None`.
            onnx_opset_version: Target opset version. Uses class default when `None`.

        Returns:
            Converted ONNX model proto, or `None` if conversion fails.

        Raises:
            ValueError: If no ONNX model is available.
        """
        if onnx_opset_version is None:
            onnx_opset_version = self.opset_version

        if onnx_model is None and self.onnx_model is None:
            raise ValueError(
                "No ONNx model provided for conversion and no model stored in onnx_model attribute.")
        if onnx_model is None:
            onnx_model = self.onnx_model

        try:
            return onnx.version_converter.convert_version(model=onnx_model, target_version=onnx_opset_version)
        except Exception as conversion_error:
            print(
                f"Error converting model to opset version {onnx_opset_version}: {conversion_error}")
            return None

    def onnx_validate(self,
                      onnx_model: onnx.ModelProto | str,
                      test_sample: torch.Tensor | numpy.ndarray | tuple[torch.Tensor, ...] | None = None,
                      output_sample: torch.Tensor | numpy.ndarray | None = None,
                      ) -> None:
        """Validate an ONNX model with structural and optional inference checks.

        Args:
            onnx_model: ONNX model proto or file path.
            test_sample: Optional sample used to run ONNX Runtime inference.
            output_sample: Optional reference output used for numerical comparison.

        Raises:
            FileNotFoundError: If `onnx_model` path does not exist.
            TypeError: If `onnx_model` has unsupported type.
        """
        if isinstance(onnx_model, str):
            if not os.path.isfile(onnx_model):
                raise FileNotFoundError(
                    f"ONNX model file not found: {onnx_model}")
            onnx_model = onnx.load(onnx_model)
        elif not isinstance(onnx_model, onnx.ModelProto):
            raise TypeError(f"Invalid ONNX model class: {type(onnx_model)}")

        print(
            "\033[94mValidating model using ONNx checker.check_model... \033[0m", end=" ")
        onnx.checker.check_model(onnx_model, full_check=True)
        print("\033[92mPASSED.\033[0m")

        if test_sample is not None:
            # Use validated model as runtime source for inference checks
            self.onnx_model = onnx_model

            print(
                "\033[94mValidating model inference using onnxruntime...\033[0m", end=" ")
            ort_outs_map = self.onnx_forward(model_inputs=test_sample,
                                             providers=[
                                                 "CPUExecutionProvider"],
                                             force_new_session=True,
                                             )
            print("\033[92mPASSED.\033[0m")

            if output_sample is not None:
                print(
                    "\033[94mOutput equivalence test. Using tolerances rtol=1e-03 and atol=1e-06...\033[0m",
                    end=" ",
                )
                ort_first_output = next(iter(ort_outs_map.values()))
                assert_allclose(torch_to_numpy(output_sample),
                                ort_first_output, rtol=1e-03, atol=1e-06)
                print("\033[92mPASSED.\033[0m")
            else:
                print(
                    "\033[38;5;208mNo output sample provided for ONNX model validation. "
                    "Result validation test skipped.\033[0m"
                )
        else:
            print(
                "\033[38;5;208mNo test sample provided for ONNX model validation. "
                "Equivalence test vs torch model skipped.\033[0m"
            )

    def onnx_compare_timing(self,
                            torch_model: torch.nn.Module,
                            onnx_model: onnx.ModelProto,
                            test_sample: torch.Tensor | numpy.ndarray,
                            num_iterations: int = 100,
                            ) -> dict:
        """Compare average inference runtime between torch and ONNX Runtime.

        Args:
            torch_model: Torch model to benchmark.
            onnx_model: ONNX model to benchmark.
            test_sample: Sample used as input for both runtimes.
            num_iterations: Number of iterations used for average timings.

        Returns:
            Dictionary with average torch and ONNX inference times.
        """
        torch_model.to("cpu")
        self.onnx_model = onnx_model

        # TODO add multiple device selection if available

        # Utility function to perform forward using onnxruntime and measure time
        def ort_forward(sample: torch.Tensor | numpy.ndarray):
            return self.onnx_forward(
                model_inputs=sample,
                providers=["CPUExecutionProvider"],
            )

        return {
            "avg_time_torch": timeit_averaged_(torch_model, num_iterations, test_sample),
            "avg_time_onnx": timeit_averaged_(ort_forward, num_iterations, test_sample),
        }

    def onnx_load(self, onnx_filepath: str = "") -> onnx.ModelProto:
        """Load ONNX model from disk and store it in handler state.

        Args:
            onnx_filepath: Path to ONNX file. Uses last exported path when empty.

        Returns:
            Loaded ONNX model proto.
        """
        if onnx_filepath == "":
            onnx_filepath = self.onnx_filepath

        self.onnx_model = onnx.load(onnx_filepath)
        return self.onnx_model

    def onnx_load_to_torch(self) -> torch.nn.Module:
        """Load ONNX model and convert it to torch module.

        Raises:
            NotImplementedError: Conversion path is not implemented.
        """
        raise NotImplementedError(
            "ONNX-to-torch conversion is not implemented yet.")

    def onnx_simplify(self, onnx_model: onnx.ModelProto) -> tuple[str, onnx.ModelProto]:
        """Simplify and overwrite an ONNX model file.

        Args:
            onnx_model: ONNX model proto to simplify.

        Returns:
            Tuple with output filepath and simplified model proto.
        """

        try:
            from onnxsim import simplify

        except ImportError as import_err:
            print(
                "\033[38;5;208mONNX simplification requested but onnx-simplify package is not installed. "
                "Please install onnx-simplify to enable this feature. Skipping simplification.\033[0m")

            return self.onnx_filepath, onnx_model

        model_simplified, check = simplify(onnx_model)
        onnx.save(model_simplified, self.onnx_filepath)
        print(f"ONNX model simplified and saved: {self.onnx_filepath}")

        if not check:
            print(
                "\033[38;5;208mWarning: ONNX model simplifier internal validation failed.\033[0m")

        return self.onnx_filepath, model_simplified
