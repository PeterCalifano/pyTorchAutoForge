"""TensorRT engine export utilities.

This module provides a single class, :class:`TRTengineExporter`, that supports
three export workflows:

1. ONNX file path -> TensorRT engine.
2. ONNX ModelProto -> TensorRT engine.
3. torch model -> ONNX (via ``ModelHandlerONNx``) -> TensorRT engine.

Dependency matrix:
- ``TRTengineExporterMode.TRTEXEC`` requires ``trtexec`` in ``PATH``.
- ``TRTengineExporterMode.PYTHON`` requires the ``tensorrt`` Python package.
- torch->onnx convenience methods require torch/onnx dependencies already used by
  ``pyTorchAutoForge.api.onnx.ModelHandlerONNx``.

Jetson compatibility notes:
- No x86-specific flags are injected.
- ``trtexec`` is discovered from ``PATH`` (Jetson default deployment style).
- Dynamic shape profiles and workspace size are configurable to support
  constrained-memory targets.

Example (ONNX path -> engine):
    >>> from pyTorchAutoForge.api.tensorrt import TRTengineExporter
    >>> exporter = TRTengineExporter()
    >>> _ = exporter.build_engine_from_onnx_path("/tmp/model.onnx", "/tmp/model.engine")
    Engine built successfully and saved to: /tmp/model.engine

Example (torch -> onnx -> engine):
    >>> # Assuming model_ and sample_ are valid torch model/input.
    >>> exporter = TRTengineExporter()
    >>> _ = exporter.export_torch_to_trt_engine(
    ...     torch_model=model_,
    ...     input_sample=sample_,
    ...     onnx_export_path="/tmp/intermediate.onnx",
    ...     output_engine_path="/tmp/final.engine",
    ... )
    Engine built successfully and saved to: /tmp/final.engine
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import importlib
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import onnx
    import torch


class TRTprecision(Enum):
    """Supported TensorRT precision modes."""

    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"


class TRTengineExporterMode(Enum):
    """Engine build backends."""

    TRTEXEC = "trtexec"
    PYTHON = "python"


@dataclass(frozen=True)
class TRTDynamicShapeProfile:
    """Dynamic shape profile for one ONNX network input."""

    input_name: str
    min_shape: tuple[int, ...]
    opt_shape: tuple[int, ...]
    max_shape: tuple[int, ...]


@dataclass
class TRTengineExporterConfig:
    """Configuration object for :class:`TRTengineExporter`."""

    exporter_mode: TRTengineExporterMode = TRTengineExporterMode.TRTEXEC
    precision: TRTprecision = TRTprecision.FP32
    default_output_engine_path: str | None = None
    verbose: bool = False
    workspace_size_bytes: int | None = None
    dynamic_shape_profiles: tuple[TRTDynamicShapeProfile, ...] | None = None
    trtexec_extra_args: tuple[str, ...] = ()


class TRTengineExporter:
    """Helper class to export TensorRT engines from ONNX or torch models.

    The constructor is side-effect free: no export/build operation is executed
    at instantiation time.
    """

    def __init__(self,
                 config: TRTengineExporterConfig | None = None,
                 *,
                 exporter_mode: TRTengineExporterMode | None = None,
                 precision: TRTprecision | None = None,
                 default_output_engine_path: str | None = None,
                 verbose: bool | None = None,
                 workspace_size_bytes: int | None = None,
                 dynamic_shape_profiles: tuple[TRTDynamicShapeProfile, ...] | None = None,
                 trtexec_extra_args: tuple[str, ...] | None = None,
                 ) -> None:
        """Initialize exporter configuration.

        Keyword arguments override ``config`` fields when they are not ``None``.
        """

        # Get base configuration from object
        config_ = config if config is not None else TRTengineExporterConfig()

        # Process overrides by kwargs
        self.exporter_mode = exporter_mode or config_.exporter_mode
        self.precision = precision or config_.precision
        self.default_output_engine_path = (
            default_output_engine_path
            if default_output_engine_path is not None
            else config_.default_output_engine_path
        )

        self.verbose = verbose if verbose is not None else config_.verbose
        self.workspace_size_bytes = (
            workspace_size_bytes
            if workspace_size_bytes is not None
            else config_.workspace_size_bytes
        )
        self.dynamic_shape_profiles = (
            dynamic_shape_profiles
            if dynamic_shape_profiles is not None
            else config_.dynamic_shape_profiles
        )
        self.trtexec_extra_args = (
            trtexec_extra_args
            if trtexec_extra_args is not None
            else config_.trtexec_extra_args
        )
        self._trtexec_supports_mem_pool_size_cache: bool | None = None

        # Validate configuration
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        """Validate configuration fields for consistency and basic correctness."""
        if self.workspace_size_bytes is not None and self.workspace_size_bytes <= 0:
            raise ValueError("workspace_size_bytes must be > 0 when provided.")

        if self.dynamic_shape_profiles is None:
            return

        for profile_ in self.dynamic_shape_profiles:
            self._validate_profile(profile_)

    def _validate_profile(self, profile: TRTDynamicShapeProfile) -> None:
        """Validate a single dynamic shape profile for basic correctness."""
        if profile.input_name.strip() == "":
            raise ValueError(
                "TRTDynamicShapeProfile.input_name cannot be empty.")

        if not (len(profile.min_shape) == len(profile.opt_shape)
                and len(profile.opt_shape) == len(profile.max_shape)):

            raise ValueError(
                f"Shape rank mismatch for input '{profile.input_name}'."
            )

        # Check that all dimensions are > 0 and that min <= opt <= max for each dimension
        for min_dim_, opt_dim_, max_dim_ in zip(profile.min_shape,
                                                profile.opt_shape,
                                                profile.max_shape):

            if min_dim_ <= 0 or opt_dim_ <= 0 or max_dim_ <= 0:
                raise ValueError(
                    f"All dimensions must be > 0 for input '{profile.input_name}'."
                )
            if not (min_dim_ <= opt_dim_ <= max_dim_):
                raise ValueError(
                    f"Expected min <= opt <= max for input '{profile.input_name}'."
                )

    def _import_tensorrt(self) -> Any:
        """Try to import the TensorRT Python API, raising error if it fails."""
        try:
            return importlib.import_module("tensorrt")

        except ImportError as import_error_:
            raise ImportError(
                "TensorRT Python API is not available. Install 'tensorrt' or "
                "use TRTengineExporterMode.TRTEXEC."
            ) from import_error_

    def _import_onnx(self) -> Any:
        """Try to import the ONNX package, raising error if it fails. Required for ONNX"""
        try:
            return importlib.import_module("onnx")
        except ImportError as import_error_:
            raise ImportError(
                "ONNX package is required for ONNX model object export pathways."
            ) from import_error_

    def _import_model_handler_onnx(self) -> Any:
        """Try to import ModelHandlerONNx, raising error if it fails. Required for torch->onnx export pathways."""
        try:
            module_ = importlib.import_module("pyTorchAutoForge.api.onnx")
            return module_.ModelHandlerONNx
        except Exception as import_error_:
            raise ImportError(
                "Failed to import pyTorchAutoForge.api.onnx.ModelHandlerONNx."
            ) from import_error_

    def _resolve_output_engine_path(self,
                                    output_engine_path: str | None,
                                    onnx_model_path: str | None = None,
                                    ) -> str:
        """Resolve the output engine path based on priority: explicit argument, default config, ONNX path stem, or fallback."""

        if output_engine_path is not None:
            output_path_ = Path(output_engine_path)
        elif self.default_output_engine_path is not None:
            output_path_ = Path(self.default_output_engine_path)
        elif onnx_model_path is not None:
            output_path_ = Path(onnx_model_path).with_suffix(".engine")
        else:
            output_path_ = Path("./trt_engine.engine")

        output_path_.parent.mkdir(parents=True, exist_ok=True)
        return str(output_path_)

    def _validate_onnx_input_path(self, onnx_model_path: str) -> str:
        """Validate the ONNX input path for existence and correct extension."""
        if not onnx_model_path.lower().endswith(".onnx"):
            raise ValueError(
                f"Input ONNX model path must end with '.onnx': {onnx_model_path}"
            )

        if not os.path.isfile(onnx_model_path):
            raise FileNotFoundError(
                f"Input ONNX model file not found: {onnx_model_path}")

        return onnx_model_path

    def _shape_to_trtexec_string(self, shape: tuple[int, ...]) -> str:
        """Convert a shape tuple to the string format expected by trtexec (e.g., '1x3x224x224')."""
        return "x".join(str(dim_) for dim_ in shape)

    def _build_dynamic_shapes_flag(self, shape_kind: str) -> str:
        """Define a single dynamic shape flag (min/opt/max) for trtexec based on the configured profiles."""
        if self.dynamic_shape_profiles is None:
            raise ValueError("Dynamic shape profiles are not configured.")

        fragments_: list[str] = []
        for profile_ in self.dynamic_shape_profiles:
            shape_ = getattr(profile_, shape_kind)
            fragments_.append(
                f"{profile_.input_name}:{self._shape_to_trtexec_string(shape_)}"
            )

        return ",".join(fragments_)

    def _apply_workspace_size_to_builder_config(self,
                                                builder_config: Any,
                                                trt_module: Any,
                                                ) -> None:
        """Apply workspace size with TensorRT API compatibility guards.

        The parameter is accepted across TensorRT versions. When workspace APIs
        are missing/deprecated on a runtime, this method silently skips setting
        the limit (optionally reporting in verbose mode).
        """
        if self.workspace_size_bytes is None:
            return

        # Preferred API for modern TensorRT versions.
        if hasattr(builder_config, "set_memory_pool_limit"):
            memory_pool_type_ = getattr(trt_module, "MemoryPoolType", None)
            workspace_pool_ = (
                getattr(memory_pool_type_, "WORKSPACE", None)
                if memory_pool_type_ is not None
                else None
            )
            if workspace_pool_ is not None:
                try:
                    builder_config.set_memory_pool_limit(
                        workspace_pool_, self.workspace_size_bytes
                    )
                    return
                except Exception as workspace_error_:
                    if self.verbose:
                        print(
                            "TensorRT workspace pool limit call failed and will be ignored: "
                            f"{workspace_error_}"
                        )

        # Legacy API fallback for older TensorRT versions.
        if hasattr(builder_config, "max_workspace_size"):
            try:
                builder_config.max_workspace_size = self.workspace_size_bytes
                return
            except Exception as workspace_error_:
                if self.verbose:
                    print(
                        "TensorRT legacy max_workspace_size assignment failed and will be ignored: "
                        f"{workspace_error_}"
                    )

        if self.verbose:
            print(
                "workspace_size_bytes was provided but no compatible TensorRT workspace API is available. "
                "Continuing without explicit workspace size."
            )

    def _build_with_trtexec(self,
                            onnx_model_path: str,
                            output_engine_path: str,
                            ) -> str:
        """Build a TensorRT engine from an ONNX file using the trtexec CLI tool."""

        trtexec_path_ = shutil.which("trtexec")
        if trtexec_path_ is None:
            raise FileNotFoundError(
                "trtexec executable not found in PATH. Install TensorRT runtime "
                "tools and ensure trtexec is available."
            )

        # Compose command with required arguments and optional flags based on configuration
        command_: list[str] = [
            trtexec_path_,
            f"--onnx={onnx_model_path}",
            f"--saveEngine={output_engine_path}",
        ]

        # Precision
        if self.precision == TRTprecision.FP16:
            command_.append("--fp16")
        elif self.precision == TRTprecision.INT8:
            command_.append("--int8")

        # Workspace size
        if self.workspace_size_bytes is not None:
            workspace_mib_ = max(
                1, int(math.ceil(self.workspace_size_bytes / (1024 * 1024))))
            if self._trtexec_supports_mem_pool_size(trtexec_path_):
                command_.append(f"--memPoolSize=workspace:{workspace_mib_}")
            else:
                command_.append(f"--workspace={workspace_mib_}")

        # Dynamic shapes
        if self.dynamic_shape_profiles is not None:
            command_.append(
                f"--minShapes={self._build_dynamic_shapes_flag('min_shape')}")
            command_.append(
                f"--optShapes={self._build_dynamic_shapes_flag('opt_shape')}")
            command_.append(
                f"--maxShapes={self._build_dynamic_shapes_flag('max_shape')}")

        # Extra args
        command_.extend(self.trtexec_extra_args)

        # Run trt in subprocess (captures terminal out in pipe)
        result_ = subprocess.run(command_,
                                 check=False,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 text=True,
                                 )

        if result_.returncode != 0:
            raise RuntimeError(
                "trtexec failed with non-zero exit code. "
                f"stdout: {result_.stdout.strip()} | stderr: {result_.stderr.strip()}"
            )

        if not os.path.isfile(output_engine_path):
            raise RuntimeError(
                "trtexec completed but output engine file was not found: "
                f"{output_engine_path}"
            )

        # Print trtexec output when verbose is enabled
        if self.verbose:
            print(result_.stdout)

        print(
            f"\033[92mtrt engine built successfully and saved to: {output_engine_path}\033[0m")
        return output_engine_path

    def _trtexec_supports_mem_pool_size(self, trtexec_path: str) -> bool:
        """Check whether trtexec supports `--memPoolSize` workspace syntax."""
        if self._trtexec_supports_mem_pool_size_cache is not None:
            return self._trtexec_supports_mem_pool_size_cache

        try:
            help_result_ = subprocess.run(
                [trtexec_path, "--help"],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as help_error_:
            if self.verbose:
                print(
                    "Unable to probe trtexec help for --memPoolSize support; "
                    f"falling back to --workspace. Details: {help_error_}"
                )
            self._trtexec_supports_mem_pool_size_cache = False
            return False

        help_output_ = f"{help_result_.stdout}\n{help_result_.stderr}"
        supports_mem_pool_size_ = "--memPoolSize" in help_output_
        self._trtexec_supports_mem_pool_size_cache = supports_mem_pool_size_
        return supports_mem_pool_size_

    def _build_with_python_api_from_onnx_bytes(self,
                                               onnx_model_bytes: bytes,
                                               output_engine_path: str,
                                               ) -> str:
        """Build a TensorRT engine from ONNX model bytes using the TensorRT Python API."""
        trt_ = self._import_tensorrt()

        # Set up logger, builder, network, and parser
        logger_level_ = trt_.Logger.VERBOSE if self.verbose else trt_.Logger.WARNING
        logger_ = trt_.Logger(logger_level_)
        builder_ = trt_.Builder(logger_)

        network_flags_ = 1 << int(
            trt_.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network_ = builder_.create_network(network_flags_)
        parser_ = trt_.OnnxParser(network_, logger_)

        if not parser_.parse(onnx_model_bytes):
            num_errors_ = int(getattr(parser_, "num_errors", 0))
            parse_errors_ = [str(parser_.get_error(idx_))
                             for idx_ in range(num_errors_)]
            raise RuntimeError(
                "TensorRT ONNX parser failed: " + " | ".join(parse_errors_)
            )

        builder_config_ = builder_.create_builder_config()

        # Set precision
        if self.precision == TRTprecision.FP16:
            builder_config_.set_flag(trt_.BuilderFlag.FP16)

        elif self.precision == TRTprecision.INT8:
            builder_config_.set_flag(trt_.BuilderFlag.INT8)

        # Set workspace limit with compatibility guards.
        self._apply_workspace_size_to_builder_config(builder_config_, trt_)

        # Setup dynamic shape profiles
        if self.dynamic_shape_profiles is not None:

            # Create optimization profile
            optimization_profile_ = builder_.create_optimization_profile()

            for profile_ in self.dynamic_shape_profiles:
                optimization_profile_.set_shape(
                    profile_.input_name,
                    profile_.min_shape,
                    profile_.opt_shape,
                    profile_.max_shape,
                )
            builder_config_.add_optimization_profile(optimization_profile_)

        # Build serialized engine
        serialized_engine_: Any | None = None
        if hasattr(builder_, "build_serialized_network"):
            serialized_engine_ = builder_.build_serialized_network(
                network_, builder_config_)
        else:
            engine_ = builder_.build_engine(network_, builder_config_)
            if engine_ is not None:
                serialized_engine_ = engine_.serialize()

        if serialized_engine_ is None:
            raise RuntimeError(
                "TensorRT engine build failed: serialized engine is None.")

        # Get serialized engine bytes and write to output path
        if isinstance(serialized_engine_, (bytes, bytearray)):
            serialized_engine_bytes_ = bytes(serialized_engine_)
        else:
            serialized_engine_bytes_ = bytes(serialized_engine_)

        with open(output_engine_path, "wb") as file_handle_:
            file_handle_.write(serialized_engine_bytes_)

        print(f"Engine built successfully and saved to: {output_engine_path}")
        return output_engine_path

    def _dispatch_build_from_onnx_path(self,
                                       onnx_model_path: str,
                                       output_engine_path: str | None = None,
                                       ) -> str:
        """Dispatch method to build engine from ONNX path based on exporter mode."""
        validated_path_ = self._validate_onnx_input_path(onnx_model_path)
        resolved_output_path_ = self._resolve_output_engine_path(
            output_engine_path=output_engine_path,
            onnx_model_path=validated_path_,
        )

        # Dispatch call to correct implementation
        if self.exporter_mode == TRTengineExporterMode.TRTEXEC:
            return self._build_with_trtexec(validated_path_, resolved_output_path_)

        if self.exporter_mode == TRTengineExporterMode.PYTHON:

            # Read ONNX model bytes from file
            with open(validated_path_, "rb") as file_handle_:
                onnx_model_bytes_ = file_handle_.read()

            return self._build_with_python_api_from_onnx_bytes(
                onnx_model_bytes_, resolved_output_path_
            )

        raise ValueError(f"Unsupported exporter mode: {self.exporter_mode}")

    def _save_onnx_model_to_path(self, onnx_model: "onnx.ModelProto", onnx_path: str) -> None:
        onnx_ = self._import_onnx()
        onnx_.save(onnx_model, onnx_path)

    def _serialize_onnx_model_to_bytes(self, onnx_model: "onnx.ModelProto") -> bytes:
        if hasattr(onnx_model, "SerializeToString"):
            return bytes(onnx_model.SerializeToString())

        raise TypeError(
            "ONNX model object must expose SerializeToString() for PYTHON export mode."
        )

    def _dispatch_build_from_onnx_model(self,
                                        onnx_model: "onnx.ModelProto",
                                        output_engine_path: str | None = None,
                                        temporary_onnx_path: str | None = None,
                                        ) -> str:
        """Dispatch method to build engine from ONNX model object based on exporter mode. For PYTHON mode, the model is serialized in-memory. For TRTEXEC mode, the model is saved to a temporary ONNX file path used for building, which is cleaned up after build completion."""

        resolved_output_path_ = self._resolve_output_engine_path(
            output_engine_path=output_engine_path,
            onnx_model_path=None,
        )

        if self.exporter_mode == TRTengineExporterMode.PYTHON:
            onnx_model_bytes_ = self._serialize_onnx_model_to_bytes(onnx_model)
            return self._build_with_python_api_from_onnx_bytes(
                onnx_model_bytes_, resolved_output_path_
            )

        if self.exporter_mode != TRTengineExporterMode.TRTEXEC:
            raise ValueError(
                f"Unsupported exporter mode: {self.exporter_mode}")

        # TRTEXEC mode: save temporary onnx file to memory
        temporary_path_created_ = False

        if temporary_onnx_path is None:
            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp_file_:
                temporary_onnx_path_ = tmp_file_.name
            temporary_path_created_ = True

        else:

            temporary_path_ = Path(temporary_onnx_path)
            if temporary_path_.suffix.lower() != ".onnx":
                temporary_path_ = temporary_path_.with_suffix(".onnx")

            temporary_path_.parent.mkdir(parents=True, exist_ok=True)
            temporary_onnx_path_ = str(temporary_path_)

        self._save_onnx_model_to_path(
            onnx_model=onnx_model, onnx_path=temporary_onnx_path_)

        # Try to run build and clean up temporary model
        try:
            return self._build_with_trtexec(
                onnx_model_path=temporary_onnx_path_,
                output_engine_path=resolved_output_path_,
            )
        finally:
            if temporary_path_created_ and os.path.exists(temporary_onnx_path_):
                os.remove(temporary_onnx_path_)

    # PUBLIC API METHODS
    def build_engine_from_onnx_path(self,
                                    onnx_model_path: str,
                                    output_engine_path: str | None = None,
                                    ) -> str:
        """Build a TensorRT engine from an ONNX file path.

        Args:
            onnx_model_path: Input ONNX file path.
            output_engine_path: Optional output engine path. When omitted, the
                engine path is resolved from config defaults or from the ONNX
                path stem.

        Returns:
            The generated engine file path.

        Raises:
            FileNotFoundError: If ONNX input path or `trtexec` is missing.
            ValueError: If ONNX path extension is invalid.
            RuntimeError: If TensorRT build fails.

        Example:
            >>> from pyTorchAutoForge.api.tensorrt import TRTengineExporter
            >>> exporter = TRTengineExporter()
            >>> _ = exporter.build_engine_from_onnx_path(\"/tmp/model.onnx\", \"/tmp/model.engine\")
            Engine built successfully and saved to: /tmp/model.engine
        """

        return self._dispatch_build_from_onnx_path(
            onnx_model_path=onnx_model_path,
            output_engine_path=output_engine_path,
        )

    def build_engine_from_onnx_model(self,
                                     onnx_model: "onnx.ModelProto",
                                     output_engine_path: str | None = None,
                                     temporary_onnx_path: str | None = None,
                                     ) -> str:
        """Build a TensorRT engine from an in-memory ONNX model object.

        Args:
            onnx_model: Input ONNX model object.
            output_engine_path: Optional output engine path.
            temporary_onnx_path: Optional ONNX temporary file path used by
                `trtexec` mode.

        Returns:
            The generated engine file path.

        Raises:
            ImportError: If ONNX package is not available for serialization.
            RuntimeError: If TensorRT build fails.
        """

        return self._dispatch_build_from_onnx_model(
            onnx_model=onnx_model,
            output_engine_path=output_engine_path,
            temporary_onnx_path=temporary_onnx_path,
        )

    def export_torch_to_onnx(self,
                             torch_model: "torch.nn.Module",
                             input_sample: "torch.Tensor | tuple[torch.Tensor, ...]",
                             *,
                             onnx_export_path: str,
                             backend: str = "legacy",
                             fallback_to_legacy: bool = False,
                             run_export_validation: bool = True,
                             opset_version: int = 13,
                             additional_export_kwargs: dict[str,
                                                            object] | None = None,
                             onnx_model_name: str | None = None,
                             ) -> str:
        """Export a torch model to ONNX via ``ModelHandlerONNx``.

        Args:
            torch_model: PyTorch model.
            input_sample: Input sample for export graph tracing.
            onnx_export_path: Export directory or explicit ONNX filepath.
            backend: ONNX export backend (`legacy` or `dynamo`).
            fallback_to_legacy: Enables fallback when dynamo backend fails.
            run_export_validation: Enables ONNX validation after export.
            opset_version: Target ONNX opset.
            additional_export_kwargs: Extra kwargs forwarded to exporter.
            onnx_model_name: Optional filename/path override.

        Returns:
            The ONNX filepath generated by `ModelHandlerONNx`.
        """

        model_handler_onnx_class_ = self._import_model_handler_onnx()
        handler_ = model_handler_onnx_class_(
            model=torch_model,
            dummy_input_sample=input_sample,
            onnx_export_path=onnx_export_path,
            opset_version=opset_version,
            run_export_validation=run_export_validation,
            generate_report=False,
            run_onnx_simplify=False,
        )

        exported_onnx_path_ = handler_.export_onnx(
            model_inputs=input_sample,
            backend=backend,
            fallback_to_legacy=fallback_to_legacy,
            additional_export_kwargs=additional_export_kwargs,
            onnx_model_name=onnx_model_name,
        )

        return str(exported_onnx_path_)

    def export_torch_to_trt_engine(self,
                                   torch_model: "torch.nn.Module",
                                   input_sample: "torch.Tensor | tuple[torch.Tensor, ...]",
                                   *,
                                   onnx_export_path: str,
                                   output_engine_path: str | None = None,
                                   backend: str = "legacy",
                                   fallback_to_legacy: bool = False,
                                   run_export_validation: bool = True,
                                   opset_version: int = 13,
                                   additional_export_kwargs: dict[str,
                                                                  object] | None = None,
                                   onnx_model_name: str | None = None,
                                   ) -> str:
        """Export a torch model to TensorRT by chaining torch->onnx->trt.

        Args:
            torch_model: PyTorch model.
            input_sample: Input sample for ONNX export.
            onnx_export_path: Intermediate ONNX output path or directory.
            output_engine_path: Optional TensorRT engine output path.
            backend: ONNX export backend.
            fallback_to_legacy: ONNX dynamo fallback behavior.
            run_export_validation: ONNX validation behavior.
            opset_version: ONNX opset version.
            additional_export_kwargs: Extra ONNX export kwargs.
            onnx_model_name: Optional ONNX filename/path override.

        Returns:
            The generated TensorRT engine filepath.

        Example:
            >>> from pyTorchAutoForge.api.tensorrt import TRTengineExporter
            >>> exporter = TRTengineExporter()
            >>> _ = exporter.export_torch_to_trt_engine(
            ...     torch_model=model_,
            ...     input_sample=sample_,
            ...     onnx_export_path=\"/tmp/intermediate.onnx\",
            ...     output_engine_path=\"/tmp/final.engine\",
            ... )
            Engine built successfully and saved to: /tmp/final.engine
        """

        exported_onnx_path_ = self.export_torch_to_onnx(
            torch_model=torch_model,
            input_sample=input_sample,
            onnx_export_path=onnx_export_path,
            backend=backend,
            fallback_to_legacy=fallback_to_legacy,
            run_export_validation=run_export_validation,
            opset_version=opset_version,
            additional_export_kwargs=additional_export_kwargs,
            onnx_model_name=onnx_model_name,
        )

        return self.build_engine_from_onnx_path(
            onnx_model_path=exported_onnx_path_,
            output_engine_path=output_engine_path,
        )

    def export_engine_end_to_end(self,
                                 *,
                                 torch_model: "torch.nn.Module | None" = None,
                                 input_sample: "torch.Tensor | tuple[torch.Tensor, ...] | None" = None,
                                 onnx_model_path: str | None = None,
                                 onnx_model: "onnx.ModelProto | None" = None,
                                 onnx_export_path: str | None = None,
                                 output_engine_path: str | None = None,
                                 backend: str = "legacy",
                                 fallback_to_legacy: bool = False,
                                 run_export_validation: bool = True,
                                 opset_version: int = 13,
                                 additional_export_kwargs: dict[str,
                                                                object] | None = None,
                                 onnx_model_name: str | None = None,
                                 ) -> str:
        """End-to-end exporter with explicit route selection and validation.

        Exactly one source route must be selected:
        - `torch_model` (+ `input_sample` + `onnx_export_path`)
        - `onnx_model_path`
        - `onnx_model`

        Args:
            torch_model: Optional torch model source.
            input_sample: Required for torch route.
            onnx_model_path: Optional ONNX path route.
            onnx_model: Optional ONNX model route.
            onnx_export_path: Required for torch route.
            output_engine_path: Optional output engine path.
            backend: ONNX export backend for torch route.
            fallback_to_legacy: ONNX dynamo fallback for torch route.
            run_export_validation: ONNX validation for torch route.
            opset_version: ONNX opset version for torch route.
            additional_export_kwargs: Extra ONNX exporter kwargs.
            onnx_model_name: Optional ONNX model name override.

        Returns:
            The generated TensorRT engine filepath.

        Raises:
            ValueError: If route selection is missing or ambiguous.
        """

        num_routes_selected_ = int(torch_model is not None) + int(
            onnx_model_path is not None) + int(onnx_model is not None)

        if num_routes_selected_ == 0:
            raise ValueError(
                "No input source provided. Provide one of: torch_model, "
                "onnx_model_path, onnx_model."
            )

        if num_routes_selected_ > 1:
            raise ValueError(
                "Ambiguous input source. Provide exactly one of: torch_model, "
                "onnx_model_path, onnx_model."
            )

        if torch_model is not None:
            # Use torch model if provided
            if input_sample is None:
                raise ValueError(
                    "input_sample is required when torch_model is provided.")
            if onnx_export_path is None:
                raise ValueError(
                    "onnx_export_path is required when torch_model is provided."
                )

            return self.export_torch_to_trt_engine(
                torch_model=torch_model,
                input_sample=input_sample,
                onnx_export_path=onnx_export_path,
                output_engine_path=output_engine_path,
                backend=backend,
                fallback_to_legacy=fallback_to_legacy,
                run_export_validation=run_export_validation,
                opset_version=opset_version,
                additional_export_kwargs=additional_export_kwargs,
                onnx_model_name=onnx_model_name,
            )

        # Else, export onnx model from path or object
        if onnx_model_path is not None:
            return self.build_engine_from_onnx_path(
                onnx_model_path=onnx_model_path,
                output_engine_path=output_engine_path,
            )

        # Else, onnx_model must be not None and run from it
        assert onnx_model is not None, "ERROR: onnx_model is None when torch_model and onnx_model_path are also None. Please provide a torch model, an ONNX model path, or an ONNX model object."
        return self.build_engine_from_onnx_model(onnx_model=onnx_model,
                                                 output_engine_path=output_engine_path,
                                                 )
