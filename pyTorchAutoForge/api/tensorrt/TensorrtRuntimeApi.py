"""TensorRT runtime inference API.

Example:
    >>> import torch
    >>> from pyTorchAutoForge.api.tensorrt import TensorrtRuntimeApi
    >>> runtime_api = TensorrtRuntimeApi(engine_filepath="/tmp/model.engine")
    >>> # runtime_api.forward(model_inputs=torch.randn(1, 3, 224, 224))  # Requires CUDA + TensorRT.
"""

from __future__ import annotations

import hashlib
import os
import warnings
from typing import Any

import numpy
import torch

from pyTorchAutoForge.utils import torch_to_numpy

_UNSET_SOURCE = object()
# TODO typing for TensorRT module and engine objects when available


class TensorrtRuntimeApi:
    """Runtime API for TensorRT engine inference.

    This class manages TensorRT engine loading, execution context lifecycle,
    input normalization, and cached runtime sessions.

    Notes:
        Source resolution order for :meth:`forward` is:
        1. Explicit ``engine_bytes`` argument.
        2. Stored ``self.engine_bytes``.
        3. Explicit ``engine_filepath`` argument.
        4. Stored ``self.engine_filepath``.
    """

    def __init__(self,
                 engine_bytes: bytes | None = None,
                 engine_filepath: str | None = None,
                 device: str = "cuda:0",
                 ) -> None:
        """Initialize TensorRT runtime API.

        Args:
            engine_bytes: Optional in-memory serialized TensorRT engine.
            engine_filepath: Optional serialized engine filepath.
            device: Target execution device. CUDA device is required.
        """
        self.engine_bytes = engine_bytes
        self.engine_filepath = engine_filepath
        self.device = device
        self._session_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._session_meta: dict[tuple[Any, ...], dict[str, Any]] = {}

    def _import_tensorrt(self):
        """Import TensorRT Python module.

        Returns:
            Imported ``tensorrt`` module object.

        Raises:
            ImportError: If TensorRT is not available.
        """
        try:
            import tensorrt as trt
        except ImportError as import_error:
            raise ImportError(
                "TensorRT Python API is not available. Install 'tensorrt' to use TensorrtRuntimeApi."
            ) from import_error
        return trt

    def _resolve_engine_source(self,
                               engine_bytes: bytes | None = None,
                               engine_filepath: str | None = None,
                               ) -> tuple[str, bytes | str]:
        """Resolve engine source payload in priority order.

        Args:
            engine_bytes: Optional bytes source override.
            engine_filepath: Optional filepath source override.

        Returns:
            Tuple of ``(source_kind, source_payload)``, where source kind is
            ``"bytes"`` or ``"file"``.

        Raises:
            ValueError: If no usable engine source is available.
        """
        bytes_source = engine_bytes if engine_bytes is not None else self.engine_bytes
        if bytes_source is not None:
            return "bytes", bytes(bytes_source)

        filepath_source = engine_filepath if engine_filepath is not None else self.engine_filepath
        if filepath_source is not None and os.path.isfile(filepath_source):
            return "file", os.path.abspath(filepath_source)

        raise ValueError(
            "No TensorRT engine source available. Provide engine_bytes or a valid engine_filepath."
        )

    def _build_session_cache_key(self,
                                 source_kind: str,
                                 source_payload: bytes | str,
                                 device: str,
                                 ) -> tuple[Any, ...]:
        """Build deterministic cache key for TensorRT session reuse."""
        if source_kind == "bytes":
            source_hash = hashlib.sha256(source_payload).hexdigest()
            source_identity = ("bytes", source_hash, len(source_payload))
        else:
            stat = os.stat(source_payload)
            source_identity = ("file", source_payload,
                               stat.st_mtime_ns, stat.st_size)
        return (source_identity, device)

    def _load_engine_blob(self, source_kind: str, source_payload: bytes | str) -> bytes:
        """Load serialized engine bytes from source payload."""
        if source_kind == "bytes":
            return source_payload

        with open(source_payload, "rb") as file_handle:
            return file_handle.read()

    def _resolve_io_metadata(self, trt, engine) -> dict[str, Any]:
        """Resolve TensorRT engine input/output metadata."""
        if not hasattr(engine, "num_io_tensors"):
            raise RuntimeError(
                "TensorRT engine does not expose num_io_tensors/get_tensor_name API required for runtime execution."
            )

        input_names: list[str] = []
        output_names: list[str] = []
        input_dtypes: dict[str, Any] = {}
        output_dtypes: dict[str, Any] = {}

        for idx in range(int(engine.num_io_tensors)):
            tensor_name = engine.get_tensor_name(idx)
            tensor_mode = engine.get_tensor_mode(tensor_name)
            tensor_dtype = engine.get_tensor_dtype(tensor_name)

            if tensor_mode == trt.TensorIOMode.INPUT:
                input_names.append(tensor_name)
                input_dtypes[tensor_name] = tensor_dtype
            elif tensor_mode == trt.TensorIOMode.OUTPUT:
                output_names.append(tensor_name)
                output_dtypes[tensor_name] = tensor_dtype

        if len(input_names) == 0:
            raise RuntimeError("TensorRT engine has no input tensors.")
        if len(output_names) == 0:
            raise RuntimeError("TensorRT engine has no output tensors.")

        return {
            "input_names": input_names,
            "output_names": output_names,
            "input_dtypes": input_dtypes,
            "output_dtypes": output_dtypes,
        }

    def _create_session_entry(self, trt, source_kind: str, source_payload: bytes | str) -> dict[str, Any]:
        """Create a TensorRT runtime session entry from source."""
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        engine_blob = self._load_engine_blob(source_kind, source_payload)
        engine = runtime.deserialize_cuda_engine(engine_blob)
        if engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine.")

        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("Failed to create TensorRT execution context.")

        metadata = self._resolve_io_metadata(trt=trt, engine=engine)
        return {
            "logger": logger,
            "runtime": runtime,
            "engine": engine,
            "context": context,
            **metadata,
        }

    def _to_numpy_input(self, value: torch.Tensor | numpy.ndarray) -> numpy.ndarray:
        """Convert supported runtime input value to numpy array."""
        if isinstance(value, (torch.Tensor, numpy.ndarray)):
            return torch_to_numpy(value)
        raise TypeError(f"Unsupported input type: {type(value)}")

    def _build_input_map(self,
                         input_names: list[str],
                         model_inputs: (torch.Tensor
                                        | numpy.ndarray
                                        | tuple[torch.Tensor | numpy.ndarray, ...]
                                        | list[torch.Tensor | numpy.ndarray]
                                        | dict[str, torch.Tensor | numpy.ndarray]
                                        | None
                                        ),
                         input_tensor: (torch.Tensor
                                        | numpy.ndarray
                                        | tuple[torch.Tensor | numpy.ndarray, ...]
                                        | list[torch.Tensor | numpy.ndarray]
                                        | None
                                        ),
                         ) -> dict[str, numpy.ndarray]:
        """Normalize user input values to tensor-name input map.

        Args:
            input_names: TensorRT engine input tensor names.
            model_inputs: Input values in dict, tuple/list, or single-input form.
            input_tensor: Deprecated alias for ``model_inputs``.

        Returns:
            Input map keyed by TensorRT engine input names.

        Raises:
            ValueError: If values are missing, conflicting, or mismatched.
            TypeError: If unsupported input value type is provided.
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

        if model_inputs is None:
            raise ValueError(
                "model_inputs is required for TensorRT runtime forward.")

        if isinstance(model_inputs, dict):
            provided_keys = set(model_inputs.keys())
            expected_keys = set(input_names)
            missing_keys = sorted(expected_keys - provided_keys)
            extra_keys = sorted(provided_keys - expected_keys)

            if missing_keys or extra_keys:
                raise ValueError(
                    f"Invalid input dict keys. missing={missing_keys}, extra={extra_keys}, expected={input_names}"
                )
            return {name: self._to_numpy_input(model_inputs[name]) for name in input_names}

        if isinstance(model_inputs, (tuple, list)):
            if len(model_inputs) != len(input_names):
                raise ValueError(
                    f"Expected {len(input_names)} inputs, got {len(model_inputs)}.")
            return {
                input_names[idx]: self._to_numpy_input(value)
                for idx, value in enumerate(model_inputs)
            }

        if len(input_names) != 1:
            raise ValueError(
                f"Single input value provided but model expects {len(input_names)} inputs."
            )
        return {input_names[0]: self._to_numpy_input(model_inputs)}

    def _trt_dtype_to_torch_dtype(self, trt, trt_dtype: Any) -> torch.dtype:
        """Map TensorRT tensor dtype to torch dtype."""

        if hasattr(trt, "nptype"):
            np_dtype = numpy.dtype(trt.nptype(trt_dtype))
            return torch.from_numpy(numpy.empty((), dtype=np_dtype)).dtype

        dtype_name = str(trt_dtype).lower()

        if "float16" in dtype_name or "half" in dtype_name:
            return torch.float16
        if "float32" in dtype_name or "float" in dtype_name:
            return torch.float32
        if "int8" in dtype_name:
            return torch.int8
        if "int32" in dtype_name:
            return torch.int32
        if "bool" in dtype_name:
            return torch.bool

        raise TypeError(f"Unsupported TensorRT dtype: {trt_dtype}")

    def _run_engine(self,
                    trt,
                    session_entry: dict[str, Any],
                    input_map: dict[str, numpy.ndarray],
                    device: str,
                    ) -> dict[str, numpy.ndarray]:
        """Execute TensorRT inference for one forward call (internal implementation)."""
        device_obj = torch.device(device)
        if device_obj.type != "cuda":
            raise ValueError("TensorRT runtime requires a CUDA device.")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. TensorRT runtime requires CUDA.")

        context = session_entry["context"]
        input_names = session_entry["input_names"]
        output_names = session_entry["output_names"]
        input_dtypes = session_entry["input_dtypes"]
        output_dtypes = session_entry["output_dtypes"]

        input_tensors: dict[str, torch.Tensor] = {}
        output_tensors: dict[str, torch.Tensor] = {}

        for input_name in input_names:

            # Construct input tensors
            input_np = numpy.ascontiguousarray(input_map[input_name])
            torch_dtype = self._trt_dtype_to_torch_dtype(
                trt, input_dtypes[input_name])

            input_tensor = torch.as_tensor(
                input_np, dtype=torch_dtype, device=device_obj
            ).contiguous()

            input_tensors[input_name] = input_tensor

            if hasattr(context, "set_input_shape"):
                shape_ok = context.set_input_shape(
                    input_name, tuple(input_tensor.shape))
                if shape_ok is False:
                    raise ValueError(
                        f"Unable to set TensorRT input shape for tensor '{input_name}' to {tuple(input_tensor.shape)}."
                    )

            addr_ok = context.set_tensor_address(
                input_name, int(input_tensor.data_ptr()))

            if addr_ok is False:
                raise RuntimeError(
                    f"Failed to bind input tensor address for '{input_name}'.")

        # Construct output tensors and bind addresses
        for output_name in output_names:
            output_shape = tuple(int(dim)
                                 for dim in context.get_tensor_shape(output_name))
            if any(dim < 0 for dim in output_shape):
                raise RuntimeError(
                    f"TensorRT output shape for '{output_name}' is unresolved after binding inputs: {output_shape}."
                )

            output_dtype = self._trt_dtype_to_torch_dtype(
                trt, output_dtypes[output_name])
            output_tensor = torch.empty(
                output_shape, dtype=output_dtype, device=device_obj)
            output_tensors[output_name] = output_tensor

            addr_ok = context.set_tensor_address(
                output_name, int(output_tensor.data_ptr()))
            if addr_ok is False:
                raise RuntimeError(
                    f"Failed to bind output tensor address for '{output_name}'.")

        # Setup CUDA stream and execute asynchronously
        stream = torch.cuda.current_stream(device=device_obj)
        run_ok = context.execute_async_v3(stream.cuda_stream)

        if run_ok is False:
            raise RuntimeError(
                "TensorRT execution failed in execute_async_v3.")
        stream.synchronize()  # Wait for execution to complete

        # Return output tensors as numpy arrays
        return {
            output_name: output_tensor.detach().cpu().numpy()
            for output_name, output_tensor in output_tensors.items()
        }

    # PUBLIC API METHODS
    def set_engine_source(self,
                          engine_bytes: bytes | None | object = _UNSET_SOURCE,
                          engine_filepath: str | None | object = _UNSET_SOURCE,
                          ) -> None:
        """Update TensorRT engine source references.

        Args:
            engine_bytes: Optional serialized engine bytes. If explicitly set to
                ``None``, the stored byte source is cleared.
            engine_filepath: Optional serialized engine path. If explicitly set to
                ``None``, the stored file source is cleared.
        """
        if engine_bytes is not _UNSET_SOURCE:
            self.engine_bytes = engine_bytes
        if engine_filepath is not _UNSET_SOURCE:
            self.engine_filepath = engine_filepath

    def clear_session_cache(self) -> None:
        """Clear cached TensorRT runtime sessions."""
        self._session_cache.clear()
        self._session_meta.clear()

    def forward(self,
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
                force_new_session: bool = False,
                engine_bytes: bytes | None = None,
                engine_filepath: str | None = None,
                device: str | None = None,
                ) -> dict[str, numpy.ndarray]:
        """Run TensorRT forward inference.

        Args:
            model_inputs: Runtime model inputs in tensor, tuple/list, or dict form.
            input_tensor: Deprecated alias for ``model_inputs``.
            force_new_session: Whether to bypass and refresh session cache.
            engine_bytes: Optional serialized engine override source.
            engine_filepath: Optional serialized engine filepath override source.
            device: Optional device override. Uses class default when omitted.

        Returns:
            Output dictionary keyed by TensorRT output names.

        Raises:
            ImportError: If TensorRT Python package is unavailable.
            RuntimeError: If CUDA is unavailable or TensorRT execution fails.
            ValueError: If source or input contract is invalid.
            TypeError: If unsupported input dtype is encountered.
        """

        trt = self._import_tensorrt()
        target_device = device if device is not None else self.device
        source_kind, source_payload = self._resolve_engine_source(
            engine_bytes=engine_bytes,
            engine_filepath=engine_filepath,
        )
        cache_key = self._build_session_cache_key(
            source_kind=source_kind,
            source_payload=source_payload,
            device=target_device,
        )

        if (not force_new_session) and (cache_key in self._session_cache):
            session_entry = self._session_cache[cache_key]
        else:
            # Initialize tensorrt engine and execution context
            session_entry = self._create_session_entry(trt=trt,
                                                       source_kind=source_kind,
                                                       source_payload=source_payload,
                                                       )
            self._session_cache[cache_key] = session_entry
            self._session_meta[cache_key] = {
                "source_kind": source_kind,
                "device": target_device,
                "input_names": tuple(session_entry["input_names"]),
                "output_names": tuple(session_entry["output_names"]),
            }

        input_map = self._build_input_map(input_names=session_entry["input_names"],
                                          model_inputs=model_inputs,
                                          input_tensor=input_tensor,
                                          )

        return self._run_engine(trt=trt,
                                session_entry=session_entry,
                                input_map=input_map,
                                device=target_device,
                                )
