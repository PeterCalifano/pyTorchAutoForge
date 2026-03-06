"""Unified runtime facade for model inference backends.

Example:
    >>> import torch
    >>> from pyTorchAutoForge.api.runtime import ModelRuntimeApi
    >>> runtime_api = ModelRuntimeApi()
    >>> # runtime_api.forward(backend="onnx", model_inputs=torch.randn(1, 3))  # Requires valid ONNX source.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

import numpy
import onnx
import torch

from pyTorchAutoForge.api.onnx.OnnxRuntimeApi import OnnxRuntimeApi
from pyTorchAutoForge.api.tensorrt.TensorrtRuntimeApi import TensorrtRuntimeApi


class RuntimeBackend(Enum):
    """Supported runtime backends for :class:`ModelRuntimeApi`."""

    ONNX = "onnx"
    TENSORRT = "tensorrt"

# TODO add Protocol for runtime API interface and type check concrete APIs against it? Only if different runtimes can be unified with single signature. Seems unlikely though
# TODO add typing types to avoid long signatures


class ModelRuntimeApi:
    """Unified runtime API that delegates to backend-specific runtime classes.

    Notes:
        The facade lazily initializes concrete backend APIs and keeps independent
        caches per backend.
    """

    def __init__(self) -> None:
        """Initialize runtime facade without creating backend sessions."""
        self._onnx_api: OnnxRuntimeApi | None = None
        self._tensorrt_api: TensorrtRuntimeApi | None = None

    def _get_onnx_api(self) -> OnnxRuntimeApi:
        """Get lazy-initialized ONNX runtime API."""
        if self._onnx_api is None:
            self._onnx_api = OnnxRuntimeApi()
        return self._onnx_api

    def _get_tensorrt_api(self) -> TensorrtRuntimeApi:
        """Get lazy-initialized TensorRT runtime API."""
        if self._tensorrt_api is None:
            self._tensorrt_api = TensorrtRuntimeApi()
        return self._tensorrt_api

    # PUBLIC API METHODS
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
                     onnx_model: onnx.ModelProto | None = None,
                     onnx_filepath: str | None = None,
                     ) -> dict[str, numpy.ndarray]:
        """Delegate ONNX inference to :class:`OnnxRuntimeApi`."""

        onnx_api = self._get_onnx_api()
        return onnx_api.forward(model_inputs=model_inputs,
                                input_tensor=input_tensor,
                                providers=providers,
                                provider_options=provider_options,
                                force_new_session=force_new_session,
                                session_options=session_options,
                                onnx_model=onnx_model,
                                onnx_filepath=onnx_filepath,
                                )

    def tensorrt_forward(self,
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
        """Delegate TensorRT inference to :class:`TensorrtRuntimeApi`."""

        # Call TensorRT runtime API forward method
        tensorrt_api = self._get_tensorrt_api()
        return tensorrt_api.forward(
            model_inputs=model_inputs,
            input_tensor=input_tensor,
            force_new_session=force_new_session,
            engine_bytes=engine_bytes,
            engine_filepath=engine_filepath,
            device=device,
        )

    def forward(self,
                backend: RuntimeBackend | Literal["onnx", "tensorrt"],
                **kwargs: Any,
                ) -> dict[str, numpy.ndarray]:
        """Dispatch inference call to selected backend.

        Args:
            backend: Runtime backend enum or value string.
            **kwargs: Keyword arguments passed to backend-specific forward method.

        Returns:
            Output dictionary keyed by backend output tensor names.

        Raises:
            ValueError: If backend value is unsupported.
        """

        # Dispatch call to required backend
        backend_value = backend.value if isinstance(
            backend, RuntimeBackend) else str(backend).lower()
        if backend_value == RuntimeBackend.ONNX.value:
            return self.onnx_forward(**kwargs)
        if backend_value == RuntimeBackend.TENSORRT.value:
            return self.tensorrt_forward(**kwargs)
        raise ValueError(
            f"Unsupported backend: {backend}. Expected one of: '{RuntimeBackend.ONNX.value}', '{RuntimeBackend.TENSORRT.value}'."
        )

    def clear_session_cache(self,
                            backend: Literal["onnx",
                                             "tensorrt", "all"] = "all",
                            ) -> None:
        """Clear cached sessions in selected backend(s).

        Args:
            backend: Backend cache selector. Supported values are ``onnx``,
                ``tensorrt``, and ``all``.

        Raises:
            ValueError: If backend selector is unsupported.
        """
        backend_value = str(backend).lower()
        if backend_value == "onnx":
            if self._onnx_api is not None:
                self._onnx_api.clear_session_cache()
            return

        if backend_value == "tensorrt":
            if self._tensorrt_api is not None:
                self._tensorrt_api.clear_session_cache()
            return

        if backend_value == "all":
            if self._onnx_api is not None:
                self._onnx_api.clear_session_cache()
            if self._tensorrt_api is not None:
                self._tensorrt_api.clear_session_cache()
            return

        raise ValueError(
            "Unsupported backend cache selector. Expected one of: 'onnx', 'tensorrt', 'all'."
        )
