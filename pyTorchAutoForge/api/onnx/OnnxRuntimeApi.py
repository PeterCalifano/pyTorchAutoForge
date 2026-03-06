"""ONNX Runtime inference API.

Example:
    >>> import onnx
    >>> import torch
    >>> from pyTorchAutoForge.api.onnx import OnnxRuntimeApi
    >>> runtime_api = OnnxRuntimeApi(onnx_model=onnx.ModelProto())
    >>> # runtime_api.forward(model_inputs=torch.randn(1, 3))  # Requires valid ONNX graph.
"""

import hashlib
import os
import warnings
from typing import Any

import numpy
import onnx
import torch

from pyTorchAutoForge.utils import torch_to_numpy

_UNSET_SOURCE = object()

# TODO use typing types instead of very long lists in signatures


class OnnxRuntimeApi:
    """Runtime helper for ONNX inference through ONNX Runtime.

    This class encapsulates ONNX Runtime provider selection, session creation,
    session caching, and input/output normalization.

    Notes:
        Source resolution order for :meth:`forward` is:
        1. Explicit ``onnx_model`` argument.
        2. Stored ``self.onnx_model``.
        3. Explicit ``onnx_filepath`` argument.
        4. Stored ``self.onnx_filepath``.
    """

    def __init__(self,
                 onnx_model: onnx.ModelProto | None = None,
                 onnx_filepath: str | None = None,
                 default_providers: list[str] | tuple[str, ...] | None = None,
                 ) -> None:
        """Initialize ONNX runtime API.

        Args:
            onnx_model: Optional in-memory ONNX model source.
            onnx_filepath: Optional path-based ONNX source.
            default_providers: Default execution providers order. If omitted,
                uses `["CPUExecutionProvider"]`.

        Notes:
            Session cache keys include source identity, providers, provider
            options, and session options signature.
        """
        self.onnx_model = onnx_model
        self.onnx_filepath = onnx_filepath
        self.default_providers = list(default_providers) if default_providers is not None else [
            "CPUExecutionProvider"
        ]
        self._session_cache: dict[tuple[Any, ...], Any] = {}
        self._session_meta: dict[tuple[Any, ...], dict[str, Any]] = {}

    def _resolve_source_payload(self,
                                onnx_model: onnx.ModelProto | None = None,
                                onnx_filepath: str | None = None,
                                ) -> tuple[str, bytes | str]:
        """Resolve inference source payload in priority order.

        Priority:
            1. Explicit `onnx_model` argument.
            2. Stored `self.onnx_model`.
            3. Explicit `onnx_filepath` argument.
            4. Stored `self.onnx_filepath`.

        Args:
            onnx_model: Optional model override.
            onnx_filepath: Optional filepath override.

        Returns:
            Tuple of `(source_kind, source_payload)`, where `source_kind` is
            `"proto"` or `"file"`.

        Raises:
            ValueError: If no usable ONNX source is available.
        """
        model_source = onnx_model if onnx_model is not None else self.onnx_model
        if model_source is not None:
            return "proto", model_source.SerializeToString()

        filepath_source = onnx_filepath if onnx_filepath is not None else self.onnx_filepath
        if filepath_source is not None and os.path.isfile(filepath_source):
            return "file", os.path.abspath(filepath_source)

        raise ValueError(
            "No ONNX source available for runtime inference. Provide onnx_model or a valid onnx_filepath."
        )

    def _resolve_requested_providers(self,
                                     providers: list[str] | tuple[str, ...] | None,
                                     ) -> tuple[list[str], tuple[str, ...]]:
        """Resolve ordered execution providers with availability filtering.

        Args:
            providers: Requested providers priority list.

        Returns:
            Tuple `(active_providers, available_providers)`.

        Raises:
            ValueError: If none of the requested providers are available.
        """
        import onnxruntime as ort

        available = tuple(ort.get_available_providers())
        requested = list(providers) if providers is not None else list(
            self.default_providers)
        if len(requested) == 0:
            requested = ["CPUExecutionProvider"]

        missing = [
            provider for provider in requested if provider not in available]
        if missing:
            warnings.warn(
                f"Requested providers not available and will be skipped: {missing}",
                RuntimeWarning,
                stacklevel=2,
            )

        active = [provider for provider in requested if provider in available]
        if len(active) == 0:
            raise ValueError(
                f"No usable execution providers found. Requested={requested}, available={list(available)}"
            )
        return active, available

    def _resolve_provider_options(self,
                                  active_providers: list[str],
                                  provider_options: dict[str, dict[str, Any]] | None,
                                  ) -> list[dict[str, Any]] | None:
        """Build provider-options list aligned to active providers.

        Args:
            active_providers: Ordered provider list selected for inference.
            provider_options: Mapping provider -> provider options.

        Returns:
            Provider options list aligned with providers, or `None` if empty.

        Raises:
            TypeError: If provider_options is not a dictionary.
        """
        if provider_options is None:
            return None
        if not isinstance(provider_options, dict):
            raise TypeError(
                "provider_options must be a dictionary when provided.")

        aligned_options: list[dict[str, Any]] = []
        any_non_empty = False
        for provider in active_providers:
            options = provider_options.get(provider, {})
            if not isinstance(options, dict):
                raise TypeError(
                    f"provider_options['{provider}'] must be a dictionary.")
            if len(options) > 0:
                any_non_empty = True
            aligned_options.append(options)

        return aligned_options if any_non_empty else None

    def _provider_options_signature(self,
                                    active_providers: list[str],
                                    aligned_provider_options: list[dict[str, Any]] | None,
                                    ) -> tuple[tuple[str, tuple[tuple[str, str], ...]], ...]:
        """Build hashable provider options signature for session cache keys."""
        if aligned_provider_options is None:
            return tuple((provider, tuple()) for provider in active_providers)

        signature_items = []
        for provider, options in zip(active_providers, aligned_provider_options):
            option_items = tuple(sorted((str(key), repr(value))
                                 for key, value in options.items()))
            signature_items.append((provider, option_items))
        return tuple(signature_items)

    def _build_session_cache_key(self,
                                 source_kind: str,
                                 source_payload: bytes | str,
                                 active_providers: list[str],
                                 aligned_provider_options: list[dict[str, Any]] | None,
                                 session_options: Any | None,
                                 ) -> tuple[Any, ...]:
        """Build deterministic cache key for runtime session reuse."""
        if source_kind == "proto":
            source_digest = hashlib.sha256(source_payload).hexdigest()
            source_identity = ("proto", source_digest)
        else:
            stat = os.stat(source_payload)
            source_identity = ("file", source_payload,
                               stat.st_mtime_ns, stat.st_size)

        provider_signature = tuple(active_providers)
        options_signature = self._provider_options_signature(
            active_providers, aligned_provider_options
        )
        session_signature = repr(session_options)
        return (source_identity, provider_signature, options_signature, session_signature)

    def _create_session(self,
                        source_payload: bytes | str,
                        active_providers: list[str],
                        aligned_provider_options: list[dict[str, Any]] | None,
                        session_options: Any | None,
                        ):
        """Instantiate ONNX Runtime session from resolved source."""
        import onnxruntime as ort

        session_kwargs: dict[str, Any] = {
            "providers": active_providers,
        }
        if aligned_provider_options is not None:
            session_kwargs["provider_options"] = aligned_provider_options
        if session_options is not None:
            session_kwargs["sess_options"] = session_options

        return ort.InferenceSession(source_payload, **session_kwargs)

    def _to_numpy_input(self, value: torch.Tensor | numpy.ndarray) -> numpy.ndarray:
        """Convert supported runtime input value to numpy array."""
        if isinstance(value, (torch.Tensor, numpy.ndarray)):
            return torch_to_numpy(value)
        raise TypeError(f"Unsupported input type: {type(value)}")

    def _build_input_feed(self,
                          session,
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
        """Normalize user inputs to ONNX Runtime input-feed format.

        Args:
            session: ONNX Runtime inference session.
            model_inputs: Runtime inputs in dict, tuple/list, or single tensor
                format.
            input_tensor: Deprecated alias for ``model_inputs``.

        Returns:
            Input feed mapping keyed by ONNX input names.

        Raises:
            ValueError: If inputs are missing, conflicting, or incompatible with
                model input arity/names.
            TypeError: If values cannot be converted to numpy arrays.
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
                "model_inputs is required for ONNX runtime forward.")

        input_infos = session.get_inputs()
        input_names = [input_info.name for input_info in input_infos]

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
                    f"Expected {len(input_names)} inputs, got {len(model_inputs)}."
                )
            return {
                input_names[index]: self._to_numpy_input(value)
                for index, value in enumerate(model_inputs)
            }

        if len(input_names) != 1:
            raise ValueError(
                f"Single input value provided but model expects {len(input_names)} inputs."
            )
        return {input_names[0]: self._to_numpy_input(model_inputs)}

    def set_onnx_source(self,
                        onnx_model: onnx.ModelProto | None | object = _UNSET_SOURCE,
                        onnx_filepath: str | None | object = _UNSET_SOURCE,
                        ) -> None:
        """Update ONNX source references for subsequent inferences.

        Args:
            onnx_model: Optional in-memory ONNX model. If explicitly set to
                ``None``, the stored model source is cleared.
            onnx_filepath: Optional ONNX filepath. If explicitly set to
                ``None``, the stored filepath source is cleared.
        """
        if onnx_model is not _UNSET_SOURCE:
            self.onnx_model = onnx_model
        if onnx_filepath is not _UNSET_SOURCE:
            self.onnx_filepath = onnx_filepath

    # PUBLIC API METHODS
    def clear_session_cache(self) -> None:
        """Clear all cached ONNX Runtime sessions.

        Returns:
            ``None``.
        """
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
                providers: list[str] | tuple[str, ...] | None = None,
                provider_options: dict[str, dict[str, Any]] | None = None,
                force_new_session: bool = False,
                session_options: Any | None = None,
                onnx_model: onnx.ModelProto | None = None,
                onnx_filepath: str | None = None,) -> dict[str, numpy.ndarray]:
        """Run ONNX Runtime forward inference.

        Args:
            model_inputs: Runtime model inputs in tensor, tuple/list, or dict form.
            input_tensor: Deprecated alias for `model_inputs`.
            providers: Ordered execution providers preference list.
            provider_options: Optional provider-specific options.
            force_new_session: Whether to bypass and refresh session cache.
            session_options: Optional ORT session options object.
            onnx_model: Optional ONNX model override source.
            onnx_filepath: Optional ONNX filepath override source.

        Returns:
            Output dictionary keyed by ONNX output names.

        Raises:
            ValueError: If no ONNX source is available or providers are unusable.
            TypeError: If provider options or model input types are invalid.

        Notes:
            Requested providers are filtered against
            ``onnxruntime.get_available_providers()`` while keeping order.
            Missing providers are skipped with ``RuntimeWarning`` and execution
            continues if at least one provider remains.
        """

        # Get onnx source payload and model
        source_kind, source_payload = self._resolve_source_payload(
            onnx_model=onnx_model, onnx_filepath=onnx_filepath
        )

        # Resolve providers and provider options
        active_providers, _ = self._resolve_requested_providers(providers)
        aligned_provider_options = self._resolve_provider_options(
            active_providers, provider_options
        )

        # Build session cache key and retrieve or create session
        cache_key = self._build_session_cache_key(source_kind=source_kind,
                                                  source_payload=source_payload,
                                                  active_providers=active_providers,
                                                  aligned_provider_options=aligned_provider_options,
                                                  session_options=session_options,
                                                  )

        if (not force_new_session) and (cache_key in self._session_cache):
            # Use cached session
            session = self._session_cache[cache_key]
        else:
            # Initialize new session
            session = self._create_session(source_payload=source_payload,
                                           active_providers=active_providers,
                                           aligned_provider_options=aligned_provider_options,
                                           session_options=session_options,
                                           )

            self._session_cache[cache_key] = session
            self._session_meta[cache_key] = {
                "source_kind": source_kind,
                "providers": tuple(active_providers),
                "provider_options": aligned_provider_options,
            }

        # Assemble input
        input_feed = self._build_input_feed(session=session,
                                            model_inputs=model_inputs,
                                            input_tensor=input_tensor,
                                            )

        # Run inference
        output_values = session.run(None, input_feed)

        output_names = [output_info.name
                        for output_info in session.get_outputs()]

        # Return output
        return {name: value for name, value in zip(output_names, output_values)}
