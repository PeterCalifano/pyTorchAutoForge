from dataclasses import dataclass, field
from typing import TypeAlias, Callable, Sequence, cast
import torch
import numpy as np
from enum import Enum

from pyTorchAutoForge.datasets.LabelsClasses import PTAF_Datakey

ValueType: TypeAlias = float | int | torch.Tensor | np.ndarray


class DistributionType(Enum):
    GAUSSIAN = 0
    POISSON = 1
    UNIFORM = 2
    NEG_EXPONENTIAL = 3
    # TODO add more distributions as needed


@dataclass(frozen=True, slots=True)
class GaussianParams:
    """Parameters for Gaussian noise."""

    mean: ValueType
    std: ValueType


@dataclass(frozen=True, slots=True)
class PoissonParams:
    """Parameters for Poisson noise."""

    rate: ValueType


@dataclass(frozen=True, slots=True)
class UniformParams:
    """Parameters for Uniform noise."""

    low: ValueType
    high: ValueType


@dataclass(frozen=True, slots=True)
class NegExponentialParams:
    """Parameters for NegExponential noise."""

    rate: ValueType


DistributionParams: TypeAlias = (
    GaussianParams | PoissonParams | UniformParams | NegExponentialParams
)


@dataclass(frozen=True, slots=True)
class Vector1dErrorModel:
    """
    Configuration for a 1D error model.

    If target_keys are provided, target_indices are interpreted as local indices
    within the concatenated target_keys slice. If target_keys are omitted,
    target_indices are interpreted as global indices in the full vector.
    """

    variable_name: str
    shape: tuple[int, ...]
    error_type: DistributionType
    parameters: DistributionParams
    target_keys: tuple[PTAF_Datakey | str, ...] | None = None
    target_indices: tuple[int, ...] | None = None

    # Define internal callable sampler functor
    _sampler: Callable[[tuple[int, ...], torch.device,
                        torch.dtype], torch.Tensor] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """
        __post_init__ _summary_

        _extended_summary_

        :raises ValueError: _description_
        :raises TypeError: _description_
        """
        type_map = {
            DistributionType.GAUSSIAN: GaussianParams,
            DistributionType.POISSON: PoissonParams,
            DistributionType.UNIFORM: UniformParams,
            DistributionType.NEG_EXPONENTIAL: NegExponentialParams,
        }

        # Verify parameters type and if matches error_type
        expected = type_map.get(self.error_type)
        if expected is None:
            raise ValueError(f"Unsupported error_type: {self.error_type}")

        if not isinstance(self.parameters, expected):
            raise TypeError(
                f"Expected parameters of type {expected.__name__} for {self.error_type}, "
                f"got {type(self.parameters).__name__}"
            )

        # Normalize target keys and indices
        if self.target_keys is not None:
            normalized = _normalize_datakeys(self.target_keys)
            object.__setattr__(self, "target_keys", normalized)

        if self.target_indices is not None:
            normalized = _normalize_indices(self.target_indices)
            object.__setattr__(self, "target_indices", normalized)

        # Validate parameters at init to surface configuration issues early.
        self._validate_params()
        object.__setattr__(self, "_sampler", self._bind_sampler())

    def sample(
        self,
        shape: tuple[int, ...] | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        sample_shape = self.shape if shape is None else shape
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float32

        # Materialize parameters at sample time to respect device/dtype.
        return self._sampler(sample_shape, device, dtype)

    def _bind_sampler(
        self,
    ) -> Callable[[tuple[int, ...], torch.device, torch.dtype], torch.Tensor]:
        if self.error_type == DistributionType.GAUSSIAN:
            return self._sample_gaussian
        if self.error_type == DistributionType.POISSON:
            return self._sample_poisson
        if self.error_type == DistributionType.UNIFORM:
            return self._sample_uniform
        if self.error_type == DistributionType.NEG_EXPONENTIAL:
            return self._sample_neg_exponential
        raise ValueError(f"Unsupported error_type: {self.error_type}")

    def _validate_params(self) -> None:
        params = self.parameters
        if isinstance(params, GaussianParams):
            _validate_value_type(params.mean, "mean")
            _validate_value_type(params.std, "std")
            _validate_non_negative(params.std, "std")
        elif isinstance(params, PoissonParams):
            _validate_value_type(params.rate, "rate")
            _validate_non_negative(params.rate, "rate")
        elif isinstance(params, UniformParams):
            _validate_value_type(params.low, "low")
            _validate_value_type(params.high, "high")
            _validate_low_high(params.low, params.high)
        elif isinstance(params, NegExponentialParams):
            _validate_value_type(params.rate, "rate")
            _validate_positive(params.rate, "rate")

    def _sample_gaussian(
        self, sample_shape: tuple[int, ...], device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        params = cast(GaussianParams, self.parameters)
        mean = _as_tensor(params.mean, device=device,
                          dtype=dtype).expand(sample_shape)
        std = _as_tensor(params.std, device=device,
                         dtype=dtype).expand(sample_shape)
        return torch.randn(sample_shape, device=device, dtype=dtype) * std + mean

    def _sample_poisson(
        self, sample_shape: tuple[int, ...], device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        params = cast(PoissonParams, self.parameters)
        rate = _as_tensor(params.rate, device=device,
                          dtype=dtype).expand(sample_shape)
        return torch.poisson(rate)

    def _sample_uniform(
        self, sample_shape: tuple[int, ...], device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        params = cast(UniformParams, self.parameters)
        low = _as_tensor(params.low, device=device,
                         dtype=dtype).expand(sample_shape)
        high = _as_tensor(params.high, device=device,
                          dtype=dtype).expand(sample_shape)
        return low + (high - low) * torch.rand(sample_shape, device=device, dtype=dtype)

    def _sample_neg_exponential(
        self, sample_shape: tuple[int, ...], device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        params = cast(NegExponentialParams, self.parameters)
        rate = _as_tensor(params.rate, device=device,
                          dtype=dtype).expand(sample_shape)
        dist = torch.distributions.NegExponential(rate)
        return dist.sample()


@dataclass(frozen=True, slots=True)
class Vector1dErrorStackModel:
    """
    Apply multiple Vector1dErrorModel instances to a keyed 1D vector.

    The keys define the layout of the 1D vector. When keys are PTAF_Datakey,
    their sizes are inferred and validated. Optional key_sizes can be provided
    to validate (or define) sizes per key.
    """

    keys: tuple[PTAF_Datakey | str, ...]
    error_models: tuple[Vector1dErrorModel, ...]
    key_sizes: tuple[int, ...] | None = None
    strict_key_check: bool = True
    _key_slices: dict[PTAF_Datakey, slice] = field(init=False, repr=False)
    _total_size: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        normalized_keys = _normalize_datakeys(self.keys)
        object.__setattr__(self, "keys", normalized_keys)

        if len(set(normalized_keys)) != len(normalized_keys):
            raise ValueError("Keys must be unique.")

        sizes = _resolve_key_sizes(normalized_keys, self.key_sizes)
        key_slices, total_size = _build_key_slices(normalized_keys, sizes)
        object.__setattr__(self, "_key_slices", key_slices)
        object.__setattr__(self, "_total_size", total_size)

        if self.strict_key_check:
            for model in self.error_models:
                if model.target_keys is None:
                    continue
                unknown = [k for k in model.target_keys if k not in key_slices]
                if unknown:
                    raise KeyError(
                        f"Unknown target_keys in error model: {unknown}")

        for model in self.error_models:
            target_size = self._target_size_for_model(model)
            _validate_target_indices(model.target_indices, target_size)

    def apply(
        self, values: torch.Tensor, return_error: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if values.shape[-1] != self._total_size:
            raise ValueError(
                f"Expected last dimension size {self._total_size} based on keys, "
                f"got {values.shape[-1]}"
            )

        # Accumulate per-model noise to support combined error sources.
        total_error = torch.zeros_like(values)
        for model in self.error_models:
            indices = self._resolve_indices(model)
            if not indices:
                continue
            target_shape = values[..., indices].shape
            noise = model.sample(
                target_shape, device=values.device, dtype=values.dtype)
            total_error[..., indices] += noise

        output = values + total_error
        return (output, total_error) if return_error else output

    def _resolve_indices(self, model: Vector1dErrorModel) -> list[int]:
        if model.target_keys is None:
            if model.target_indices is None:
                return list(range(self._total_size))
            return list(model.target_indices)

        keys = self._filter_keys(model.target_keys)
        key_indices = self._indices_from_keys(keys)
        if model.target_indices is None:
            return key_indices

        # target_indices are local to the target_keys slice.
        return [key_indices[i] for i in model.target_indices]

    def _filter_keys(self, keys: tuple[PTAF_Datakey, ...]) -> tuple[PTAF_Datakey, ...]:
        if self.strict_key_check:
            return keys
        return tuple(k for k in keys if k in self._key_slices)

    def _target_size_for_model(self, model: Vector1dErrorModel) -> int:
        if model.target_keys is None:
            return self._total_size
        keys = self._filter_keys(model.target_keys)
        return sum(
            self._key_slices[key].stop - self._key_slices[key].start
            for key in keys
        )

    def _indices_from_keys(self, keys: tuple[PTAF_Datakey, ...]) -> list[int]:
        indices: list[int] = []
        for key in keys:
            key_slice = self._key_slices[key]
            indices.extend(range(key_slice.start, key_slice.stop))
        return indices


def _normalize_datakeys(keys: Sequence[PTAF_Datakey | str]) -> tuple[PTAF_Datakey, ...]:
    """
    _normalize_datakeys _summary_

    _extended_summary_

    :param keys: _description_
    :type keys: Sequence[PTAF_Datakey  |  str]
    :raises ValueError: _description_
    :raises TypeError: _description_
    :return: _description_
    :rtype: tuple[PTAF_Datakey, ...]
    """

    normalized: list[PTAF_Datakey] = []
    # Iterate over input keys and normalize
    for key in keys:
        if isinstance(key, PTAF_Datakey):
            normalized.append(key)
        elif isinstance(key, str):
            try:
                normalized.append(PTAF_Datakey[key.upper()])
            except KeyError as exc:
                raise ValueError(f"Unknown PTAF_Datakey: {key}") from exc
        else:
            raise TypeError(
                f"Keys must be PTAF_Datakey or str, got {type(key).__name__}"
            )
    return tuple(normalized)


def _normalize_indices(indices: Sequence[int]) -> tuple[int, ...]:
    normalized: list[int] = []
    for idx in indices:
        if not isinstance(idx, (int, np.integer)):
            raise TypeError(
                f"Indices must be int values, got {type(idx).__name__}"
            )
        normalized.append(int(idx))
    return tuple(normalized)


def _resolve_key_sizes(
    keys: tuple[PTAF_Datakey, ...],
    key_sizes: tuple[int, ...] | None,
) -> tuple[int, ...]:
    if key_sizes is not None and len(key_sizes) != len(keys):
        raise ValueError("key_sizes length must match keys length.")

    resolved: list[int] = []
    for idx, key in enumerate(keys):
        declared_size = key.get_lbl_vector_size()
        override = key_sizes[idx] if key_sizes is not None else None

        if override is not None:
            if not isinstance(override, int) or isinstance(override, bool):
                raise TypeError("key_sizes values must be int.")
            if override <= 0:
                raise ValueError("key_sizes values must be positive.")

        if declared_size is None or declared_size <= 0:
            if override is None:
                raise ValueError(
                    f"Size for datakey {key.name} must be provided via key_sizes."
                )
            resolved.append(override)
            continue

        if override is not None and override != declared_size:
            raise ValueError(
                f"Key size mismatch for {key.name}: expected {declared_size}, got {override}."
            )
        resolved.append(declared_size)

    return tuple(resolved)


def _build_key_slices(
    keys: tuple[PTAF_Datakey, ...], sizes: tuple[int, ...]
) -> tuple[dict[PTAF_Datakey, slice], int]:
    key_slices: dict[PTAF_Datakey, slice] = {}
    offset = 0
    for key, size in zip(keys, sizes):
        key_slices[key] = slice(offset, offset + size)
        offset += size
    return key_slices, offset


def _validate_target_indices(indices: tuple[int, ...] | None, allowed_size: int) -> None:
    if indices is None:
        return
    if len(indices) > allowed_size:
        raise ValueError(
            "Number of target indices cannot exceed the size for the selected keys."
        )
    if not indices:
        return
    if any(idx < 0 for idx in indices):
        raise ValueError("Target indices must be non-negative.")
    max_idx = max(indices)
    if max_idx >= allowed_size:
        raise ValueError(
            "Target indices must be within the vector size derived from keys."
        )


def _validate_value_type(value: ValueType, name: str) -> None:
    if not isinstance(value, (int, float, torch.Tensor, np.ndarray)):
        raise TypeError(
            f"{name} must be a float/int, torch.Tensor, or np.ndarray, got {type(value).__name__}"
        )


def _validate_non_negative(value: ValueType, name: str) -> None:
    if isinstance(value, torch.Tensor):
        if not torch.all(value >= 0).item():
            raise ValueError(f"{name} must be non-negative.")
    elif isinstance(value, np.ndarray):
        if not np.all(value >= 0):
            raise ValueError(f"{name} must be non-negative.")
    else:
        if value < 0:
            raise ValueError(f"{name} must be non-negative.")


def _validate_positive(value: ValueType, name: str) -> None:
    if isinstance(value, torch.Tensor):
        if not torch.all(value > 0).item():
            raise ValueError(f"{name} must be positive.")
    elif isinstance(value, np.ndarray):
        if not np.all(value > 0):
            raise ValueError(f"{name} must be positive.")
    else:
        if value <= 0:
            raise ValueError(f"{name} must be positive.")


def _validate_low_high(low: ValueType, high: ValueType) -> None:
    if isinstance(low, torch.Tensor) or isinstance(high, torch.Tensor):
        low_t = _as_tensor(low, device=torch.device(
            "cpu"), dtype=torch.float32)
        high_t = _as_tensor(high, device=torch.device(
            "cpu"), dtype=torch.float32)
        if not torch.all(low_t <= high_t).item():
            raise ValueError("low must be <= high.")
        return

    if isinstance(low, np.ndarray) or isinstance(high, np.ndarray):
        low_a = np.asarray(low)
        high_a = np.asarray(high)
        if not np.all(low_a <= high_a):
            raise ValueError("low must be <= high.")
        return

    if low > high:
        raise ValueError("low must be <= high.")


def _as_tensor(value: ValueType, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)
