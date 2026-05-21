"""Shared monitoring type definitions."""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol, TypeAlias, runtime_checkable
from collections.abc import Mapping

class MetricSplit(Enum):
    """Common metric namespace prefixes."""

    TRAIN = "train"
    VALID = "valid"
    EVAL = "eval"


@runtime_checkable
class MetricsAsDict(Protocol):
    """Protocol for metric containers accepted by run loggers."""

    def as_dict(self) -> Mapping[str, Any]:
        """Return metric values keyed by metric name."""


MetricInput: TypeAlias = Mapping[str, Any] | MetricsAsDict
"""Metric payload accepted by monitoring loggers."""
