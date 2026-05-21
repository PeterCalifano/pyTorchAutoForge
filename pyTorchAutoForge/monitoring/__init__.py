"""Experiment monitoring helpers with import-safe public exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .types import MetricInput, MetricSplit, MetricsAsDict

_LOGGER_EXPORTS: dict[str, str] = {
    "CompositeRunLogger": ".run_loggers",
    "MlflowRunLogger": ".run_loggers",
    "TensorBoardRunLogger": ".run_loggers",
}

__all__ = [
    "CompositeRunLogger",
    "MetricInput",
    "MetricSplit",
    "MetricsAsDict",
    "MlflowRunLogger",
    "TensorBoardRunLogger",
]


def __getattr__(name: str) -> Any:
    """Resolve optional logger exports only when callers request them."""
    module_name_ = _LOGGER_EXPORTS.get(name)
    if module_name_ is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    module_ = import_module(module_name_, package=__name__)
    export_ = getattr(module_, name)
    globals()[name] = export_
    return export_


def __dir__() -> list[str]:
    """Return dynamic dir listing including lazy logger exports."""
    return sorted(set(globals().keys()) | set(__all__))
