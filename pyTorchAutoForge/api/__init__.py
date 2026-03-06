"""API namespace exports resolved lazily at access time."""

from __future__ import annotations

from importlib import import_module
from typing import Any
import warnings


class OptionalDependencyImportWarning(UserWarning):
    """Warning emitted when an API symbol needs a missing optional dependency."""


_SYMBOL_TO_MODULE: dict[str, str] = {
    "ModelHandlerONNx": ".onnx",
    "DataProcessor": ".tcp",
    "pytcp_server": ".tcp",
    "pytcp_requestHandler": ".tcp",
    "ProcessingMode": ".tcp",
    "LoadModel": ".torch",
    "SaveModel": ".torch",
    "LoadDataset": ".torch",
    "SaveDataset": ".torch",
    "AutoForgeModuleSaveMode": ".torch",
    "StartMLflowUI": ".mlflow",
    "TorchModelMATLABwrapper": ".matlab",
}
_SUBMODULES: tuple[str, ...] = ("onnx", "tcp", "torch", "mlflow", "matlab")
_WARNED_IMPORTS: set[tuple[str, str]] = set()

__all__ = [
    "LoadModel",
    "SaveModel",
    "ModelHandlerONNx",
    "LoadDataset",
    "SaveDataset",
    "StartMLflowUI",
    "ModelRuntimeApi",
    "TorchModelMATLABwrapper",
    "DataProcessor",
    "pytcp_server",
    "pytcp_requestHandler",
    "ProcessingMode",
    "AutoForgeModuleSaveMode",
]


def _warn_missing_dependency_once(
    *,
    symbol_name: str,
    module_path: str,
    import_error: ImportError,
) -> None:
    missing_dep_name_ = getattr(import_error, "name", "") or "<unknown>"
    warning_key_ = (symbol_name, missing_dep_name_)
    if warning_key_ in _WARNED_IMPORTS:
        return

    _WARNED_IMPORTS.add(warning_key_)
    warnings.warn(
        "Optional dependency missing while resolving "
        f"pyTorchAutoForge.api.{symbol_name} from "
        f"pyTorchAutoForge.api{module_path}: {import_error}",
        category=OptionalDependencyImportWarning,
        stacklevel=2,
    )


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        module_ = import_module(f".{name}", package=__name__)
        globals()[name] = module_
        return module_

    target_module_path_ = _SYMBOL_TO_MODULE.get(name)
    if target_module_path_ is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    try:
        target_module_ = import_module(target_module_path_, package=__name__)
        value_ = getattr(target_module_, name)
    except ImportError as import_error_:
        _warn_missing_dependency_once(
            symbol_name=name,
            module_path=target_module_path_,
            import_error=import_error_,
        )
        raise ImportError(
            f"Cannot access 'pyTorchAutoForge.api.{name}' because importing "
            f"'pyTorchAutoForge.api{target_module_path_}' failed. Install required "
            f"dependencies for this feature. Original error: {import_error_}"
        ) from import_error_

    globals()[name] = value_
    return value_


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__) | set(_SUBMODULES))
