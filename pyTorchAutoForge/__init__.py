"""Top-level package exports resolved lazily at access time."""

from __future__ import annotations

import ast
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
import warnings


class OptionalDependencyImportWarning(UserWarning):
    """Warning emitted when a lazily loaded symbol needs a missing dependency."""


_SUBPACKAGES: tuple[str, ...] = (
    "utils",
    "optimization",
    "model_building",
    "hparams_optim",
    "api",
    "evaluation",
    "datasets",
    "setup",
)
_PACKAGE_DIR = Path(__file__).resolve().parent
_WARNED_IMPORTS: set[tuple[str, str]] = set()


def _extract_all_symbols(init_file: Path) -> tuple[str, ...]:
    if not init_file.exists():
        return ()

    try:
        parsed_file_ = ast.parse(init_file.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return ()

    for statement_ in parsed_file_.body:
        if not isinstance(statement_, ast.Assign):
            continue

        for target_ in statement_.targets:
            if isinstance(target_, ast.Name) and target_.id == "__all__":
                try:
                    exported_names_ = ast.literal_eval(statement_.value)
                except (ValueError, TypeError):
                    return ()

                if isinstance(exported_names_, (list, tuple)) and all(
                    isinstance(item_, str) for item_ in exported_names_
                ):
                    return tuple(exported_names_)
                return ()

    return ()


def _build_symbol_map() -> dict[str, str]:
    symbol_to_subpackage_: dict[str, str] = {}
    for subpackage_ in _SUBPACKAGES:
        symbol_to_subpackage_[subpackage_] = subpackage_
        exported_symbols_ = _extract_all_symbols(
            _PACKAGE_DIR / subpackage_ / "__init__.py"
        )
        for symbol_ in exported_symbols_:
            symbol_to_subpackage_[symbol_] = subpackage_

    return symbol_to_subpackage_


_SYMBOL_TO_SUBPACKAGE = _build_symbol_map()
__all__ = sorted(_SYMBOL_TO_SUBPACKAGE.keys())

try:
    __version__ = version("pyTorchAutoForge")
except PackageNotFoundError:
    __version__ = "unknown"


def _warn_missing_dependency_once(
    *,
    symbol_name: str,
    subpackage_name: str,
    import_error: ImportError,
) -> None:
    missing_dep_name_ = getattr(import_error, "name", "") or "<unknown>"
    warning_key_ = (symbol_name, missing_dep_name_)
    if warning_key_ in _WARNED_IMPORTS:
        return

    _WARNED_IMPORTS.add(warning_key_)
    warnings.warn(
        "Optional dependency missing while resolving "
        f"pyTorchAutoForge.{symbol_name} from subpackage "
        f"pyTorchAutoForge.{subpackage_name}: {import_error}",
        category=OptionalDependencyImportWarning,
        stacklevel=2,
    )


def __getattr__(name: str) -> Any:
    subpackage_name_ = _SYMBOL_TO_SUBPACKAGE.get(name)
    if subpackage_name_ is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    try:
        target_module_ = import_module(f".{subpackage_name_}", package=__name__)
        value_ = target_module_ if name == subpackage_name_ else getattr(target_module_, name)
    except ImportError as import_error_:
        _warn_missing_dependency_once(
            symbol_name=name,
            subpackage_name=subpackage_name_,
            import_error=import_error_,
        )
        raise ImportError(
            f"Cannot access 'pyTorchAutoForge.{name}' because importing "
            f"'pyTorchAutoForge.{subpackage_name_}' failed. Install required "
            f"dependencies for this feature. Original error: {import_error_}"
        ) from import_error_

    globals()[name] = value_
    return value_


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__) | {"__version__"})
