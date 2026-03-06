from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_root_import_is_lazy_for_heavy_submodules() -> None:
    command_ = [
        sys.executable,
        "-c",
        (
            "import pyTorchAutoForge, sys; "
            "print(int('pyTorchAutoForge.utils.utils' in sys.modules))"
        ),
    ]
    result_ = subprocess.run(
        command_,
        cwd=str(REPO_ROOT),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result_.returncode == 0, result_.stderr
    assert result_.stdout.strip() == "0"


def test_root_getattr_warns_and_raises_on_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyTorchAutoForge

    pyTorchAutoForge.__dict__.pop("api", None)
    import_module_original_ = pyTorchAutoForge.import_module

    def fake_import_module(module_name: str, package: str | None = None):
        if module_name == ".api" and package == "pyTorchAutoForge":
            raise ImportError("No module named 'mlflow'")
        return import_module_original_(module_name, package=package)

    monkeypatch.setattr(pyTorchAutoForge, "import_module", fake_import_module)

    with pytest.warns(pyTorchAutoForge.OptionalDependencyImportWarning):
        with pytest.raises(ImportError, match="pyTorchAutoForge.api"):
            _ = pyTorchAutoForge.api


def test_api_getattr_warns_and_raises_on_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyTorchAutoForge.api

    pyTorchAutoForge.api.__dict__.pop("StartMLflowUI", None)
    import_module_original_ = pyTorchAutoForge.api.import_module

    def fake_import_module(module_name: str, package: str | None = None):
        if module_name == ".mlflow" and package == "pyTorchAutoForge.api":
            raise ImportError("No module named 'mlflow'")
        return import_module_original_(module_name, package=package)

    monkeypatch.setattr(pyTorchAutoForge.api, "import_module", fake_import_module)

    with pytest.warns(pyTorchAutoForge.api.OptionalDependencyImportWarning):
        with pytest.raises(ImportError, match="pyTorchAutoForge.api.StartMLflowUI"):
            _ = pyTorchAutoForge.api.StartMLflowUI
