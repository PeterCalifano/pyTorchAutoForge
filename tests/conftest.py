import os
import shutil
from pathlib import Path

import pytest

# Force a non-interactive plotting backend unless explicitly enabled.
if os.getenv("PTAF_ENABLE_TEST_PLOTS") != "1":
    import matplotlib

    matplotlib.use("Agg", force=True)


REPO_ROOT = Path(__file__).resolve().parent.parent
LEGACY_TEST_OUTPUT_DIRS = (
    REPO_ROOT / "test_output",
    REPO_ROOT / "tests" / "test_output",
    REPO_ROOT / "tests" / "model_building" / "test_output",
)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run tests marked slow.",
    )
    parser.addoption(
        "--run-gpu",
        action="store_true",
        default=False,
        help="Run tests marked gpu.",
    )
    parser.addoption(
        "--run-visual",
        action="store_true",
        default=False,
        help="Run tests marked visual.",
    )


def pytest_collection_modifyitems(config: pytest.Config,
                                  items: list[pytest.Item],
                                  ) -> None:
    marker_options_ = {
        "slow": ("--run-slow", "slow test skipped by default"),
        "gpu": ("--run-gpu", "GPU test skipped by default"),
        "visual": ("--run-visual", "visual test skipped by default"),
    }

    for item_ in items:
        for marker_name_, (option_name_, reason_) in marker_options_.items():
            if marker_name_ in item_.keywords and not config.getoption(option_name_):
                item_.add_marker(pytest.mark.skip(reason=reason_))
                break


def _cleanup_legacy_output_dirs() -> None:
    repo_root_resolved_ = REPO_ROOT.resolve()
    for output_dir_ in LEGACY_TEST_OUTPUT_DIRS:
        resolved_output_dir_ = output_dir_.resolve()
        if not resolved_output_dir_.is_relative_to(repo_root_resolved_):
            continue
        if output_dir_.exists():
            shutil.rmtree(output_dir_)


@pytest.fixture(scope="session", autouse=True)
def _cleanup_legacy_output_dirs_session() -> None:
    _cleanup_legacy_output_dirs()
    yield
    _cleanup_legacy_output_dirs()


@pytest.fixture(autouse=True)
def _disable_plot_show_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    if os.getenv("PTAF_ENABLE_TEST_PLOTS") == "1":
        return

    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
