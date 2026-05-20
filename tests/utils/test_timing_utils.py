from __future__ import annotations

from itertools import count

import pytest

import pyTorchAutoForge.utils.timing_utils as timing_utils
from pyTorchAutoForge.utils import timeit_averaged, timeit_averaged_


def _PatchPerfCounter(monkeypatch: pytest.MonkeyPatch,
                      step: float = 0.01,
                      ) -> None:
    counter_ = count()

    def FakePerfCounter() -> float:
        return next(counter_) * step

    monkeypatch.setattr(timing_utils.time, "perf_counter", FakePerfCounter)


def test_timeit_averaged_returns_wrapped_result(monkeypatch: pytest.MonkeyPatch) -> None:
    _PatchPerfCounter(monkeypatch)
    calls_: list[int] = []

    @timeit_averaged(2)
    def SampleFunction(value_: int) -> int:
        calls_.append(value_)
        return value_ + 1

    assert SampleFunction(4) == 5
    assert calls_ == [4, 4]


def test_timeit_averaged_preserves_function_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    _PatchPerfCounter(monkeypatch)

    @timeit_averaged(1)
    def SampleFunction() -> str:
        return "ok"

    assert SampleFunction.__name__ == "SampleFunction"
    assert SampleFunction() == "ok"


def test_timeit_averaged_function() -> None:
    def SampleFunction(x_: int, y_: int) -> int:
        return x_ + y_

    average_time_ = timeit_averaged_(SampleFunction, 3, 2, 3)

    assert isinstance(average_time_, float)
    assert average_time_ >= 0.0


def test_timeit_averaged_function_with_kwargs() -> None:
    def SampleFunction(x_: int, y_: int, scale_: int = 1) -> int:
        return scale_ * x_ * y_

    average_time_ = timeit_averaged_(SampleFunction, 2, 3, 4, scale_=2)

    assert isinstance(average_time_, float)
    assert average_time_ >= 0.0
