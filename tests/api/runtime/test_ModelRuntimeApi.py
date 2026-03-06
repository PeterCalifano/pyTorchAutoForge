from __future__ import annotations

import numpy
import onnx
import pytest
import torch

from pyTorchAutoForge.api.runtime import ModelRuntimeApi, RuntimeBackend


class FakeOnnxApi:
    def __init__(self) -> None:
        self.forward_calls: list[dict] = []
        self.clear_cache_calls = 0

    def forward(self, **kwargs):
        self.forward_calls.append(kwargs)
        return {"onnx_output": numpy.zeros((1,), dtype=numpy.float32)}

    def clear_session_cache(self):
        self.clear_cache_calls += 1


class FakeTensorrtApi:
    def __init__(self) -> None:
        self.forward_calls: list[dict] = []
        self.clear_cache_calls = 0

    def forward(self, **kwargs):
        self.forward_calls.append(kwargs)
        return {"trt_output": numpy.zeros((1,), dtype=numpy.float32)}

    def clear_session_cache(self):
        self.clear_cache_calls += 1


def test_model_runtime_api_onnx_forward_delegates_to_onnx_api() -> None:
    runtime_api = ModelRuntimeApi()
    fake_onnx_api = FakeOnnxApi()
    runtime_api._onnx_api = fake_onnx_api

    test_model = onnx.ModelProto()
    output_map = runtime_api.onnx_forward(
        model_inputs=torch.randn(1, 3),
        onnx_model=test_model,
    )

    assert "onnx_output" in output_map
    assert len(fake_onnx_api.forward_calls) == 1
    assert fake_onnx_api.forward_calls[0]["onnx_model"] is test_model


def test_model_runtime_api_tensorrt_forward_delegates_to_tensorrt_api() -> None:
    runtime_api = ModelRuntimeApi()
    fake_tensorrt_api = FakeTensorrtApi()
    runtime_api._tensorrt_api = fake_tensorrt_api

    output_map = runtime_api.tensorrt_forward(
        model_inputs=torch.randn(1, 3),
        engine_bytes=b"serialized_engine",
    )

    assert "trt_output" in output_map
    assert len(fake_tensorrt_api.forward_calls) == 1
    assert fake_tensorrt_api.forward_calls[0]["engine_bytes"] == b"serialized_engine"


def test_model_runtime_api_forward_dispatch_with_enum_and_string() -> None:
    runtime_api = ModelRuntimeApi()
    fake_onnx_api = FakeOnnxApi()
    fake_tensorrt_api = FakeTensorrtApi()
    runtime_api._onnx_api = fake_onnx_api
    runtime_api._tensorrt_api = fake_tensorrt_api

    output_onnx = runtime_api.forward(
        backend=RuntimeBackend.ONNX,
        model_inputs=torch.randn(1, 3),
        onnx_model=onnx.ModelProto(),
    )
    output_tensorrt = runtime_api.forward(
        backend="tensorrt",
        model_inputs=torch.randn(1, 3),
        engine_bytes=b"serialized_engine",
    )

    assert "onnx_output" in output_onnx
    assert "trt_output" in output_tensorrt


def test_model_runtime_api_clear_cache_routing() -> None:
    runtime_api = ModelRuntimeApi()
    fake_onnx_api = FakeOnnxApi()
    fake_tensorrt_api = FakeTensorrtApi()
    runtime_api._onnx_api = fake_onnx_api
    runtime_api._tensorrt_api = fake_tensorrt_api

    runtime_api.clear_session_cache(backend="onnx")
    assert fake_onnx_api.clear_cache_calls == 1
    assert fake_tensorrt_api.clear_cache_calls == 0

    runtime_api.clear_session_cache(backend="tensorrt")
    assert fake_onnx_api.clear_cache_calls == 1
    assert fake_tensorrt_api.clear_cache_calls == 1

    runtime_api.clear_session_cache(backend="all")
    assert fake_onnx_api.clear_cache_calls == 2
    assert fake_tensorrt_api.clear_cache_calls == 2


def test_model_runtime_api_rejects_unsupported_backend() -> None:
    runtime_api = ModelRuntimeApi()

    with pytest.raises(ValueError, match="Unsupported backend"):
        runtime_api.forward(backend="invalid", model_inputs=torch.randn(1, 3))

    with pytest.raises(ValueError, match="Unsupported backend cache selector"):
        runtime_api.clear_session_cache(backend="invalid")  # type: ignore[arg-type]
