from __future__ import annotations

import numpy
import pytest
import torch

from pyTorchAutoForge.api.tensorrt import TensorrtRuntimeApi


def _make_runtime_api_with_fake_session(monkeypatch: pytest.MonkeyPatch) -> tuple[TensorrtRuntimeApi, dict[str, int], dict[str, dict[str, numpy.ndarray]]]:
    runtime_api = TensorrtRuntimeApi(engine_bytes=b"serialized_engine")
    counters = {"create_session_calls": 0}
    captured = {"input_maps": []}

    monkeypatch.setattr(runtime_api, "_import_tensorrt", lambda: object())

    def fake_create_session_entry(trt, source_kind, source_payload):
        _ = trt
        _ = source_kind
        _ = source_payload
        counters["create_session_calls"] += 1
        return {
            "context": object(),
            "input_names": ["input"],
            "output_names": ["output"],
            "input_dtypes": {"input": "float32"},
            "output_dtypes": {"output": "float32"},
        }

    def fake_run_engine(trt, session_entry, input_map, device):
        _ = trt
        _ = session_entry
        _ = device
        captured["input_maps"].append(input_map)
        return {"output": numpy.ones((1, 3), dtype=numpy.float32)}

    monkeypatch.setattr(runtime_api, "_create_session_entry", fake_create_session_entry)
    monkeypatch.setattr(runtime_api, "_run_engine", fake_run_engine)
    return runtime_api, counters, captured


def test_tensorrt_forward_raises_import_error_when_tensorrt_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_api = TensorrtRuntimeApi(engine_bytes=b"serialized_engine")

    def raise_import_error():
        raise ImportError("mock missing tensorrt")

    monkeypatch.setattr(runtime_api, "_import_tensorrt", raise_import_error)

    with pytest.raises(ImportError, match="mock missing tensorrt"):
        runtime_api.forward(model_inputs=torch.randn(1, 3))


def test_tensorrt_forward_raises_when_source_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_api = TensorrtRuntimeApi()
    monkeypatch.setattr(runtime_api, "_import_tensorrt", lambda: object())

    with pytest.raises(ValueError, match="No TensorRT engine source available"):
        runtime_api.forward(model_inputs=torch.randn(1, 3))


def test_tensorrt_source_resolution_priority(tmp_path) -> None:
    stored_engine_path = tmp_path / "stored.engine"
    explicit_engine_path = tmp_path / "explicit.engine"
    stored_engine_path.write_bytes(b"stored")
    explicit_engine_path.write_bytes(b"explicit")

    runtime_api = TensorrtRuntimeApi(
        engine_bytes=b"stored_bytes",
        engine_filepath=str(stored_engine_path),
    )

    source_kind_1, source_payload_1 = runtime_api._resolve_engine_source(
        engine_bytes=b"explicit_bytes",
        engine_filepath=str(explicit_engine_path),
    )
    assert source_kind_1 == "bytes"
    assert source_payload_1 == b"explicit_bytes"

    source_kind_2, source_payload_2 = runtime_api._resolve_engine_source(
        engine_filepath=str(explicit_engine_path),
    )
    assert source_kind_2 == "bytes"
    assert source_payload_2 == b"stored_bytes"

    runtime_api.set_engine_source(engine_bytes=None, engine_filepath=str(stored_engine_path))
    source_kind_3, source_payload_3 = runtime_api._resolve_engine_source(
        engine_filepath=str(explicit_engine_path),
    )
    assert source_kind_3 == "file"
    assert source_payload_3 == str(explicit_engine_path.resolve())


def test_tensorrt_cache_reuse_force_refresh_and_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_api, counters, _ = _make_runtime_api_with_fake_session(monkeypatch)

    runtime_api.forward(model_inputs=torch.randn(1, 3))
    runtime_api.forward(model_inputs=torch.randn(1, 3))
    assert counters["create_session_calls"] == 1

    runtime_api.forward(model_inputs=torch.randn(1, 3), force_new_session=True)
    assert counters["create_session_calls"] == 2

    runtime_api.clear_session_cache()
    runtime_api.forward(model_inputs=torch.randn(1, 3))
    assert counters["create_session_calls"] == 3


def test_tensorrt_input_normalization_and_output_map(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_api = TensorrtRuntimeApi(engine_bytes=b"serialized_engine")
    monkeypatch.setattr(runtime_api, "_import_tensorrt", lambda: object())

    def fake_create_session_entry(trt, source_kind, source_payload):
        _ = trt
        _ = source_kind
        _ = source_payload
        return {
            "context": object(),
            "input_names": ["a", "b"],
            "output_names": ["y"],
            "input_dtypes": {"a": "float32", "b": "float32"},
            "output_dtypes": {"y": "float32"},
        }

    captured = {"input_map": None}

    def fake_run_engine(trt, session_entry, input_map, device):
        _ = trt
        _ = session_entry
        _ = device
        captured["input_map"] = input_map
        return {"y": numpy.zeros((1,), dtype=numpy.float32)}

    monkeypatch.setattr(runtime_api, "_create_session_entry", fake_create_session_entry)
    monkeypatch.setattr(runtime_api, "_run_engine", fake_run_engine)

    output_map = runtime_api.forward(
        model_inputs={"a": torch.randn(1, 2), "b": torch.randn(1, 2)}
    )
    assert list(output_map.keys()) == ["y"]
    assert set(captured["input_map"].keys()) == {"a", "b"}

    with pytest.raises(ValueError, match="Expected 2 inputs"):
        runtime_api.forward(model_inputs=(torch.randn(1, 2),))

    with pytest.raises(ValueError, match="model expects 2 inputs"):
        runtime_api.forward(model_inputs=torch.randn(1, 2))


def test_tensorrt_run_engine_raises_when_cuda_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_api = TensorrtRuntimeApi(engine_bytes=b"serialized_engine")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA is not available"):
        runtime_api._run_engine(
            trt=object(),
            session_entry={
                "context": object(),
                "input_names": ["input"],
                "output_names": ["output"],
                "input_dtypes": {"input": "float32"},
                "output_dtypes": {"output": "float32"},
            },
            input_map={"input": numpy.zeros((1, 3), dtype=numpy.float32)},
            device="cuda:0",
        )
