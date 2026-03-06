import os
from types import SimpleNamespace

import onnx
import pytest
import torch

from pyTorchAutoForge.api.onnx import OnnxRuntimeApi


class FakeInferenceSession:
    init_calls = 0
    last_source = None
    last_kwargs = {}
    input_names = ["input"]
    output_names = ["output"]

    def __init__(self, source, **kwargs):
        FakeInferenceSession.init_calls += 1
        FakeInferenceSession.last_source = source
        FakeInferenceSession.last_kwargs = kwargs

    def get_inputs(self):
        return [SimpleNamespace(name=name) for name in FakeInferenceSession.input_names]

    def get_outputs(self):
        return [SimpleNamespace(name=name) for name in FakeInferenceSession.output_names]

    def run(self, output_names, input_feed):
        first_input_name = FakeInferenceSession.input_names[0]
        return [input_feed[first_input_name]]


def _reset_fake_session():
    FakeInferenceSession.init_calls = 0
    FakeInferenceSession.last_source = None
    FakeInferenceSession.last_kwargs = {}
    FakeInferenceSession.input_names = ["input"]
    FakeInferenceSession.output_names = ["output"]


def _patch_ort(monkeypatch, available_providers):
    import onnxruntime as ort

    monkeypatch.setattr(ort, "get_available_providers", lambda: list(available_providers))
    monkeypatch.setattr(ort, "InferenceSession", FakeInferenceSession)


def test_runtime_provider_ordered_fallback_warns_and_uses_available(monkeypatch):
    _reset_fake_session()
    _patch_ort(monkeypatch, available_providers=["CPUExecutionProvider"])

    runtime_handler = OnnxRuntimeApi(onnx_model=onnx.ModelProto())

    with pytest.warns(RuntimeWarning, match="will be skipped"):
        output_dict = runtime_handler.forward(
            model_inputs=torch.randn(1, 3),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    assert "output" in output_dict
    assert FakeInferenceSession.last_kwargs["providers"] == ["CPUExecutionProvider"]


def test_runtime_provider_none_usable_raises(monkeypatch):
    _reset_fake_session()
    _patch_ort(monkeypatch, available_providers=["CPUExecutionProvider"])

    runtime_handler = OnnxRuntimeApi(onnx_model=onnx.ModelProto())

    with pytest.raises(ValueError, match="No usable execution providers"):
        runtime_handler.forward(
            model_inputs=torch.randn(1, 3),
            providers=["CUDAExecutionProvider"],
        )


def test_runtime_source_resolution_prefers_proto_over_filepath(monkeypatch, tmp_path):
    _reset_fake_session()
    _patch_ort(monkeypatch, available_providers=["CPUExecutionProvider"])

    filepath = tmp_path / "source.onnx"
    filepath.write_bytes(b"placeholder")

    runtime_handler = OnnxRuntimeApi(
        onnx_model=onnx.ModelProto(),
        onnx_filepath=str(filepath),
    )
    runtime_handler.forward(model_inputs=torch.randn(1, 3))

    assert isinstance(FakeInferenceSession.last_source, (bytes, bytearray))


def test_runtime_source_resolution_uses_filepath_when_model_missing(monkeypatch, tmp_path):
    _reset_fake_session()
    _patch_ort(monkeypatch, available_providers=["CPUExecutionProvider"])

    filepath = tmp_path / "source_file_only.onnx"
    filepath.write_bytes(b"placeholder")

    runtime_handler = OnnxRuntimeApi(onnx_filepath=str(filepath))
    runtime_handler.forward(model_inputs=torch.randn(1, 3))

    assert FakeInferenceSession.last_source == os.path.abspath(str(filepath))


def test_runtime_session_cache_reuse_and_force_refresh(monkeypatch):
    _reset_fake_session()
    _patch_ort(monkeypatch, available_providers=["CPUExecutionProvider"])

    runtime_handler = OnnxRuntimeApi(onnx_model=onnx.ModelProto())

    runtime_handler.forward(model_inputs=torch.randn(1, 3))
    runtime_handler.forward(model_inputs=torch.randn(1, 3))
    assert FakeInferenceSession.init_calls == 1

    runtime_handler.forward(model_inputs=torch.randn(1, 3), force_new_session=True)
    assert FakeInferenceSession.init_calls == 2


def test_runtime_clear_session_cache_recreates_session(monkeypatch):
    _reset_fake_session()
    _patch_ort(monkeypatch, available_providers=["CPUExecutionProvider"])

    runtime_handler = OnnxRuntimeApi(onnx_model=onnx.ModelProto())
    runtime_handler.forward(model_inputs=torch.randn(1, 3))
    runtime_handler.forward(model_inputs=torch.randn(1, 3))
    assert FakeInferenceSession.init_calls == 1

    runtime_handler.clear_session_cache()
    runtime_handler.forward(model_inputs=torch.randn(1, 3))
    assert FakeInferenceSession.init_calls == 2


def test_runtime_input_dict_and_input_count_validation(monkeypatch):
    _reset_fake_session()
    _patch_ort(monkeypatch, available_providers=["CPUExecutionProvider"])

    FakeInferenceSession.input_names = ["a", "b"]
    runtime_handler = OnnxRuntimeApi(onnx_model=onnx.ModelProto())

    output_dict = runtime_handler.forward(model_inputs={"a": torch.randn(1, 2), "b": torch.randn(1, 2)})
    assert "output" in output_dict

    with pytest.raises(ValueError, match="missing"):
        runtime_handler.forward(model_inputs={"a": torch.randn(1, 2)})

    with pytest.raises(ValueError, match="extra"):
        runtime_handler.forward(
            model_inputs={"a": torch.randn(1, 2), "b": torch.randn(1, 2), "c": torch.randn(1, 2)}
        )

    with pytest.raises(ValueError, match="Expected 2 inputs"):
        runtime_handler.forward(model_inputs=(torch.randn(1, 2),))

    with pytest.raises(ValueError, match="model expects 2 inputs"):
        runtime_handler.forward(model_inputs=torch.randn(1, 2))


def test_runtime_returns_output_name_map(monkeypatch):
    _reset_fake_session()
    _patch_ort(monkeypatch, available_providers=["CPUExecutionProvider"])

    FakeInferenceSession.output_names = ["y0"]
    runtime_handler = OnnxRuntimeApi(onnx_model=onnx.ModelProto())

    output_dict = runtime_handler.forward(model_inputs=torch.randn(1, 3))
    assert list(output_dict.keys()) == ["y0"]


def test_runtime_set_source_can_clear_model_and_use_filepath(monkeypatch, tmp_path):
    _reset_fake_session()
    _patch_ort(monkeypatch, available_providers=["CPUExecutionProvider"])

    filepath = tmp_path / "clear_source.onnx"
    filepath.write_bytes(b"placeholder")

    runtime_handler = OnnxRuntimeApi(
        onnx_model=onnx.ModelProto(),
        onnx_filepath=str(filepath),
    )
    runtime_handler.set_onnx_source(onnx_model=None, onnx_filepath=str(filepath))
    runtime_handler.forward(model_inputs=torch.randn(1, 3))

    assert FakeInferenceSession.last_source == os.path.abspath(str(filepath))


def test_runtime_provider_options_aligned_with_selected_provider(monkeypatch):
    _reset_fake_session()
    _patch_ort(monkeypatch, available_providers=["CPUExecutionProvider"])

    runtime_handler = OnnxRuntimeApi(onnx_model=onnx.ModelProto())
    runtime_handler.forward(
        model_inputs=torch.randn(1, 3),
        providers=["CPUExecutionProvider"],
        provider_options={"CPUExecutionProvider": {"intra_op_num_threads": "1"}},
    )

    assert FakeInferenceSession.last_kwargs["provider_options"] == [{"intra_op_num_threads": "1"}]


def test_runtime_input_tensor_alias_warning(monkeypatch):
    _reset_fake_session()
    _patch_ort(monkeypatch, available_providers=["CPUExecutionProvider"])

    runtime_handler = OnnxRuntimeApi(onnx_model=onnx.ModelProto())

    with pytest.warns(FutureWarning, match="input_tensor is deprecated"):
        output_dict = runtime_handler.forward(input_tensor=torch.randn(1, 3))

    assert "output" in output_dict


def test_runtime_forward_requires_source(monkeypatch):
    _reset_fake_session()
    _patch_ort(monkeypatch, available_providers=["CPUExecutionProvider"])

    runtime_handler = OnnxRuntimeApi()

    with pytest.raises(ValueError, match="No ONNX source available"):
        runtime_handler.forward(model_inputs=torch.randn(1, 3))
