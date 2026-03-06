from datetime import datetime
import importlib
import os

import pytest
import torch

from pyTorchAutoForge.api.onnx import ModelHandlerONNx

model_handler_module = importlib.import_module("pyTorchAutoForge.api.onnx.ModelHandlerONNx")


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


def _make_handler(tmp_path, onnx_export_path):
    return ModelHandlerONNx(
        model=SimpleModel().eval(),
        dummy_input_sample=torch.randn(1, 10),
        onnx_export_path=str(onnx_export_path),
        opset_version=13,
        run_export_validation=False,
        generate_report=False,
        run_onnx_simplify=False,
    )


def test_resolve_export_target_treats_export_path_onnx_as_full_file(tmp_path):
    target_path = tmp_path / "artifact.onnx"
    handler = _make_handler(tmp_path, target_path)

    resolved = handler._resolve_export_target()

    assert resolved == str(target_path)
    assert handler.onnx_filepath == str(target_path)


def test_resolve_export_target_with_bare_name_keeps_directory_mode(tmp_path):
    export_dir = tmp_path / "exports"
    handler = _make_handler(tmp_path, export_dir)

    resolved = handler._resolve_export_target(onnx_model_name="my_model")

    assert resolved == str(export_dir / "my_model.onnx")


def test_resolve_export_target_with_model_name_path_overrides_export_dir(tmp_path):
    export_dir = tmp_path / "exports"
    override_path = tmp_path / "nested" / "custom_model"
    handler = _make_handler(tmp_path, export_dir)

    resolved = handler._resolve_export_target(onnx_model_name=str(override_path))

    assert resolved == str(override_path.with_suffix(".onnx"))
    assert os.path.isdir(override_path.parent)


def test_auto_name_uses_minute_timestamp_and_overwrites_same_path(tmp_path, monkeypatch):
    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 3, 3, 12, 34)

    monkeypatch.setattr(model_handler_module, "datetime", FrozenDateTime)

    call_counter = {"count": 0}

    def fake_export(model, args, f, **kwargs):
        call_counter["count"] += 1
        with open(f, "wb") as file_handle:
            file_handle.write(f"call_{call_counter['count']}".encode("ascii"))
        return None

    monkeypatch.setattr(torch.onnx, "export", fake_export)

    export_dir = tmp_path / "exports"
    handler = _make_handler(tmp_path, export_dir)

    path_1 = handler.export_onnx(backend="legacy")
    path_2 = handler.export_onnx(backend="legacy")

    assert path_1 == path_2
    assert path_1.endswith("exports_20260303_1234.onnx")
    with open(path_1, "rb") as file_handle:
        assert file_handle.read() == b"call_2"


def test_export_onnx_dynamo_fallbacks_to_legacy(tmp_path, monkeypatch):
    def fake_export(model, args, f, **kwargs):
        if kwargs.get("dynamo", False):
            raise RuntimeError("dynamo failed")
        with open(f, "wb") as file_handle:
            file_handle.write(b"legacy_ok")
        return None

    monkeypatch.setattr(torch.onnx, "export", fake_export)

    handler = _make_handler(tmp_path, tmp_path / "fallback.onnx")

    with pytest.warns(RuntimeWarning):
        exported_path = handler.export_onnx(backend="dynamo", fallback_to_legacy=True)

    assert exported_path == str(tmp_path / "fallback.onnx")
    assert os.path.isfile(exported_path)


def test_wrapper_methods_emit_deprecation_warning_and_call_expected_backend(tmp_path, monkeypatch):
    recorded_calls = []

    def fake_export_onnx(**kwargs):
        recorded_calls.append(kwargs)
        return str(tmp_path / "mock.onnx")

    handler = _make_handler(tmp_path, tmp_path / "mock.onnx")
    monkeypatch.setattr(handler, "export_onnx", fake_export_onnx)

    with pytest.warns(FutureWarning):
        handler.torch_export()
    with pytest.warns(FutureWarning):
        handler.torch_dynamo_export()

    assert len(recorded_calls) == 2
    assert recorded_calls[0]["backend"] == "legacy"
    assert recorded_calls[0]["fallback_to_legacy"] is False
    assert recorded_calls[1]["backend"] == "dynamo"
    assert recorded_calls[1]["fallback_to_legacy"] is True


def test_export_onnx_passes_additional_export_kwargs_to_torch_export(tmp_path, monkeypatch):
    captured = {}

    def fake_export(model, args, f, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        with open(f, "wb") as file_handle:
            file_handle.write(b"ok")
        return None

    monkeypatch.setattr(torch.onnx, "export", fake_export)

    handler = _make_handler(tmp_path, tmp_path / "with_extra_args.onnx")
    primary_input = torch.randn(1, 10)

    exported_path = handler.export_onnx(
        model_inputs=primary_input,
        additional_export_kwargs={"training": torch.onnx.TrainingMode.EVAL},
        backend="legacy",
    )

    assert exported_path == str(tmp_path / "with_extra_args.onnx")
    assert os.path.isfile(exported_path)
    assert torch.equal(captured["args"], primary_input)
    assert captured["kwargs"]["training"] == torch.onnx.TrainingMode.EVAL


def test_export_onnx_supports_deprecated_input_tensor_alias(tmp_path, monkeypatch):
    captured = {}

    def fake_export(model, args, f, **kwargs):
        captured["args"] = args
        with open(f, "wb") as file_handle:
            file_handle.write(b"ok")
        return None

    monkeypatch.setattr(torch.onnx, "export", fake_export)

    handler = _make_handler(tmp_path, tmp_path / "alias_input.onnx")
    alias_input = torch.randn(1, 10)

    with pytest.warns(FutureWarning, match="input_tensor is deprecated"):
        exported_path = handler.export_onnx(
            input_tensor=alias_input,
            backend="legacy",
        )

    assert exported_path == str(tmp_path / "alias_input.onnx")
    assert os.path.isfile(exported_path)
    assert torch.equal(captured["args"], alias_input)


def test_export_onnx_rejects_both_model_inputs_and_input_tensor(tmp_path):
    handler = _make_handler(tmp_path, tmp_path / "invalid_inputs.onnx")
    first_input = torch.randn(1, 10)
    second_input = torch.randn(1, 10)

    with pytest.raises(ValueError, match="only one of: model_inputs or input_tensor"):
        handler.export_onnx(
            model_inputs=first_input,
            input_tensor=second_input,
            backend="legacy",
        )
