from __future__ import annotations

from pathlib import Path
import subprocess
from types import SimpleNamespace
import importlib.util
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TRT_EXPORTER_MODULE_PATH = REPO_ROOT / "pyTorchAutoForge/api/tensorrt/TRTengineExporter.py"
SPEC = importlib.util.spec_from_file_location(
    "ptaf_trt_exporter_module", TRT_EXPORTER_MODULE_PATH
)
assert SPEC is not None
assert SPEC.loader is not None
trt_exporter_module = importlib.util.module_from_spec(SPEC)
sys.modules["ptaf_trt_exporter_module"] = trt_exporter_module
SPEC.loader.exec_module(trt_exporter_module)

TRTengineExporter = trt_exporter_module.TRTengineExporter
TRTengineExporterConfig = trt_exporter_module.TRTengineExporterConfig
TRTengineExporterMode = trt_exporter_module.TRTengineExporterMode
TRTprecision = trt_exporter_module.TRTprecision
TRTDynamicShapeProfile = trt_exporter_module.TRTDynamicShapeProfile


def _make_dummy_onnx_file(tmp_path: Path, filename: str = "model.onnx") -> Path:
    onnx_path_ = tmp_path / filename
    onnx_path_.write_bytes(b"dummy_onnx")
    return onnx_path_


def _make_fake_trt_module(parse_success: bool = True, serialized_engine: bytes | None = b"serialized_engine") -> SimpleNamespace:
    class Logger:
        WARNING = 1
        VERBOSE = 2

        def __init__(self, level: int) -> None:
            self.level = level

    class NetworkDefinitionCreationFlag:
        EXPLICIT_BATCH = 0

    class BuilderFlag:
        FP16 = 1
        INT8 = 2

    class MemoryPoolType:
        WORKSPACE = 1

    class FakeOptimizationProfile:
        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = []

        def set_shape(self, input_name: str, min_shape: tuple[int, ...], opt_shape: tuple[int, ...], max_shape: tuple[int, ...]) -> bool:
            self.calls.append((input_name, min_shape, opt_shape, max_shape))
            return True

    class FakeBuilderConfig:
        def __init__(self) -> None:
            self.flags: list[int] = []
            self.workspace_limit: int | None = None
            self.optimization_profiles: list[FakeOptimizationProfile] = []

        def set_flag(self, flag: int) -> None:
            self.flags.append(flag)

        def set_memory_pool_limit(self, pool_type: int, workspace_limit: int) -> None:
            _ = pool_type
            self.workspace_limit = workspace_limit

        def add_optimization_profile(self, profile: FakeOptimizationProfile) -> None:
            self.optimization_profiles.append(profile)

    class FakeParser:
        def __init__(self, network: object, logger: Logger) -> None:
            _ = network
            _ = logger
            self.num_errors = 0 if parse_success else 1

        def parse(self, onnx_bytes: bytes) -> bool:
            _ = onnx_bytes
            return parse_success

        def get_error(self, index: int) -> str:
            _ = index
            return "mock parser error"

    class FakeBuilder:
        def __init__(self, logger: Logger) -> None:
            self.logger = logger
            self.config = FakeBuilderConfig()

        def create_network(self, flags: int) -> object:
            _ = flags
            return object()

        def create_builder_config(self) -> FakeBuilderConfig:
            return self.config

        def create_optimization_profile(self) -> FakeOptimizationProfile:
            return FakeOptimizationProfile()

        def build_serialized_network(self, network: object, config: FakeBuilderConfig) -> bytes | None:
            _ = network
            _ = config
            return serialized_engine

    return SimpleNamespace(
        Logger=Logger,
        Builder=FakeBuilder,
        OnnxParser=FakeParser,
        NetworkDefinitionCreationFlag=NetworkDefinitionCreationFlag,
        BuilderFlag=BuilderFlag,
        MemoryPoolType=MemoryPoolType,
    )


def test_precision_enum_does_not_include_int16() -> None:
    assert not hasattr(TRTprecision, "INT16")
    assert {mode.name for mode in TRTprecision} == {"FP32", "FP16", "INT8"}


def test_constructor_is_side_effect_free_and_supports_kwargs_override() -> None:
    config_ = TRTengineExporterConfig(
        exporter_mode=TRTengineExporterMode.TRTEXEC,
        precision=TRTprecision.FP32,
        verbose=False,
    )

    exporter_ = TRTengineExporter(config=config_, precision=TRTprecision.INT8, verbose=True)

    assert exporter_.precision == TRTprecision.INT8
    assert exporter_.verbose is True
    assert exporter_.exporter_mode == TRTengineExporterMode.TRTEXEC


def test_validate_onnx_input_path_rejects_missing_file(tmp_path: Path) -> None:
    exporter_ = TRTengineExporter()
    missing_path_ = tmp_path / "missing.onnx"

    with pytest.raises(FileNotFoundError):
        exporter_._validate_onnx_input_path(str(missing_path_))


def test_validate_onnx_input_path_rejects_non_onnx_extension(tmp_path: Path) -> None:
    exporter_ = TRTengineExporter()
    invalid_path_ = tmp_path / "model.txt"
    invalid_path_.write_text("not_onnx", encoding="utf-8")

    with pytest.raises(ValueError, match="must end with '.onnx'"):
        exporter_._validate_onnx_input_path(str(invalid_path_))


def test_resolve_output_engine_path_covers_explicit_default_and_fallback(tmp_path: Path) -> None:
    explicit_path_ = tmp_path / "explicit.engine"
    exporter_ = TRTengineExporter()
    assert exporter_._resolve_output_engine_path(str(explicit_path_)) == str(explicit_path_)

    onnx_path_ = tmp_path / "model.onnx"
    assert exporter_._resolve_output_engine_path(None, str(onnx_path_)) == str(tmp_path / "model.engine")

    fallback_path_ = exporter_._resolve_output_engine_path(None, None)
    assert fallback_path_.endswith("trt_engine.engine")


def test_build_with_trtexec_raises_when_binary_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exporter_ = TRTengineExporter()
    onnx_path_ = _make_dummy_onnx_file(tmp_path)
    output_path_ = tmp_path / "model.engine"

    monkeypatch.setattr(trt_exporter_module.shutil, "which", lambda _: None)

    with pytest.raises(FileNotFoundError, match="trtexec executable not found"):
        exporter_._build_with_trtexec(str(onnx_path_), str(output_path_))


def test_build_with_trtexec_uses_safe_arg_list_and_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    profile_ = TRTDynamicShapeProfile(
        input_name="input",
        min_shape=(1, 3, 224, 224),
        opt_shape=(2, 3, 224, 224),
        max_shape=(4, 3, 224, 224),
    )
    exporter_ = TRTengineExporter(
        precision=TRTprecision.FP16,
        workspace_size_bytes=16 * 1024 * 1024,
        dynamic_shape_profiles=(profile_,),
        trtexec_extra_args=("--skipInference",),
    )

    onnx_path_ = _make_dummy_onnx_file(tmp_path, "with space.onnx")
    output_path_ = tmp_path / "with space.engine"
    call_data_: dict[str, object] = {}

    def fake_run(command: list[str], check: bool, stdout: int, stderr: int, text: bool):
        call_data_["command"] = command
        call_data_["check"] = check
        call_data_["text"] = text
        for arg_ in command:
            if arg_.startswith("--saveEngine="):
                engine_path_ = Path(arg_.split("=", maxsplit=1)[1])
                engine_path_.write_bytes(b"engine")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(trt_exporter_module.shutil, "which", lambda _: "/usr/bin/trtexec")
    monkeypatch.setattr(trt_exporter_module.subprocess, "run", fake_run)

    built_engine_path_ = exporter_._build_with_trtexec(str(onnx_path_), str(output_path_))

    assert built_engine_path_ == str(output_path_)
    command_ = call_data_["command"]
    assert isinstance(command_, list)
    assert "--fp16" in command_
    assert "--workspace=16" in command_
    assert "--skipInference" in command_
    assert "--minShapes=input:1x3x224x224" in command_
    assert "--optShapes=input:2x3x224x224" in command_
    assert "--maxShapes=input:4x3x224x224" in command_


def test_build_with_trtexec_surfaces_subprocess_error_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exporter_ = TRTengineExporter()
    onnx_path_ = _make_dummy_onnx_file(tmp_path)
    output_path_ = tmp_path / "model.engine"

    monkeypatch.setattr(trt_exporter_module.shutil, "which", lambda _: "/usr/bin/trtexec")

    def fake_run(command: list[str], check: bool, stdout: int, stderr: int, text: bool):
        _ = check
        _ = stdout
        _ = stderr
        _ = text
        return subprocess.CompletedProcess(command, 1, stdout="std_out", stderr="std_err")

    monkeypatch.setattr(trt_exporter_module.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="std_out"):
        exporter_._build_with_trtexec(str(onnx_path_), str(output_path_))


def test_python_builder_raises_when_tensorrt_missing() -> None:
    exporter_ = TRTengineExporter(exporter_mode=TRTengineExporterMode.PYTHON)

    with pytest.raises(ImportError, match="TensorRT Python API is not available"):
        exporter_._build_with_python_api_from_onnx_bytes(b"onnx", "/tmp/model.engine")


def test_python_builder_parser_failure_reports_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exporter_ = TRTengineExporter(exporter_mode=TRTengineExporterMode.PYTHON)
    fake_trt_ = _make_fake_trt_module(parse_success=False)
    monkeypatch.setattr(exporter_, "_import_tensorrt", lambda: fake_trt_)

    output_path_ = tmp_path / "model.engine"

    with pytest.raises(RuntimeError, match="mock parser error"):
        exporter_._build_with_python_api_from_onnx_bytes(b"onnx_bytes", str(output_path_))


def test_python_builder_success_writes_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    profile_ = TRTDynamicShapeProfile(
        input_name="input",
        min_shape=(1, 3, 8, 8),
        opt_shape=(2, 3, 8, 8),
        max_shape=(4, 3, 8, 8),
    )

    exporter_ = TRTengineExporter(
        exporter_mode=TRTengineExporterMode.PYTHON,
        precision=TRTprecision.FP16,
        workspace_size_bytes=1024,
        dynamic_shape_profiles=(profile_,),
    )
    fake_trt_ = _make_fake_trt_module(parse_success=True, serialized_engine=b"engine_data")
    monkeypatch.setattr(exporter_, "_import_tensorrt", lambda: fake_trt_)

    output_path_ = tmp_path / "python_mode.engine"
    built_path_ = exporter_._build_with_python_api_from_onnx_bytes(b"onnx_bytes", str(output_path_))

    assert built_path_ == str(output_path_)
    assert output_path_.read_bytes() == b"engine_data"


def test_trtexec_dispatch_does_not_trigger_tensorrt_import(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exporter_ = TRTengineExporter(exporter_mode=TRTengineExporterMode.TRTEXEC)
    onnx_path_ = _make_dummy_onnx_file(tmp_path)
    output_path_ = tmp_path / "dispatch.engine"

    monkeypatch.setattr(exporter_, "_import_tensorrt", lambda: (_ for _ in ()).throw(AssertionError("unexpected import")))
    monkeypatch.setattr(exporter_, "_build_with_trtexec", lambda onnx_model_path, output_engine_path: output_engine_path)

    built_path_ = exporter_._dispatch_build_from_onnx_path(str(onnx_path_), str(output_path_))
    assert built_path_ == str(output_path_)


def test_dispatch_from_onnx_model_trtexec_uses_tempfile_and_cleans_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exporter_ = TRTengineExporter(exporter_mode=TRTengineExporterMode.TRTEXEC)
    observed_temp_path_: dict[str, str] = {}

    class FakeOnnxModel:
        pass

    def fake_save_onnx_model_to_path(onnx_model: object, onnx_path: str) -> None:
        _ = onnx_model
        Path(onnx_path).write_bytes(b"serialized")

    def fake_build_with_trtexec(onnx_model_path: str, output_engine_path: str) -> str:
        observed_temp_path_["path"] = onnx_model_path
        Path(output_engine_path).write_bytes(b"engine")
        return output_engine_path

    monkeypatch.setattr(exporter_, "_save_onnx_model_to_path", fake_save_onnx_model_to_path)
    monkeypatch.setattr(exporter_, "_build_with_trtexec", fake_build_with_trtexec)

    output_path_ = tmp_path / "model.engine"
    built_path_ = exporter_._dispatch_build_from_onnx_model(FakeOnnxModel(), str(output_path_), None)

    assert built_path_ == str(output_path_)
    assert "path" in observed_temp_path_
    assert not Path(observed_temp_path_["path"]).exists()


def test_export_torch_to_onnx_delegates_to_model_handler(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exporter_ = TRTengineExporter()
    exported_onnx_path_ = tmp_path / "exported.onnx"
    capture_: dict[str, object] = {}

    class FakeModelHandler:
        def __init__(self, **kwargs):
            capture_["init_kwargs"] = kwargs

        def export_onnx(self, **kwargs):
            capture_["export_kwargs"] = kwargs
            return str(exported_onnx_path_)

    monkeypatch.setattr(exporter_, "_import_model_handler_onnx", lambda: FakeModelHandler)

    result_path_ = exporter_.export_torch_to_onnx(
        torch_model=object(),
        input_sample=object(),
        onnx_export_path=str(tmp_path),
        backend="legacy",
        fallback_to_legacy=False,
        run_export_validation=True,
        opset_version=18,
        additional_export_kwargs={"custom_flag": True},
        onnx_model_name="model_name",
    )

    assert result_path_ == str(exported_onnx_path_)
    assert capture_["init_kwargs"]["onnx_export_path"] == str(tmp_path)
    assert capture_["export_kwargs"]["onnx_model_name"] == "model_name"


def test_export_engine_end_to_end_dispatches_to_torch_route(monkeypatch: pytest.MonkeyPatch) -> None:
    exporter_ = TRTengineExporter()

    monkeypatch.setattr(
        exporter_,
        "export_torch_to_trt_engine",
        lambda **kwargs: "torch_route.engine",
    )

    result_ = exporter_.export_engine_end_to_end(
        torch_model=object(),
        input_sample=object(),
        onnx_export_path="/tmp/model.onnx",
    )

    assert result_ == "torch_route.engine"


def test_export_engine_end_to_end_dispatches_to_onnx_path_route(monkeypatch: pytest.MonkeyPatch) -> None:
    exporter_ = TRTengineExporter()
    monkeypatch.setattr(exporter_, "build_engine_from_onnx_path", lambda **kwargs: "path_route.engine")

    result_ = exporter_.export_engine_end_to_end(onnx_model_path="/tmp/model.onnx")
    assert result_ == "path_route.engine"


def test_export_engine_end_to_end_dispatches_to_onnx_model_route(monkeypatch: pytest.MonkeyPatch) -> None:
    exporter_ = TRTengineExporter()
    monkeypatch.setattr(exporter_, "build_engine_from_onnx_model", lambda **kwargs: "model_route.engine")

    result_ = exporter_.export_engine_end_to_end(onnx_model=object())
    assert result_ == "model_route.engine"


def test_export_engine_end_to_end_rejects_ambiguous_or_missing_inputs() -> None:
    exporter_ = TRTengineExporter()

    with pytest.raises(ValueError, match="No input source provided"):
        exporter_.export_engine_end_to_end()

    with pytest.raises(ValueError, match="Ambiguous input source"):
        exporter_.export_engine_end_to_end(
            torch_model=object(),
            input_sample=object(),
            onnx_export_path="/tmp/model.onnx",
            onnx_model_path="/tmp/model.onnx",
        )


def test_public_methods_use_shared_dispatchers(monkeypatch: pytest.MonkeyPatch) -> None:
    exporter_ = TRTengineExporter()

    monkeypatch.setattr(
        exporter_,
        "_dispatch_build_from_onnx_path",
        lambda onnx_model_path, output_engine_path=None: "shared_path.engine",
    )
    monkeypatch.setattr(
        exporter_,
        "_dispatch_build_from_onnx_model",
        lambda onnx_model, output_engine_path=None, temporary_onnx_path=None: "shared_model.engine",
    )

    assert exporter_.build_engine_from_onnx_path("/tmp/model.onnx") == "shared_path.engine"
    assert exporter_.build_engine_from_onnx_model(object()) == "shared_model.engine"


def test_configuration_rejects_invalid_dynamic_profiles() -> None:
    with pytest.raises(ValueError, match="Expected min <= opt <= max"):
        TRTengineExporter(
            dynamic_shape_profiles=(
                TRTDynamicShapeProfile(
                    input_name="input",
                    min_shape=(2, 3, 8, 8),
                    opt_shape=(1, 3, 8, 8),
                    max_shape=(4, 3, 8, 8),
                ),
            )
        )
