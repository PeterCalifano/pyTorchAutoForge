import logging
from pathlib import Path
import pathlib
import pytest
import torch
from torch import nn

from pyTorchAutoForge.model_building.poolingBlocks import (
    CustomAdaptiveAvgPool2d,
    CustomAdaptiveMaxPool2d,
)


def test_custom_adaptive_avg_matches_torch_adaptive_pool():
    x = torch.randn(2, 3, 4, 4)
    target_size = (2, 2)

    custom = CustomAdaptiveAvgPool2d(target_size)
    reference = nn.AdaptiveAvgPool2d(target_size)

    out_custom = custom(x)
    out_reference = reference(x)

    assert torch.allclose(out_custom, out_reference, atol=1e-6)


def test_custom_adaptive_avg_global_matches_torch_mean():
    x = torch.randn(1, 2, 5, 5)
    custom = CustomAdaptiveAvgPool2d((1, 1))
    reference = nn.AdaptiveAvgPool2d((1, 1))

    out_custom = custom(x)
    out_reference = reference(x)

    assert torch.allclose(out_custom, out_reference, atol=1e-6)
    assert out_custom.shape == (1, 2, 1, 1)


def test_custom_adaptive_max_matches_torch_adaptive_pool():
    x = torch.randn(2, 3, 4, 4)
    target_size = (2, 2)

    custom = CustomAdaptiveMaxPool2d(target_size)
    reference = nn.AdaptiveMaxPool2d(target_size)

    out_custom = custom(x)
    out_reference = reference(x)

    assert torch.equal(out_custom, out_reference)


def test_custom_adaptive_max_global_matches_torch_max():
    x = torch.randn(1, 2, 5, 5)
    custom = CustomAdaptiveMaxPool2d((1, 1))
    reference = nn.AdaptiveMaxPool2d((1, 1))

    out_custom = custom(x)
    out_reference = reference(x)

    assert torch.equal(out_custom, out_reference)
    assert out_custom.shape == (1, 2, 1, 1)


def test_custom_adaptive_max_raises_on_non_divisible_window():
    x = torch.randn(1, 1, 5, 4)
    layer = CustomAdaptiveMaxPool2d((3, 2))

    with pytest.raises(ValueError):
        layer(x)


def test_custom_pooling_gap_onnx_export(tmp_path):
    onnx = pytest.importorskip("onnx")
    logging.getLogger("onnxscript").setLevel(logging.WARNING)
    logging.getLogger("torch.onnx").setLevel(logging.WARNING)
    
    # Define test class
    class DummyPoolingModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Use global pooling to hit the ONNX-safe code paths.
            self.avg = CustomAdaptiveAvgPool2d((1, 1))
            self.max = CustomAdaptiveMaxPool2d((1, 1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            avg_out = self.avg(x)
            max_out = self.max(x)
            return avg_out + max_out#type: ignore

    model = DummyPoolingModel().eval()
    dummy_input = torch.randn(1, 2, 3, 3)
    export_path = Path(tmp_path) / "gap_pooling_model.onnx"

    torch.onnx.export(
        model,
        dummy_input,#type: ignore
        export_path,
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
    )

    onnx_model = onnx.load(export_path)
    #onnx.checker.check_model(onnx_model)
    assert export_path.is_file()


def test_custom_pooling_generic_onnx_export(tmp_path: pathlib.Path | str = "/tmp/"):
    onnx = pytest.importorskip("onnx")
    logging.getLogger("onnxscript").setLevel(logging.DEBUG)
    logging.getLogger("torch.onnx").setLevel(logging.DEBUG)

    # Define test class
    class DummyPoolingModelGeneric(nn.Module):
        def __init__(self):
            super().__init__()
            self.avg = CustomAdaptiveAvgPool2d((2, 2))
            self.max = CustomAdaptiveMaxPool2d((2, 2))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.avg(x) + self.max(x) #type: ignore

    model = DummyPoolingModelGeneric().eval()
    dummy_input = torch.randn(1, 2, 4, 4)
    export_path = Path(tmp_path) / "generic_pooling_model.onnx"


    # Example call
    #onnx_program = torch.onnx.export(model,
    #                                (tensor_input, dict_input, list_input),
    #                                dynamic_shapes=({0: "batch_size"},
    #                                                {"tensor_x": {0: "batch_size"}}, [{0: "batch_size"}]),
    #                                input_names=input_names, output_names=output_names, dynamo=True,)

    torch.onnx.export(model,
                    dummy_input,#type: ignore
                    export_path,
                    opset_version=12,
                    input_names=["input"],
                    output_names=["output"],
                    )

    onnx_model = onnx.load(export_path)
    #onnx.checker.check_model(onnx_model)
    assert export_path.is_file()


class testTorchAdaptivePoolModel(nn.Module):
    def __init__(self, pool_layer: nn.Module):
        super().__init__()
        self.pool = pool_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x) #type: ignore


@pytest.mark.parametrize(
    ("pool_ctor", "should_succeed"),
    [
        (lambda: nn.AdaptiveAvgPool2d((4, 4)), True),
        (lambda: nn.AdaptiveMaxPool2d((4, 4)), False),
    ],
)
def test_torch_builtin_adaptive_pool_dynamo_export(
    pool_ctor, should_succeed, tmp_path: pathlib.Path | str = "/tmp/"
):
    if not hasattr(torch.onnx, "dynamo_export"):
        pytest.skip("torch.onnx.dynamo_export not available in this Torch version")

    model = testTorchAdaptivePoolModel(pool_ctor()).eval()
    dummy_input = torch.randn(1, 3, 32, 32)
    export_path = Path(tmp_path) / "builtin_pool_model.onnx"

    if should_succeed:
        onnx = pytest.importorskip("onnx")
        onnx_program = torch.onnx.export(model,
                                          dummy_input, #type: ignore
                                            opset_version=18, 
                                            dynamo=True)

        assert onnx_program is not None
        onnx_program.save(str(export_path)) 

        onnx_model = onnx.load(export_path)
        onnx.checker.check_model(onnx_model)
        assert export_path.is_file()
    else:
        with pytest.raises(Exception):
            torch.onnx.export(model, dummy_input, opset_version=18, dynamo=True) #type: ignore


#######################################################################################
# FOR MANUAL TESTING PURPOSES ONLY
def manual_run():
    # test_custom_adaptive_avg_matches_torch_adaptive_pool()
    # test_custom_adaptive_avg_global_matches_torch_mean()
    # test_custom_adaptive_max_matches_torch_adaptive_pool()
    # test_custom_adaptive_max_global_matches_torch_max()
    # test_custom_adaptive_max_raises_on_non_divisible_window()
    #test_custom_pooling_generic_onnx_export()

    #def pool_ctor_avg(): return nn.AdaptiveAvgPool2d((4, 4))
    #test_torch_builtin_adaptive_pool_dynamo_export(pool_ctor_avg)

    #def pool_ctor_max(): return nn.AdaptiveMaxPool2d((4, 4))
    #test_torch_builtin_adaptive_pool_dynamo_export(pool_ctor_max)
    pass


if __name__ == "__main__":
    manual_run()
