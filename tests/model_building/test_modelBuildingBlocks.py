import pytest
import torch

from pyTorchAutoForge.model_building.modelBuildingBlocks import AutoForgeModule, DenormalizeImg, TemplateConvNetConfig2d, TemplateNetBaseConfig, NormalizeImg, TemplateConvNet2d
from torch import nn
from pyTorchAutoForge.model_building.modelBuildingBlocks import DropoutEnsemblingNetworkWrapper


# %% AutoForgeModule tests
def test_autoforge_module_default_name():
    m = AutoForgeModule()
    # When no name is provided, moduleName defaults to the class name
    assert m.moduleName == "AutoForgeModule"

def test_autoforge_module_custom_name():
    m = AutoForgeModule(moduleName="my_module", enable_tracing=True)
    # Provided name should override the default
    assert m.moduleName == "my_module"

# %% Image normalization classes tests
def test_normalize_img_forward():
    norm = NormalizeImg(normaliz_factor=10.0)
    x = torch.tensor([10.0, 20.0, 30.0])
    out = norm(x)
    # Should divide every element by 10
    assert torch.allclose(out, torch.tensor([1.0, 2.0, 3.0]))

def test_renormalize_img_forward():
    renorm = DenormalizeImg(normaliz_factor=5.0)
    x = torch.tensor([1.0, 2.0, 3.0])
    out = renorm(x)
    # Should multiply every element by 5
    assert torch.allclose(out, torch.tensor([5.0, 10.0, 15.0]))

# %% TemplateConvNetConfig2d tests
def test_template_convnet_config2d_valid_lists():
    cfg = TemplateConvNetConfig2d(
        kernel_sizes=[3, 5],
        pool_type="MaxPool2d",
        pool_kernel_sizes=[2, 2],
        out_channels_sizes=[8, 16],
        num_input_channels=3,
    )
    # Attributes should be set as provided
    assert cfg.kernel_sizes == [3, 5]
    assert cfg.pool_kernel_sizes == [2, 2]
    assert cfg.out_channels_sizes == [8, 16]
    assert cfg.num_input_channels == 3

@pytest.mark.parametrize("bad_pool", ["MaxPool3d", "AvgPool1d", "Adapt_MaxPool1d"])
def test_template_convnet_config2d_invalid_pool_type(bad_pool):
    with pytest.raises(TypeError) as exc:
        TemplateConvNetConfig2d(
            kernel_sizes=[3],
            pool_type=bad_pool,
            pool_kernel_sizes=[2],
            out_channels_sizes=[4],
            num_input_channels=1,
        )
    assert "pool_type must be of type" in str(exc.value)

def test_template_convnet_config2d_kernel_none():
    with pytest.raises(ValueError) as exc:
        TemplateConvNetConfig2d(
            kernel_sizes=None,
            pool_type="MaxPool2d",
            pool_kernel_sizes=[2],
            out_channels_sizes=[4],
            num_input_channels=1,
        )
    assert "'kernel_sizes' cannot be None" in str(exc.value)

def test_template_convnet_config2d_pool_none():
    with pytest.raises(ValueError) as exc:
        TemplateConvNetConfig2d(
            kernel_sizes=[3],
            pool_type="MaxPool2d",
            pool_kernel_sizes=None,
            out_channels_sizes=[4],
            num_input_channels=1,
        )
    assert "'pool_kernel_sizes' cannot be None" in str(exc.value)

def test_template_convnet_config2d_out_channels_none():
    with pytest.raises(ValueError) as exc:
        TemplateConvNetConfig2d(
            kernel_sizes=[3],
            pool_type="MaxPool2d",
            pool_kernel_sizes=[2],
            out_channels_sizes=None,
            num_input_channels=1,
        )
    assert "out_channels_sizes cannot be None" in str(exc.value)

def test_template_convnet_config2d_length_mismatch():
    # kernel_sizes and pool_kernel_sizes length mismatch
    with pytest.raises(ValueError) as exc1:
        TemplateConvNetConfig2d(
            kernel_sizes=[3, 5],
            pool_type="MaxPool2d",
            pool_kernel_sizes=[2],
            out_channels_sizes=[4, 8],
            num_input_channels=1,
        )
    assert "must have the same length" in str(exc1.value)

    # kernel_sizes and out_channels_sizes length mismatch
    with pytest.raises(ValueError) as exc2:
        TemplateConvNetConfig2d(
            kernel_sizes=[3],
            pool_type="MaxPool2d",
            pool_kernel_sizes=[2],
            out_channels_sizes=[4, 8],
            num_input_channels=1,
        )

    assert "must have the same length" in str(exc2.value)

# %% TemplateConvNet2d tests
def test_template_convnet2d_build_and_forward_default_no_skips():
    cfg = TemplateConvNetConfig2d(
        kernel_sizes=[3, 5],
        pool_kernel_sizes=[2, 2],
        out_channels_sizes=[4, 8],
        num_input_channels=1,
    )
    model = TemplateConvNet2d(cfg)
    # Check basic attributes
    assert model.num_of_conv_blocks == 2
    assert len(model.blocks) == 2
    # Forward pass with batch size 2, 1 channel, 16x16 input
    x = torch.randn(2, 1, 16, 16)
    out, skips = model(x)
    # Default save_intermediate_features is False -> skips empty
    assert isinstance(out, torch.Tensor)
    assert skips == []
    # Compute expected spatial size after conv+pool layers:
    # First block: (16 - 3 + 1) = 14 -> pool2 -> 7
    # Second block: (7 - 5 + 1) = 3 -> pool2 -> 1
    assert out.shape == (2, 8, 1, 1)


def test_template_convnet2d_forward_with_intermediate_features():
    cfg = TemplateConvNetConfig2d(
        kernel_sizes=[3, 3, 3],
        pool_kernel_sizes=[2, 2, 2],
        out_channels_sizes=[2, 4, 6],
        num_input_channels=1,
        save_intermediate_features=True,
    )
    model = TemplateConvNet2d(cfg)
    x = torch.randn(1, 1, 20, 20)
    out, skips = model(x)
    # Should collect one intermediate feature per block
    assert len(skips) == 3
    # Final output channel matches last out_channels_sizes
    assert out.shape[1] == 6


@pytest.mark.parametrize("kernels,pools,msg", [
    (None, [2], "must not be none"),
    ([3], None, "must not be none"),
])
def test_template_convnet2d_none_kernel_or_pool_raises(kernels, pools, msg):
    cfg = TemplateConvNetConfig2d(
        kernel_sizes=kernels,
        pool_kernel_sizes=pools,
        out_channels_sizes=[4],
        num_input_channels=1,
    )
    with pytest.raises(ValueError) as exc:
        TemplateConvNet2d(cfg)
    assert msg in str(exc.value)


def test_template_convnet2d_mismatched_kernel_pool_length():
    cfg = TemplateConvNetConfig2d(
        kernel_sizes=[3, 5],
        pool_kernel_sizes=[2],
        out_channels_sizes=[4, 8],
        num_input_channels=1,
    )
    with pytest.raises(ValueError) as exc:
        TemplateConvNet2d(cfg)
    assert "must have the same length" in str(exc.value)


def test_template_convnet2d_scalar_pool_raises():
    cfg = TemplateConvNetConfig2d(
        kernel_sizes=[3],
        pool_kernel_sizes=2,
        out_channels_sizes=[4],
        num_input_channels=1,
    )
    with pytest.raises(ValueError) as exc:
        TemplateConvNet2d(cfg)
    assert "cannot be scalar" in str(exc.value)


def test_template_convnet2d_output_shape_simple():
    # Single-block network: check computed output size matches formula
    cfg = TemplateConvNetConfig2d(
        kernel_sizes=[3],
        pool_kernel_sizes=[2],
        out_channels_sizes=[4],
        num_input_channels=3,
    )
    model = TemplateConvNet2d(cfg)
    # Input spatial size 10x10
    x = torch.randn(2, 3, 10, 10)
    out, _ = model(x)
    h_out = (10 - 3 + 1) // 2  # (kernel removes 2) then pool halves
    w_out = h_out
    assert out.shape == (2, 4, h_out, w_out)


# %% DropoutEnsemblingNetworkWrapper tests
class DummyModel(nn.Module):
    def __init__(self, out_features=4):
        super().__init__()
        self.linear = nn.Linear(3, out_features)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        return self.linear(x)

def test_dropout_ensembling_wrapper_init_accepts_nn_module():
    base = DummyModel()
    wrapper = DropoutEnsemblingNetworkWrapper(base)
    assert isinstance(wrapper.base_model, nn.Module)
    assert wrapper.ensemble_size == 20
    assert wrapper.enable_ensembling_ is True

def test_dropout_ensembling_wrapper_init_rejects_non_module():
    with pytest.raises(TypeError):
        DropoutEnsemblingNetworkWrapper(model="not_a_module")

def test_dropout_ensembling_forward_training_mode():
    base = DummyModel()
    wrapper = DropoutEnsemblingNetworkWrapper(base)
    wrapper.train()
    x = torch.randn(2, 3)
    out = wrapper(x)
    # Should just call base model once, no ensembling
    assert out.shape == (2, base.linear.out_features)
    assert torch.allclose(wrapper.last_mean, out)
    assert torch.allclose(wrapper.last_median, out)
    assert torch.all(wrapper.last_variance == 0)

def test_dropout_ensembling_forward_eval_mode_batch_size_1():
    base = DummyModel()
    wrapper = DropoutEnsemblingNetworkWrapper(base, ensemble_size=5)
    wrapper.eval()
    x = torch.randn(1, 3)
    out = wrapper(x)
    # Output should be the mean of 5 forward passes, with size (1, out_features)
    assert out.shape == (1, base.linear.out_features)
    assert wrapper.last_mean.shape == (1, base.linear.out_features)
    assert wrapper.last_median.shape == (1, base.linear.out_features)
    assert wrapper.last_variance.shape == (1, base.linear.out_features)

def test_dropout_ensembling_forward_eval_mode_batch_size_gt1():
    base = DummyModel()
    wrapper = DropoutEnsemblingNetworkWrapper(base, ensemble_size=4)
    wrapper.eval()
    x = torch.randn(3, 3)
    out = wrapper(x)
    # Should stack 4 outputs of shape (3, out_features), mean over dim=0
    assert out.shape == (3, base.linear.out_features)
    assert wrapper.last_mean.shape == (3, base.linear.out_features)
    assert wrapper.last_median.shape == (3, base.linear.out_features)
    assert wrapper.last_variance.shape == (3, base.linear.out_features)

if __name__ == "__main__":
    pytest.main([__file__])