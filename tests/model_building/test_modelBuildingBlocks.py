import pytest
import torch
from pyTorchAutoForge.model_building.modelBuildingBlocks import AutoForgeModule, DenormalizeImg, TemplateConvNet2dConfig, TemplateNetBaseConfig, NormalizeImg, TemplateConvNet2d, TemplateFullyConnectedNetConfig, TemplateFullyConnectedNet,    TemplateConvNetFeatureFuser2dConfig, TemplateConvNetFeatureFuser2d,

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

# %% TemplateConvNet2dConfig tests
def test_template_convnet2d_config_pool_scalar_unroll():
    cfg = TemplateConvNet2dConfig(
        kernel_sizes=[3, 5],
        pool_type="MaxPool2d",
        pool_kernel_sizes=2,
        out_channels_sizes=[4, 8],
        num_input_channels=1,
    )
    assert cfg.pool_kernel_sizes == [2, 2]


@pytest.mark.parametrize("bad_pool", ["MaxPool3d", "AvgPool1d", "Adapt_AvgPool3"])
def test_template_convnet2d_config_invalid_pool_type(bad_pool):
    with pytest.raises(TypeError):
        TemplateConvNet2dConfig(
            kernel_sizes=[3],
            pool_type=bad_pool,
            pool_kernel_sizes=[2],
            out_channels_sizes=[4],
            num_input_channels=1,
        )

def test_template_convnet_config2d_valid_lists():
    cfg = TemplateConvNet2dConfig(
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
        TemplateConvNet2dConfig(
            kernel_sizes=[3],
            pool_type=bad_pool,
            pool_kernel_sizes=[2],
            out_channels_sizes=[4],
            num_input_channels=1,
        )
    assert "pool_type must be of type" in str(exc.value)

def test_template_convnet_config2d_kernel_none():
    with pytest.raises(ValueError) as exc:
        TemplateConvNet2dConfig(
            kernel_sizes=None,
            pool_type="MaxPool2d",
            pool_kernel_sizes=[2],
            out_channels_sizes=[4],
            num_input_channels=1,
        )
    assert "'kernel_sizes' cannot be None" in str(exc.value)

def test_template_convnet_config2d_pool_none():
    with pytest.raises(ValueError) as exc:
        TemplateConvNet2dConfig(
            kernel_sizes=[3],
            pool_type="MaxPool2d",
            pool_kernel_sizes=None,
            out_channels_sizes=[4],
            num_input_channels=1,
        )
    assert "'pool_kernel_sizes' cannot be None" in str(exc.value)

def test_template_convnet_config2d_out_channels_none():
    with pytest.raises(ValueError) as exc:
        TemplateConvNet2dConfig(
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
        TemplateConvNet2dConfig(
            kernel_sizes=[3, 5],
            pool_type="MaxPool2d",
            pool_kernel_sizes=[2],
            out_channels_sizes=[4, 8],
            num_input_channels=1,
        )
    assert "must have the same length" in str(exc1.value)

    # kernel_sizes and out_channels_sizes length mismatch
    with pytest.raises(ValueError) as exc2:
        TemplateConvNet2dConfig(
            kernel_sizes=[3],
            pool_type="MaxPool2d",
            pool_kernel_sizes=[2],
            out_channels_sizes=[4, 8],
            num_input_channels=1,
        )

    assert "must have the same length" in str(exc2.value)

def test_template_convnet2d_forward_no_intermediate():
    cfg = TemplateConvNet2dConfig(
        kernel_sizes=[3, 3],
        pool_type="MaxPool2d",
        pool_kernel_sizes=[2, 2],
        out_channels_sizes=[4, 6],
        num_input_channels=1,
    )
    model = TemplateConvNet2d(cfg)
    x = torch.randn(2, 1, 10, 10)
    out, skips = model(x)
    assert isinstance(out, torch.Tensor)
    assert skips == []


def test_template_convnet2d_forward_with_intermediate_and_regressor():
    cfg = TemplateConvNet2dConfig(
        kernel_sizes=[3],
        pool_type="AvgPool2d",
        pool_kernel_sizes=[2],
        out_channels_sizes=[4],
        num_input_channels=1,
        save_intermediate_features=True,
        add_fcn_layer_size=2
    )
    model = TemplateConvNet2d(cfg)
    x = torch.randn(3, 1, 8, 8)
    out, feats = model(x)
    # After conv3 -> 6x6, pool2 -> 3x3, adaptive avg->1x1, flatten, linear->2
    assert out.shape == (3, 2)
    assert len(feats) == 1
    assert feats[0].shape == (3, 4, 3, 3)

# %% TemplateConvNet2d tests
def test_template_convnet2d_build_and_forward_default_no_skips():
    cfg = TemplateConvNet2dConfig(
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
    cfg = TemplateConvNet2dConfig(
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
    cfg = TemplateConvNet2dConfig(
        kernel_sizes=kernels,
        pool_kernel_sizes=pools,
        out_channels_sizes=[4],
        num_input_channels=1,
    )
    with pytest.raises(ValueError) as exc:
        TemplateConvNet2d(cfg)
    assert msg in str(exc.value)


def test_template_convnet2d_mismatched_kernel_pool_length():
    cfg = TemplateConvNet2dConfig(
        kernel_sizes=[3, 5],
        pool_kernel_sizes=[2],
        out_channels_sizes=[4, 8],
        num_input_channels=1,
    )
    with pytest.raises(ValueError) as exc:
        TemplateConvNet2d(cfg)
    assert "must have the same length" in str(exc.value)


def test_template_convnet2d_scalar_pool_raises():
    cfg = TemplateConvNet2dConfig(
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
    cfg = TemplateConvNet2dConfig(
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
    # Should store the wrapped model and default ensemble size
    assert isinstance(wrapper.base_model, nn.Module)
    assert wrapper.ensemble_size == 20

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

# %% TemplateFullyConnectedNetConfig tests
def test_template_fcnet_config_valid_defaults():
    cfg = TemplateFullyConnectedNetConfig(
        out_channels_sizes=[10, 20, 30],
        input_layer_size=5,
        output_layer_size=15,
        regularized_param=0.1,
        regularization_layer_type="dropout",
        dropout_ensemble_size=3
    )
    assert cfg.out_channels_sizes == [10, 20, 30]
    assert cfg.input_layer_size == 5
    assert cfg.output_layer_size == 15
    assert cfg.regularized_param == 0.1
    assert cfg.regularization_layer_type == "dropout"
    assert cfg.dropout_ensemble_size == 3

def test_template_fcnet_config_out_channels_none():
    with pytest.raises(ValueError) as exc:
        TemplateFullyConnectedNetConfig(
            input_layer_size=4
        )
    assert "out_channels_sizes" in str(exc.value)

def test_template_fcnet_config_input_layer_none():
    with pytest.raises(ValueError) as exc:
        TemplateFullyConnectedNetConfig(
            out_channels_sizes=[5, 6]
        )
    assert "input_layer_size" in str(exc.value)

def test_template_fcnet_config_output_layer_defaults_to_last(capsys):
    cfg = TemplateFullyConnectedNetConfig(
        out_channels_sizes=[7, 8, 9],
        input_layer_size=3
    )
    captured = capsys.readouterr()
    assert cfg.output_layer_size == 9
    assert "Setting to last value of 'out_channels_sizes'" in captured.out

def test_template_fcnet_config_dropout_ensemble_invalid():
    with pytest.raises(ValueError):
        TemplateFullyConnectedNetConfig(
            out_channels_sizes=[4, 5],
            input_layer_size=2,
            dropout_ensemble_size=2
        )

def test_template_fcnet_config_input_skip_index_length_exceeds():
    with pytest.raises(ValueError) as exc:
        TemplateFullyConnectedNetConfig(
            out_channels_sizes=[3, 4],
            input_layer_size=2,
            input_skip_index=list(range(10))
        )
    assert "input_skip_index" in str(exc.value)

def test_template_fcnet_config_input_skip_index_valid():
    skip_idx = [0, 1]
    cfg = TemplateFullyConnectedNetConfig(
        out_channels_sizes=[3, 4, 5],
        input_layer_size=2,
        input_skip_index=skip_idx
    )
    assert cfg.input_skip_index == skip_idx

# %% TemplateFullyConnectedNet tests
def test_fcnet_forward_simple():
    cfg = TemplateFullyConnectedNetConfig(
        out_channels_sizes=[8, 4],
        input_layer_size=3,
        output_layer_size=4,
    )
    model = TemplateFullyConnectedNet(cfg)
    x = torch.randn(5, 3)
    out = model(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (5, 4)

def test_fcnet_forward_with_empty_skip():
    cfg = TemplateFullyConnectedNetConfig(
        out_channels_sizes=[6],
        input_layer_size=2,
        output_layer_size=6,
    )
    model = TemplateFullyConnectedNet(cfg)
    x = torch.randn(4, 2)
    skips = torch.empty(0, 2)
    out = model((x, skips))
    # skip empty => no change in batch size
    assert out.shape == (4, 6)

def test_fcnet_forward_with_nonempty_skip():
    cfg = TemplateFullyConnectedNetConfig(
        out_channels_sizes=[5],
        input_layer_size=2,
        output_layer_size=5,
    )
    model = TemplateFullyConnectedNet(cfg)
    x = torch.randn(3, 2)
    skips = torch.randn(2, 2)
    out = model((x, skips))
    # batch size increases by len(skips)
    assert out.shape == (5, 5)
def test_net_forward_shape_and_layers():
    cfg = TemplateFullyConnectedNetConfig(
        out_channels_sizes=[7, 5],
        input_layer_size=4,
        regularization_layer_type="none",
    )
    model = TemplateFullyConnectedNet(cfg)
    # Expect layers: Flatten, Linear(4->7), PReLU, Linear(7->5)
    x = torch.randn(2, 4)
    out = model(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 5)

# %% DropoutEnsemblingNetworkWrapper tests with TemplateFullyConnectedNet
def test_wrapper_requires_nn_module():
    with pytest.raises(TypeError):
        DropoutEnsemblingNetworkWrapper(model=123, ensemble_size=5)

def test_wrapper_get_last_stats_no_forward():
    # wrap a simple linear model
    base = nn.Linear(2, 2)
    wrapper = DropoutEnsemblingNetworkWrapper(base, ensemble_size=3)
    with pytest.raises(ValueError):
        wrapper.get_last_stats()


@pytest.mark.parametrize("batch_size", [1, 4])
def test_wrapper_train_and_eval_stats(batch_size):
    torch.manual_seed(0)
    base = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Dropout(0.5))
    wrapper = DropoutEnsemblingNetworkWrapper(base, ensemble_size=10)
    x = torch.randn(batch_size, 2)

    # in training mode: no ensembling, output equals base(x)
    wrapper.train()
    out_train = wrapper(x)
    mean_t, med_t, var_t = wrapper.get_last_stats()
    assert torch.allclose(out_train, mean_t)
    assert torch.allclose(out_train, med_t)
    assert torch.all(var_t == 0), "Variance in training should be zero"

    # in eval mode: ensembling enabled
    wrapper.eval()
    out_eval = wrapper(x)
    m, med, v = wrapper.get_last_stats()
    # mean returned should match out_eval
    assert torch.allclose(out_eval, m)
    # median and variance should have same shape
    assert med.shape == out_eval.shape
    assert v.shape == out_eval.shape

def test_build_dropout_ensemble_static():
    cfg = TemplateFullyConnectedNetConfig(
        out_channels_sizes=[4],
        input_layer_size=2,
        regularization_layer_type="dropout",
        regularized_param=0.2,
        dropout_ensemble_size=5,
    )
    wrapper = TemplateFullyConnectedNet.build_dropout_ensemble(
        TemplateFullyConnectedNet, cfg)
    assert isinstance(wrapper, DropoutEnsemblingNetworkWrapper)
    # ensemble size propagated
    assert wrapper.ensemble_size == 5

# %% TemplateConvNetFeatureFuser2dConfig tests


def test_template_convnet_feature_fuser2d_config_expand_types_and_heads():
    cfg = TemplateConvNetFeatureFuser2dConfig(
        kernel_sizes=[3],
        pool_type="MaxPool2d",
        pool_kernel_sizes=[2],
        out_channels_sizes=[4],
        num_input_channels=1,
        num_skip_channels=[1],
        merge_module_index=[0],
        merge_module_type=["feature_add"],
        num_attention_heads=[2],
    )
    # single entry lists stay same length
    assert cfg.merge_module_type == ["feature_add"]
    assert cfg.num_attention_heads == [2]


@pytest.mark.parametrize("nv,mi,mt", [
    ([], [0], ["identity"]),
    ([1], [], ["identity"]),
    ([1], [0], []),
])
def test_template_convnet_feature_fuser2d_config_empty_lists_raise(nv, mi, mt):
    with pytest.raises(ValueError):
        TemplateConvNetFeatureFuser2dConfig(
            kernel_sizes=[3],
            pool_type="MaxPool2d",
            pool_kernel_sizes=[2],
            out_channels_sizes=[4],
            num_input_channels=1,
            num_skip_channels=nv,
            merge_module_index=mi,
            merge_module_type=mt,
        )


@pytest.mark.parametrize("nskip, midx, mtype", [
    ([1, 2], [0], ["identity"]),
    ([1], [0, 1], ["identity"]),
    ([1], [0], ["add", "concat"]),
])
def test_template_convnet_feature_fuser2d_config_mismatch_lengths_raise(nskip, midx, mtype):
    with pytest.raises(ValueError):
        TemplateConvNetFeatureFuser2dConfig(
            kernel_sizes=[3],
            pool_type="MaxPool2d",
            pool_kernel_sizes=[2],
            out_channels_sizes=[4],
            num_input_channels=1,
            num_skip_channels=nskip,
            merge_module_index=midx,
            merge_module_type=mtype,
        )

# %% TemplateConvNetFeatureFuser2d tests
def test_template_convnet_feature_fuser2d_forward_identity_fuser():
    cfg = TemplateConvNetFeatureFuser2dConfig(
        kernel_sizes=[3],
        pool_type="MaxPool2d",
        pool_kernel_sizes=[2],
        out_channels_sizes=[4],
        num_input_channels=1,
        num_skip_channels=[1],
        merge_module_index=[0],
        merge_module_type=["identity"],
    )
    model = TemplateConvNetFeatureFuser2d(cfg)
    # x: 3-> conv3->5, pool2->2
    x = torch.randn(2, 1, 6, 6)
    # skip unused by identity fuser
    skip = torch.randn(2, 1, 6, 6)
    out, feats = model((x, [skip]))
    # After one block conv->2x2
    assert out.shape == (2, 4, 2, 2)
    assert feats == []


def test_template_convnet_feature_fuser2d_forward_with_intermediate_features():
    cfg = TemplateConvNetFeatureFuser2dConfig(
        kernel_sizes=[3, 3],
        pool_type="AvgPool2d",
        pool_kernel_sizes=[2, 2],
        out_channels_sizes=[4, 5],
        num_input_channels=1,
        num_skip_channels=[1, 1],
        merge_module_index=[0, 1],
        merge_module_type=["identity", "identity"],
        save_intermediate_features=True
    )
    model = TemplateConvNetFeatureFuser2d(cfg)
    x = torch.randn(1, 1, 8, 8)
    skips = [torch.randn_like(x), torch.randn_like(x)]
    out, feats = model((x, skips))
    # Two blocks -> two feature entries
    assert len(feats) == 2
    assert out.shape[1:] == feats[-1].shape[1:]

# %% MANUAL CALL
if __name__ == "__main__":
    pytest.main([__file__])