import pytest
import torch
from torch import nn

from pyTorchAutoForge.model_building import ConvolutionalBlock1d, ConvolutionalBlock2d, ConvolutionalBlock3d

# %% Tests for ConvolutionalBlock1d
def test_convolutionalblock1d_invalid_groupnorm_divisibility():
    with pytest.raises(ValueError):
        ConvolutionalBlock1d(
            in_channels=4,
            out_channels=5,         # 5 channels not divisible by 3 groups
            kernel_size=3,
            pool_type="none",
            activ_type="none",
            regularizer_type="groupnorm",
            regularizer_param=3,
        )

def test_convolutionalblock1d_adaptive_pooling():
    x = torch.randn(2, 4, 20)
    block = ConvolutionalBlock1d(
        in_channels=4,
        out_channels=4,
        kernel_size=3,
        pool_type="Adapt_MaxPool1d",
        activ_type="relu",
        regularizer_type="none",
        pool_kernel_size=2,
        regularizer_param=0.0,
        target_res=5
    )
    y = block(x)
    assert y.shape[-1] == 5

def test_convolutionalblock1d_invalid_activation():
    with pytest.raises(ValueError):
        ConvolutionalBlock1d(2, 2, 3, activ_type="invalid")

def test_convolutionalblock1d_invalid_pool():
    with pytest.raises(ValueError):
        ConvolutionalBlock1d(2, 2, 3, pool_type="invalid")

def test_convolutionalblock1d_invalid_regularizer():
    with pytest.raises(ValueError):
        ConvolutionalBlock1d(2, 2, 3, regularizer_type="invalid")

def test_convolutionalblock1d_adaptive_pooling_no_target():
    with pytest.raises(ValueError):
        ConvolutionalBlock1d(2, 2, 3, pool_type="Adapt_MaxPool1d")

# Parametric tests
@pytest.mark.parametrize("activ_type", ["prelu", "relu", "sigmoid", "tanh", "none"])
@pytest.mark.parametrize("pool_type", ["MaxPool1d", "AvgPool1d", "none"])
@pytest.mark.parametrize("regularizer_type, regularizer_param", [
    ("none", 0.0),
    ("dropout", 0.5),
    ("batchnorm", 0.0),
    ("groupnorm", 2),
])
def test_convolutionalblock1d_forward(activ_type, pool_type, regularizer_type, regularizer_param):

    x = torch.randn(4, 8, 32)
    block = ConvolutionalBlock1d(
        in_channels=8,
        out_channels=16,
        kernel_size=3,
        pool_kernel_size=2,
        pool_type=pool_type,
        activ_type=activ_type,
        regularizer_type=regularizer_type,
        regularizer_param=regularizer_param if regularizer_type != "none" else 0.0,
    )

    y = block(x)
    assert isinstance(y, torch.Tensor)
    assert y.shape[0] == 4
    assert y.shape[1] == 16

# %% Tests for ConvolutionalBlock2d
def test_convolutionalblock2d_invalid_groupnorm_divisibility():
    with pytest.raises(ValueError):
        ConvolutionalBlock2d(
            in_channels=4,
            out_channels=5,         # 5 channels not divisible by 3 groups
            kernel_size=3,
            pool_type="none",
            activ_type="none",
            regularizer_type="groupnorm",
            regularizer_param=3,
        )

def test_convolutionalblock2d_adaptive_pooling():
    x = torch.randn(1, 2, 10, 10)
    block = ConvolutionalBlock2d(
        in_channels=2,
        out_channels=2,
        kernel_size=3,
        pool_type="Adapt_MaxPool2d",
        activ_type="relu",
        regularizer_type="none",
        pool_kernel_size=2,
        regularizer_param=0.0,
        target_res=(4, 4)
    )
    y = block(x)
    assert y.shape[-2:] == (4, 4)

def test_convolutionalblock2d_invalid_activation():
    with pytest.raises(ValueError):
        ConvolutionalBlock2d(2, 2, 3, activ_type="invalid")

def test_convolutionalblock2d_invalid_pool():
    with pytest.raises(ValueError):
        ConvolutionalBlock2d(2, 2, 3, pool_type="invalid")

def test_convolutionalblock2d_invalid_regularizer():
    with pytest.raises(ValueError):
        ConvolutionalBlock2d(2, 2, 3, regularizer_type="invalid")

def test_convolutionalblock2d_adaptive_pooling_no_target():
    with pytest.raises(ValueError):
        ConvolutionalBlock2d(2, 2, 3, pool_type="Adapt_MaxPool2d")

# Parametric tests
@pytest.mark.parametrize("activ_type", ["prelu", "relu", "sigmoid", "tanh", "none"])
@pytest.mark.parametrize("pool_type", ["MaxPool2d", "AvgPool2d", "none"])
@pytest.mark.parametrize("regularizer_type,regularizer_param", [
    ("none", 0.0),
    ("dropout", 0.5),
    ("batchnorm", 0.0),
    ("groupnorm", 2),
])
def test_convolutionalblock2d_forward(activ_type, pool_type, regularizer_type, regularizer_param):
    x = torch.randn(2, 3, 16, 16)
    block = ConvolutionalBlock2d(
        in_channels=3,
        out_channels=6,
        kernel_size=3,
        pool_kernel_size=2,
        pool_type=pool_type,
        activ_type=activ_type,
        regularizer_type=regularizer_type,
        regularizer_param=regularizer_param if regularizer_type != "none" else 0.0,
    )
    y = block(x)
    assert isinstance(y, torch.Tensor)
    assert y.shape[0] == 2
    assert y.shape[1] == 6


# %% Tests for ConvolutionalBlock3d
def test_convolutionalblock3d_invalid_groupnorm_divisibility():
    with pytest.raises(ValueError):
        ConvolutionalBlock3d(
            in_channels=4,
            out_channels=5,         # 5 channels not divisible by 3 groups
            kernel_size=3,
            pool_type="none",
            activ_type="none",
            regularizer_type="groupnorm",
            regularizer_param=3,
        )

def test_convolutionalblock3d_adaptive_pooling():
    x = torch.randn(1, 2, 6, 6, 6)
    block = ConvolutionalBlock3d(
        in_channels=2,
        out_channels=2,
        kernel_size=3,
        pool_type="Adapt_MaxPool3d",
        activ_type="relu",
        regularizer_type="none",
        pool_kernel_size=2,
        regularizer_param=0.0,
        target_res=(2, 2, 2)
    )
    y = block(x)
    assert y.shape[-3:] == (2, 2, 2)

def test_convolutionalblock3d_invalid_activation():
    with pytest.raises(ValueError):
        ConvolutionalBlock3d(2, 2, 3, activ_type="invalid")

def test_convolutionalblock3d_invalid_pool():
    with pytest.raises(ValueError):
        ConvolutionalBlock3d(2, 2, 3, pool_type="invalid")

def test_convolutionalblock3d_invalid_regularizer():
    with pytest.raises(ValueError):
        ConvolutionalBlock3d(2, 2, 3, regularizer_type="invalid")

def test_convolutionalblock3d_adaptive_pooling_no_target():
    with pytest.raises(ValueError):
        ConvolutionalBlock3d(2, 2, 3, pool_type="Adapt_MaxPool3d")


# Parametric tests
@pytest.mark.parametrize("activ_type", ["prelu", "relu", "sigmoid", "tanh", "none"])
@pytest.mark.parametrize("pool_type", ["MaxPool3d", "AvgPool3d", "none"])
@pytest.mark.parametrize("regularizer_type, regularizer_param", [
    ("none", 0.0),
    ("dropout", 0.5),
    ("batchnorm", 0.0),
    ("groupnorm", 2),
])
def test_convolutionalblock3d_forward(activ_type, 
                                      pool_type, 
                                      regularizer_type, regularizer_param):
    x = torch.randn(2, 3, 8, 8, 8)
    out_channels = 6

    block = ConvolutionalBlock3d(
        in_channels=3,
        out_channels=out_channels,
        kernel_size=3,
        pool_kernel_size=2,
        pool_type=pool_type,
        activ_type=activ_type,
        regularizer_type=regularizer_type,
        regularizer_param=regularizer_param if regularizer_type != "none" else 0.0,
    )
    y = block(x)
    assert isinstance(y, torch.Tensor)
    assert y.shape[0] == 2
    assert y.shape[1] == out_channels


# %% Additional tests for Adaptive Average Pooling variants
def test_convolutionalblock1d_adaptive_avg_pooling():
    x = torch.randn(3, 5, 30)
    block = ConvolutionalBlock1d(
        in_channels=5,
        out_channels=5,
        kernel_size=3,
        pool_type="Adapt_AvgPool1d",
        activ_type="none",
        regularizer_type="none",
        pool_kernel_size=2,
        regularizer_param=0.0,
        target_res=10
    )
    y = block(x)
    assert y.shape[-1] == 10

def test_convolutionalblock2d_adaptive_avg_pooling():
    x = torch.randn(2, 4, 20, 15)
    block = ConvolutionalBlock2d(
        in_channels=4,
        out_channels=4,
        kernel_size=3,
        pool_type="Adapt_AvgPool2d",
        activ_type="none",
        regularizer_type="none",
        pool_kernel_size=2,
        regularizer_param=0.0,
        target_res=(8, 5)
    )
    y = block(x)
    assert y.shape[-2:] == (8, 5)

def test_convolutionalblock3d_adaptive_avg_pooling():
    x = torch.randn(1, 3, 12, 10, 8)
    block = ConvolutionalBlock3d(
        in_channels=3,
        out_channels=3,
        kernel_size=3,
        pool_type="Adapt_AvgPool3d",
        activ_type="none",
        regularizer_type="none",
        pool_kernel_size=2,
        regularizer_param=0.0,
        target_res=(6, 5, 4)
    )
    y = block(x)
    assert y.shape[-3:] == (6, 5, 4)


# %% FeatureMapFuser tests
import pytest
import torch
from torch import nn
from pyTorchAutoForge.model_building.convolutionalBlocks import (
    FeatureMapFuser,
    FeatureMapFuserConfig,
    _feature_map_fuser_factory
)

# --- Feature-add fusion tests ---
def test_feature_add_fuser_1d_same_shape():
    """Test feature-add fuser for 1D inputs with same shape."""
    
    fuser = FeatureMapFuser(num_dims=2, fuser_type="feature_add", in_channels=1)
    
    x = torch.randn(3, 5)
    skip = torch.randn_like(x)
    out = fuser(x, skip)

    # Asserts
    assert out.shape == x.shape
    assert torch.allclose(out, x + skip)

def test_feature_add_fuser_2d_same_shape():
    """Test feature-add fuser for 2D inputs with same shape."""

    fuser = FeatureMapFuser(num_dims=4, fuser_type="feature_add", in_channels=3)
    
    x = torch.randn(2, 3, 4, 4)
    skip = torch.randn_like(x)
    out = fuser(x, skip)

    # Assets
    assert out.shape == x.shape
    assert torch.allclose(out, x + skip)

def test_feature_add_fuser_3d_same_shape():
    fuser = FeatureMapFuser(num_dims=5, fuser_type="feature_add", in_channels=2)
    
    x = torch.randn(1, 2, 3, 4, 5)
    skip = torch.randn_like(x)
    out = fuser(x, skip)
    
    # Asserts
    assert out.shape == x.shape
    assert torch.allclose(out, x + skip)

# --- Add fuser with interpolation based resampling tests ---
def test_identity_fuser_ignores_skip_and_returns_x():
    # identity should return x unchanged, even if skip has different shape
    fuser = FeatureMapFuser(num_dims=4, fuser_type="identity", in_channels=3)
    x = torch.randn(2, 3, 5, 5)
    skip = torch.randn(2, 3, 8, 8)
    out = fuser(x, skip)
    assert out.shape == x.shape
    assert torch.allclose(out, x)

def test_feature_add_fuser_interpolates_skip_2d():
    # x zeros, skip ones at double spatial resolution => output should be ones
    fuser = FeatureMapFuser(num_dims=4, fuser_type="feature_add", in_channels=1)
    x = torch.zeros(1, 1, 2, 2)
    skip = torch.ones(1, 1, 4, 4)
    out = fuser(x, skip)
    assert out.shape == x.shape
    assert torch.allclose(out, torch.ones_like(x))

def test_invalid_num_dims_for_fuser_raises_value_error():
    # num_dims must be 2, 4, or 5
    with pytest.raises(ValueError):
        FeatureMapFuser(num_dims=3, fuser_type="feature_add", in_channels=1)
    with pytest.raises(ValueError):
        FeatureMapFuser(num_dims=3, fuser_type="multihead_attention",
                        in_channels=2, num_attention_heads=1)

# --- Channel-concat fusion tests ---
def test_channel_concat_fuser_2d_shape_and_projection():
    in_ch, skip_ch = 4, 2
    fuser = FeatureMapFuser(num_dims=4,
                             fuser_type="channel_concat",
                             in_channels=in_ch,
                             num_skip_channels=skip_ch)
    
    # Fix conv1x1 weights to sum channels equally
    with torch.no_grad():
        fuser.proj.weight.fill_(1.0 / (in_ch + skip_ch))
        fuser.proj.bias.zero_()

    x = torch.randn(2, in_ch, 3, 3)
    skip = torch.randn(2, skip_ch, 3, 3)
    out = fuser(x, skip)
    assert out.shape == x.shape

    # Test that output is close to average across concatenated channels
    concat = torch.cat([x, skip], dim=1)
    expected = torch.sum(concat, dim=1, keepdim=True) / (in_ch + skip_ch)
    
    # Project back over each output channel should match expected repeated
    for c in range(in_ch):
        assert torch.allclose(out[:, c:c+1, :, :], expected, atol=1e-6)

def test_channel_concat_missing_skip_channels_raises():
    with pytest.raises(AssertionError):
        FeatureMapFuser(num_dims=4, fuser_type="channel_concat", in_channels=3)

def test_channel_concat_invalid_dims_raises():
    with pytest.raises(ValueError):
        FeatureMapFuser(num_dims=3,
                        fuser_type="channel_concat",
                        in_channels=3,
                        num_skip_channels=1)

def test_channel_concat_fuser_interpolation_and_projection():
    # x zeros, skip ones at double spatial resolution
    # proj weights pick only skip channels => each output element = sum(skip channels)=2
    in_ch, skip_ch = 2, 2
    fuser = FeatureMapFuser(num_dims=4,
                                fuser_type="channel_concat",
                                in_channels=in_ch,
                                num_skip_channels=skip_ch)
    
    # Overwrite proj to pick only skip-ch inputs
    with torch.no_grad():
        # proj.weight shape [out_ch, in_tot, 1,1]
        fuser.proj.weight.zero_()
        # for each output channel, set weights for skip inputs to 1
        fuser.proj.weight[:, in_ch:in_ch+skip_ch, 0, 0].fill_(1.0)
        fuser.proj.bias.zero_()

    x = torch.zeros(1, in_ch, 3, 3)
    skip = torch.ones(1, skip_ch, 6, 6)
    out = fuser(x, skip)
    assert out.shape == x.shape
    # each output value == sum of skip channels == 2
    assert torch.allclose(out, torch.full_like(out, float(skip_ch)))

# --- Multi-head attention fusion tests ---
def test_multihead_attention_fuser_shape():
    in_ch = 3
    heads = 1
    fuser = FeatureMapFuser(num_dims=4,
                             fuser_type="multihead_attention",
                             in_channels=in_ch,
                             num_attention_heads=heads)
    x = torch.randn(2, in_ch, 2, 2)
    skip = torch.randn_like(x)
    out = fuser(x, skip)
    assert out.shape == x.shape

def test_multihead_attention_missing_heads_raises():
    with pytest.raises(AssertionError):
        FeatureMapFuser(num_dims=4,
                        fuser_type="multihead_attention",
                        in_channels=3)

def test_multihead_attention_fuser_with_resample_before_attention_runs_and_shapes():
    # ensure that resample_before_attention path works without error
    in_ch, heads = 3, 1
    fuser = FeatureMapFuser(num_dims=4,
                                fuser_type="multihead_attention",
                                in_channels=in_ch,
                                num_attention_heads=heads,
                                resample_before_attention=True)
    
    x = torch.randn(2, in_ch, 2, 2)
    skip = torch.randn(2, in_ch, 4, 4)
    out = fuser(x, skip)
    assert out.shape == x.shape

# --- Factory tests ---
def test_feature_map_fuser_factory_creates_correct_instance():
    cfg = FeatureMapFuserConfig(
        in_channels=5,
        num_skip_channels=2,
        num_dims=4,
        fuser_module_type="channel_concat",
        num_attention_heads=None
    )
    fuser = _feature_map_fuser_factory(cfg)
    assert isinstance(fuser, FeatureMapFuser)

def test_factory_passes_kwargs_to_underlying():
    cfg = FeatureMapFuserConfig(in_channels=1,
                                num_skip_channels=1,
                                num_dims=2,
                                fuser_module_type="feature_add")
    fuser = _feature_map_fuser_factory(cfg, mode="linear")
    # should perform 1D feature-add with 'linear' mode
    x = torch.randn(2, 4)
    skip = torch.randn_like(x)
    out = fuser(x, skip)
    assert torch.allclose(out, x + skip)

def test_unknown_fuser_type_raises_value_error():
    with pytest.raises(ValueError):
        FeatureMapFuser(num_dims=4,
                         fuser_type="unknown_mode",
                         in_channels=3,
                         num_skip_channels=1)

# %% MANUAL CALLS FOR DEBUG
if __name__ == "__main__":
    #pytest.main([__file__])
    test_feature_add_fuser_1d_same_shape()