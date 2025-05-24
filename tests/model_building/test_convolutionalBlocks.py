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
