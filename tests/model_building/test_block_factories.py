import pytest
import torch
import torch.nn as nn

from pyTorchAutoForge.model_building.factories.block_factories import (
    _initialize_convblock_weights,
    _initialize_fcnblock_weights,
    _activation_factory,
    _pooling_factory,
    _regularizer_factory,
)

# Dummy classes for testing init functions
class DummyConvBlock:
    def __init__(self):
        self.conv = nn.Conv2d(3, 6, kernel_size=3, bias=True)
        nn.init.constant_(self.conv.weight, 1.0)
        nn.init.constant_(self.conv.bias, 1.0)

class DummyFCBlock:
    def __init__(self):
        self.linear = nn.Linear(4, 5, bias=True)
        nn.init.constant_(self.linear.weight, 1.0)
        nn.init.constant_(self.linear.bias, 1.0)

def _is_dirac_weight(weight: torch.Tensor) -> bool:
    # dirac_ sets a 1 on the central spatial position for matching in/out channels
    if weight.dim() < 3:
        return False

    expected = torch.zeros_like(weight)
    center = tuple(size // 2 for size in weight.shape[2:])
    for c in range(min(weight.shape[0], weight.shape[1])):
        expected[(c, c) + center] = 1.0

    return torch.allclose(weight, expected)

@pytest.mark.parametrize("method,check", [
    ("xavier_uniform", lambda w: torch.any(w != 1)),
    ("kaiming_uniform", lambda w: torch.any(w != 1)),
    ("xavier_normal", lambda w: torch.any(w != 1)),
    ("kaiming_normal", lambda w: torch.any(w != 1)),
    ("orthogonal", lambda w: torch.any(w != 1)),
    ("zero", lambda w: torch.allclose(w, torch.zeros_like(w))),
    ("identity", _is_dirac_weight),
])
def test_initialize_convblock_weights_methods(method, check):
    block = DummyConvBlock()
    _initialize_convblock_weights(block, init_method_type=method)
    # bias should be zeroed
    assert torch.all(block.conv.bias == 0)
    assert check(block.conv.weight)

@pytest.mark.parametrize("method,check", [
    ("xavier_uniform", lambda w: torch.any(w != 1)),
    ("kaiming_uniform", lambda w: torch.any(w != 1)),
    ("xavier_normal", lambda w: torch.any(w != 1)),
    ("kaiming_normal", lambda w: torch.any(w != 1)),
    ("orthogonal", lambda w: torch.any(w != 1)),
    ("zero", lambda w: torch.allclose(w, torch.zeros_like(w))),
    ("identity", lambda w: torch.allclose(
        w,
        torch.eye(w.size(0), w.size(1), device=w.device, dtype=w.dtype)
    )),
])
def test_initialize_fcnblock_weights_methods(method, check):
    block = DummyFCBlock()
    _initialize_fcnblock_weights(block, init_method_type=method)
    assert torch.all(block.linear.bias == 0)
    assert check(block.linear.weight)

@pytest.mark.parametrize("activ,cls", [
    ("prelu", nn.PReLU),
    ("leakyrelu", nn.LeakyReLU),
    ("relu", nn.ReLU),
    ("elu", nn.ELU),
    ("selu", nn.SELU),
    ("gelu", nn.GELU),
    ("swish", nn.SiLU),
    ("softplus", nn.Softplus),
    ("sigmoid", nn.Sigmoid),
    ("tanh", nn.Tanh),
    ("none", nn.Identity),
])
def test_activation_factory_valid(activ, cls):
    layer = _activation_factory(activ, out_channels=8, prelu_params="all")
    assert isinstance(layer, cls)

def test_activation_factory_prelu_param_counts():
    a_unique = _activation_factory("prelu", out_channels=10, prelu_params="unique")
    a_all = _activation_factory("prelu", out_channels=10, prelu_params="all")

    # Assert output is a prelu
    assert isinstance(a_unique, nn.PReLU)
    assert isinstance(a_all, nn.PReLU)

    assert a_unique.num_parameters == 1
    assert a_all.num_parameters == 10

def test_activation_factory_invalid():
    with pytest.raises(ValueError):
        _activation_factory("invalid", out_channels=4, prelu_params="all")

@pytest.mark.parametrize("pool_type,cls", [
    ("MaxPool1d", nn.MaxPool1d),
    ("AvgPool1d", nn.AvgPool1d),
    ("MaxPool2d", nn.MaxPool2d),
    ("AvgPool2d", nn.AvgPool2d),
    ("MaxPool3d", nn.MaxPool3d),
    ("AvgPool3d", nn.AvgPool3d),
    ("Adapt_MaxPool1d", nn.AdaptiveMaxPool1d),
    ("Adapt_AvgPool2d", nn.AdaptiveAvgPool2d),
])
def test_pooling_factory_valid(pool_type, cls):
    if pool_type.startswith("Adapt_"):
        layer = _pooling_factory(pool_type, kernel_size=2, target_res=3)
    else:
        layer = _pooling_factory(pool_type, kernel_size=2)
    assert isinstance(layer, cls)

def test_pooling_factory_missing_target():
    with pytest.raises(ValueError):
        _pooling_factory("Adapt_MaxPool2d", kernel_size=2)

def test_pooling_factory_invalid():
    with pytest.raises(ValueError):
        _pooling_factory("invalid", kernel_size=1)

@pytest.mark.parametrize("ndims,rtype,param,cls", [
    (1, "dropout", 0.5, nn.Dropout),
    (2, "dropout", 0.3, nn.Dropout2d),
    (3, "dropout", 0.2, nn.Dropout3d),
    (1, "batchnorm", None, nn.BatchNorm1d),
    (2, "batchnorm", None, nn.BatchNorm2d),
    (3, "batchnorm", None, nn.BatchNorm3d),
    (1, "groupnorm", 1, nn.GroupNorm),
    (1, "layernorm", None, nn.LayerNorm),
    (2, "instancenorm", None, nn.InstanceNorm2d),
    (3, "instancenorm", None, nn.InstanceNorm3d),
    (2, "none", None, nn.Identity),
])
def test_regularizer_factory_valid(ndims, rtype, param, cls):
    layer = _regularizer_factory(ndims, rtype, out_channels=4, regularizer_param=param)
    assert isinstance(layer, cls)

@pytest.mark.parametrize("param", [None, -0.1, 1.0])
def test_regularizer_dropout_invalid(param):
    with pytest.raises(ValueError):
        _regularizer_factory(1, "dropout", out_channels=4, regularizer_param=param)

@pytest.mark.parametrize("param", [None, 0, 1.5])
def test_regularizer_groupnorm_invalid(param):
    with pytest.raises(ValueError):
        _regularizer_factory(1, "groupnorm", out_channels=4, regularizer_param=param)

def test_regularizer_groupnorm_divisibility():
    with pytest.raises(ValueError):
        _regularizer_factory(1, "groupnorm", out_channels=5, regularizer_param=2)

def test_regularizer_invalid_type():
    with pytest.raises(ValueError):
        _regularizer_factory(1, "invalid", out_channels=4, regularizer_param=None)

# %% Edge cases tests
def test_pooling_factory_stride_and_padding_defaults():

    kernel_size = (3, 3)
    layer = _pooling_factory("MaxPool2d", kernel_size=kernel_size)
    assert isinstance(layer, nn.MaxPool2d)

    # Default stride should be 1
    assert layer.stride == kernel_size[0] or layer.stride == kernel_size
    # Default padding should be 0
    assert layer.padding == 0 or layer.padding == (0, 0)

def test_pooling_factory_case_insensitive():
    layer = _pooling_factory("mAxPoOl1D", kernel_size=2, stride=2, padding=1)
    assert isinstance(layer, nn.MaxPool1d)
    assert layer.stride == 2
    assert layer.padding == 1

def test_pooling_factory_unsupported_adaptive_type():
    with pytest.raises(ValueError):
        _pooling_factory("Adapt_UnknownPool1d", kernel_size=2, target_res=5)

@pytest.mark.parametrize("ndims", [0, 4])
def test_regularizer_dropout_ndims_unsupported(ndims):
    with pytest.raises(ValueError):
        _regularizer_factory(
            ndims, "dropout", out_channels=3, regularizer_param=0.5)


@pytest.mark.parametrize("ndims", [0, 4])
def test_regularizer_batchnorm_ndims_unsupported(ndims):
    with pytest.raises(ValueError):
        _regularizer_factory(ndims, "batchnorm",
                             out_channels=3, regularizer_param=None)


@pytest.mark.parametrize("ndims", [0, 4])
def test_regularizer_instancenorm_ndims_unsupported(ndims):
    with pytest.raises(ValueError):
        _regularizer_factory(ndims, "instancenorm",
                             out_channels=3, regularizer_param=None)


def test_activation_factory_prelu_invalid_prelu_param():
    layer = _activation_factory(
        "prelu", out_channels=5, prelu_params="invalid")
    # Invalid prelu_params should default to a single parameter
    assert isinstance(layer, nn.PReLU)
    assert layer.num_parameters == 1
