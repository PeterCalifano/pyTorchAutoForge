import torch
from torchvision import models
from pyTorchAutoForge.model_building.backbones.efficient_net import EfficientNetConfig, FeatureExtractorFactory, EfficientNetBackbone

import pytest
import numpy as np

import matplotlib
# NOTE: Set the backend to 'Agg' for non-interactive use
matplotlib.use('agg')
import matplotlib.pyplot as plt

from pyTorchAutoForge.model_building.backbones.input_adapters import (
    Conv2dAdapterConfig,
    ResizeAdapterConfig,
    ImageMaskFilterAdapterConfig,
    InputAdapterFactory,
    Conv2dResolutionChannelsAdapter,
    ResizeCopyChannelsAdapter,
    ImageMaskFilterAdapter,
    ScalerAdapter,
    ScalerAdapterConfig,
)

# TODO (PC) add tests for new functionalities in ScalerAdapter and ScalerAdapterConfig!

def test_conv2d_adapter_forward_and_factory_dispatch():
    # Prepare a float64 input to test dtype casting and resizing
    batch, in_ch, H_in, W_in = 2, 3, 12, 16
    out_size = (6, 8)
    out_ch = 5

    x = torch.randn(batch, in_ch, H_in, W_in, dtype=torch.float64)
    cfg = Conv2dAdapterConfig(output_size=out_size, channel_sizes=(in_ch, out_ch))
    
    # Factory should return the correct adapter type
    adapter = InputAdapterFactory(cfg)
    
    assert isinstance(adapter, Conv2dResolutionChannelsAdapter)
    
    y = adapter(x)

    assert y.shape == (batch, out_ch, *out_size)
    assert y.dtype == torch.float32

def test_resize_copy_channels_adapter_forward_and_factory_dispatch():
    batch, in_ch, H_in, W_in = 2, 1, 10, 14
    out_size = (5, 7)
    out_ch = 4
    x = torch.rand(batch, in_ch, H_in, W_in)
    cfg = ResizeAdapterConfig(output_size=out_size, channel_sizes=(in_ch, out_ch), interp_method='bilinear')
    adapter = InputAdapterFactory(cfg)
    assert isinstance(adapter, ResizeCopyChannelsAdapter)
    y = adapter(x)
    # Should have resized spatial dims and repeated channels
    assert y.shape == (batch, out_ch, *out_size)
    # All output channels should be equal to the first channel
    for c in range(1, out_ch):
        assert torch.allclose(y[:, c, :, :], y[:, 0, :, :], atol=1e-6)

def test_image_mask_filter_adapter_quantile_only():
    # Create input with known values 0..15 reshape to 4x4
    x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    cfg = ImageMaskFilterAdapterConfig(
        output_size=(4, 4),
        channel_sizes=(1, 2),  # image + mask only
        interp_method='bilinear',
        binary_mask_thr_method='quantile',
        binary_mask_thrOrQuantile=0.5,
        filter_feature_methods=None
    )
    adapter = ImageMaskFilterAdapter(cfg)
    y = adapter(x)
    assert y.shape == (1, 2, 4, 4)
    # First channel is the image itself (resize is no-op here)
    assert torch.equal(y[:, 0, :, :], x[:, 0, :, :])
    # Second channel is the quantile mask at 0.5 -> values >=8 become 1 else 0
    expected_mask = (x[:, 0, :, :] >= 8).to(torch.float32)
    assert torch.equal(y[:, 1, :, :], expected_mask)

def test_image_mask_filter_adapter_sobel_filter_and_mask():
    
    # Constant input so sobel output = 0 everywhere, mask = 1 everywhere
    x = torch.ones(1, 1, 5, 5, dtype=torch.float32)

    cfg = ImageMaskFilterAdapterConfig(
        output_size=(5, 5),
        channel_sizes=(1, 3),  # image + mask + one filter
        interp_method='bilinear',
        binary_mask_thr_method='quantile',
        binary_mask_thrOrQuantile=0.5,
        filter_feature_methods=('sobel',)
    )

    adapter = ImageMaskFilterAdapter(cfg)
    
    y = adapter(x)
    
    assert y.shape == (1, 3, 5, 5)
    # Channel 0 = input image
    assert torch.allclose(y[:, 0], x[:, 0])
    # Channel 1 = mask of ones
    assert torch.all(y[:, 1] == 1.0)
    # Channel 2 = sobel on constant -> zeros
    assert torch.allclose(y[:, 2], torch.zeros_like(y[:, 2]), atol=1e-5)

def test_image_mask_filter_adapter_invalid_channel_mismatch():

    # Filter_feature_methods adds 1 filter but out_ch = 2 only => mismatch
    with pytest.raises(ValueError):
        ImageMaskFilterAdapterConfig(
            output_size=(2, 2),
            channel_sizes=(1, 2),
            interp_method='bilinear',
            binary_mask_thr_method='quantile',
            binary_mask_thrOrQuantile=0.5,
            filter_feature_methods=('sobel',)
        )
        # even if config passed, the adapter __init__ would error
        ImageMaskFilterAdapter(
            ImageMaskFilterAdapterConfig(
                output_size=(2, 2),
                channel_sizes=(1, 2),
                interp_method='bilinear',
                binary_mask_thr_method='quantile',
                binary_mask_thrOrQuantile=0.5,
                filter_feature_methods=('sobel',)
            )
        )

def test_image_mask_filter_adapter_invalid_binary_method_in_config():
    # The dataclass __post_init__ should catch invalid methods
    with pytest.raises(ValueError):
        ImageMaskFilterAdapterConfig(
            output_size=(4, 4),
            channel_sizes=(1, 2),
            interp_method='bilinear',
            binary_mask_thr_method='invalid_method',
            binary_mask_thrOrQuantile=0.5,
            filter_feature_methods=None
        )

def test_image_mask_filter_adapter_otsu_not_implemented():
    cfg = ImageMaskFilterAdapterConfig(
        output_size=(4, 4),
        channel_sizes=(1, 2),
        interp_method='bilinear',
        binary_mask_thr_method='otsu',
        binary_mask_thrOrQuantile=0.5,
        filter_feature_methods=None
    )

    with pytest.raises(NotImplementedError):
        ImageMaskFilterAdapter(cfg)

def test_input_adapter_factory_unknown_config():
    class DummyConfig: pass
    with pytest.raises(ValueError):
        InputAdapterFactory(DummyConfig())

def test_scaler_adapter_vector_scale_and_bias_numpy():
    # using numpy arrays
    scale = np.array([1.0, 0.0])
    bias = np.array([-1.0, 2.0])
    adapter = ScalerAdapter(scale_coefficient=scale, bias_coefficient=bias)
    x = torch.tensor([[5.0, -5.0]])
    out = adapter(x)
    expected = torch.tensor([[5.0*1.0 - 1.0, -5.0*0.0 + 2.0]])
    torch.testing.assert_close(out, expected)


def test_scaler_adapter_buffers_registered():
    adapter = ScalerAdapter(scale_coefficient=[1.0], bias_coefficient=[2.0])
    sd = adapter.state_dict()
    # scale and bias should be in state_dict (as buffers)
    assert 'scale' in sd
    assert 'bias' in sd


@pytest.mark.parametrize("bad_scale", [
    "not a list/array/tensor",
    [[1.0, 2.0], [3.0, 4.0]],           # 2-D list
    np.array([[1.0, 2.0]]),           # 2-D numpy
    torch.randn(2, 2),                # 2-D tensor
])
def test_scaler_adapter_bad_scale_type_or_dim_raises(bad_scale):
    with pytest.raises((TypeError, ValueError)):
        ScalerAdapter(scale_coefficient=bad_scale)


@pytest.mark.parametrize("bad_bias", [
    "bad",
    [[1.0, 2.0], [3.0, 4.0]],
    np.array([[1.0, 2.0]]),
    torch.randn(2, 2),
])
def test_bad_bias_type_or_dim_raises(bad_bias):
    with pytest.raises((TypeError, ValueError)):
        ScalerAdapter(scale_coefficient=1.0, bias_coefficient=bad_bias)


# Manual run of tests
if __name__ == "__main__":
    test_conv2d_adapter_forward_and_factory_dispatch()
    test_resize_copy_channels_adapter_forward_and_factory_dispatch()
    test_image_mask_filter_adapter_quantile_only()
    test_image_mask_filter_adapter_sobel_filter_and_mask()
    test_image_mask_filter_adapter_invalid_channel_mismatch()
    test_image_mask_filter_adapter_invalid_binary_method_in_config()
    test_image_mask_filter_adapter_otsu_not_implemented()
    test_input_adapter_factory_unknown_config()
    test_scaler_adapter_vector_scale_and_bias_numpy()
    test_scaler_adapter_buffers_registered()
    #test_scaler_adapter_bad_scale_type_or_dim_raises()


