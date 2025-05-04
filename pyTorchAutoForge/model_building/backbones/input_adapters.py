# Module to apply activation functions in forward pass instead of defining them in the model class
import torch.nn as nn
from typing import Literal
from pyTorchAutoForge.setup import BaseConfigClass

from torch import nn
from abc import ABC
from dataclasses import dataclass, field
import kornia
import numpy as np
import torch
from functools import singledispatch


@dataclass
class BaseAdapterConfig(BaseConfigClass):
    """Marker base class for adapter configs"""
    pass


class BaseAdapter(nn.Module, ABC):
    """Common interface for all adapters"""

    def __init__(self):
        super().__init__()

    def forward(self, x):  # type: ignore
        raise NotImplementedError("Adapter must implement forward method")

@dataclass
class Conv2dAdapterConfig(BaseAdapterConfig):
    output_size: tuple      # [H, W]
    channel_sizes: tuple    # [in_channels, out_channels]


@dataclass
class ResizeAdapterConfig(BaseAdapterConfig):
    output_size: tuple      # [H, W]
    channel_sizes: tuple  = (1, 3)   # [in_channels, out_channels]
    interp_method: Literal['linear', 'bilinear',
                           'bicubic', 'trilinear'] = 'bicubic'

@dataclass
class ImageMaskFilterAdapterConfig(BaseAdapterConfig):
    output_size: tuple    # [H, W]
    channel_sizes: tuple  = (1, 3)
    interp_method: Literal['linear', 'bilinear', 
                           'bicubic', 'trilinear'] = 'bicubic' 
    binary_mask_thr_method: Literal['quantile', 'absolute', 'otsu'] | None = 'quantile' # For output channel 2
    binary_mask_thrOrQuantile: float = 0.9
    filter_feature_methods: tuple[Literal['sobel', 'local_variance', 'laplacian']] | None = ('sobel', )  # For output channels from 3 to N

    def __post_init__(self):
        # Validate quantile value
        if self.binary_mask_thr_method == 'quantile':
            if self.binary_mask_thrOrQuantile < 0 or self.binary_mask_thrOrQuantile > 1:
                raise ValueError(f"Invalid quantile value: {self.binary_mask_thrOrQuantile}. Must be in [0, 1].")
    
        # Check number of channels against mask and methods
        if self.filter_feature_methods is not None:
            if len(self.filter_feature_methods) > 0 and self.channel_sizes[1] < 2 + len(self.filter_feature_methods):
                print(
                    f"\033[93mWarning: Multiple filter feature methods specified, but output channels ({self.channel_sizes[1]}) <= 2 + {len(self.filter_feature_methods)}. No filter will be applied.\033[0m")
                # Resize tuple
                self.filter_feature_methods = self.filter_feature_methods[:self.channel_sizes[1] - 1] # Keep only the valid methods

            # Validate filter feature methods
            for method in self.filter_feature_methods:
                if method not in ['sobel', 'local_variance', 'laplacian']:
                    raise ValueError(
                        f"Invalid filter feature method: {method}. Must be 'sobel', 'local_variance', or 'laplacian'.")

        if self.binary_mask_thr_method is not None:
            # Validate binary mask threshold method
            if self.binary_mask_thr_method not in ['quantile', 'absolute', 'otsu']:
                raise ValueError(
                    f"Invalid binary mask threshold method: {self.binary_mask_thr_method}. Must be 'quantile', 'absolute', or 'otsu'.")

            if self.channel_sizes[1] < 2:
                print( f"\033[93mWarning: Output channels ({self.channel_sizes[1]}) < 2. No masking will be applied.\033[0m")

                # Set methods to None
                self.binary_mask_thr_method = None
                self.filter_feature_methods = None        
    


# ===== Adapter modules =====
class Conv2dResolutionChannelsAdapter(BaseAdapter):
    """
    Channels & resolution adapter using 1x1 convolution and pooling.
    Steps:
      1. Expand or reduce channels via 1x1 conv (stride=2 downsamples spatially).
      2. Adapt spatial size exactly via AdaptiveAvgPool2d.
    """

    def __init__(self, cfg: Conv2dAdapterConfig):
        super().__init__()

        # Unpack desired channel sizes
        in_ch, out_ch = cfg.channel_sizes

        # 1x1 conv to change channel count and downsample by 2
        self.channel_expander = nn.Conv2d(
            in_ch, out_ch, kernel_size=1, stride=2, bias=False)
        
        # Unpack spatial target dimensions
        H, W = cfg.output_size

        # Adaptive pool to force exact [h, w]
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size=(H, W))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply 1x1 conv then adaptive pooling.
        Args:
          x: input tensor of shape [B, in_ch, H_in, W_in]
        Returns:
          Tensor of shape [B, out_ch, target_h, target_w]
        """
        # Cast to torch.float32 if needed
        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        x = self.channel_expander(x)
        return self.adaptive_pool(x)

class ResizeCopyChannelsAdapter(BaseAdapter):
    """
    Adapter that resizes and duplicates channels.
    Steps:
      1. Resize spatial dims with Kornia (bilinear by default).
      2. Repeat channels if output_channels > input_channels.
    """

    def __init__(self, cfg: ResizeAdapterConfig):
        super().__init__()

        # Convert output_size list to tuple (required by Kornia)
        self.output_size = tuple(cfg.output_size)

        # Unpack input/output channels from config # e.g. [1,3] to repeat single channel
        self.in_ch, self.out_ch = cfg.channel_sizes

        # Save interpolation method for resizing
        self.interp = cfg.interp_method

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Resize and channel-repeat input tensor.
        Args:
          x: input tensor [B, in_ch, H_in, W_in]
        Returns:
          Tensor [B, out_ch, target_h, target_w]
        """
        
        # Cast to torch.float32 if needed
        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        # Spatial resize to desired output_size through kornia 2D interpolation function
        x = kornia.geometry.transform.resize(
            x, self.output_size, interpolation=self.interp)
        
        # If more output channels are needed, repeat input along channel dim
        if self.out_ch > self.in_ch:

            # Determine how many times to repeat the channels
            repeat_factor = self.out_ch // self.in_ch

            # Repeat tensor along channel dimension
            x = x.repeat(1, repeat_factor, 1, 1)

        # Return adapted tensor ready for backbone
        return x


class ImageMaskFilterAdapter(BaseAdapter):

    def __init__(self, cfg: ImageMaskFilterAdapterConfig):
        super().__init__()

        # Convert output_size list to tuple (required by Kornia)
        self.output_size = tuple(cfg.output_size)

        # Unpack input/output channels from config # e.g. [1,3] to repeat single channel
        self.in_ch, self.out_ch = cfg.channel_sizes

        # Save interpolation method for resizing
        self.interp = cfg.interp_method

        # Save binary mask threshold method
        self.binary_mask_thr_method = cfg.binary_mask_thr_method
        self.binary_mask_thrOrQuantile = cfg.binary_mask_thrOrQuantile
        self.filter_feature_methods = cfg.filter_feature_methods


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Resize, apply binary mask and filters to produce feature maps.
        Args:
          x: input tensor [B, in_ch, H_in, W_in]
        Returns:
          Tensor [B, out_ch, target_h, target_w]
        """
        
        # Cast to torch.float32 if needed
        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        # Spatial resize to desired output_size through kornia 2D interpolation function
        x = kornia.geometry.transform.resize(
            x, self.output_size, interpolation=self.interp)

        # Define output tensor 
        output_tensor = torch.zeros((x.size(0), self.out_ch, x.size(2), x.size(3)), device=x.device)    

        # Allocate first channel: image
        output_tensor[:, 0, :, :] = x[:, 0, :, :]

        # Compute second channel: binary mask
        # TODO

        # Compute 3 to N channels: filter features
        # TODO

        # Return adapted tensor ready for backbone
        return output_tensor

# === Factory for adapters ===
@singledispatch
def InputAdapterFactory(cfg) -> nn.Module:
    """
    Factory for adapter modules based on config type.
    Register new adapters with @InputAdapterFactory.register.
    """
    raise ValueError(
        f"No adapter registered for config type {type(cfg).__name__}")


@InputAdapterFactory.register
def _(cfg: Conv2dAdapterConfig) -> Conv2dResolutionChannelsAdapter:
    return Conv2dResolutionChannelsAdapter(cfg)


@InputAdapterFactory.register
def _(cfg: ResizeAdapterConfig) -> ResizeCopyChannelsAdapter:
    return ResizeCopyChannelsAdapter(cfg)
