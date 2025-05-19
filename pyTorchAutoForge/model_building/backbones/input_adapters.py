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
from pyTorchAutoForge.model_building.backbones.image_processing_operators import SobelGradient, QuantileThresholdMask, LocalVarianceMap, LaplacianOfGaussian

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
            for im, method in enumerate(self.filter_feature_methods):

                #self.filter_feature_methods[im] = method.lower() # TODO modify to list?
                if method.lower() not in ['sobel', 'local_variance', 'laplacian']:
                    raise ValueError(
                        f"Invalid filter feature method: {method.lower()}. Must be 'sobel', 'local_variance', or 'laplacian'.")

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
    

@dataclass
class ScalerAdapterConfig(BaseAdapterConfig):
    scale: float | list[float] | np.ndarray | torch.Tensor
    bias: float | list[float] | np.ndarray | torch.Tensor | None = None

    def __post_init__(self):
        # Validate scale and bias
        if isinstance(self.scale, (list, np.ndarray)) and len(self.scale) == 0:
            raise ValueError("`scale` must be a non-empty list or numpy array.")
        if self.bias is not None and isinstance(self.bias, (list, np.ndarray)) and len(self.bias) == 0:
            raise ValueError("`bias` must be a non-empty list or numpy array.")

# ===== Adapter modules =====
class ScalerAdapter(BaseAdapter):
    """
    ScalerAdapter rescales input tensors by a fixed scale and bias vectors or scalars. Useful for normalizing or shifting input data before feeding to a model.

    Args:
        scale (float): Multiplicative scaling factor. Can be a 0D (for all archs) or 1D vector (for DNNs only).
        bias (float, optional): Additive bias. Defaults to 0.0. Can be a 0D (for all archs) or 1D vector (for DNNs only).
    """
    def __init__(
        self,
        scale_coefficient: list[float] | np.ndarray | torch.Tensor,
        bias_coefficient: list[float] | np.ndarray | torch.Tensor | None = None
    ) -> None:
        
        super().__init__()
        # Handle scale input
        if isinstance(scale_coefficient, list) or isinstance(scale_coefficient, np.ndarray):
            scale = torch.as_tensor(scale_coefficient, dtype=torch.float32)

        elif torch.is_tensor(scale_coefficient):
            scale = scale_coefficient.to(dtype=torch.float32)

        else:
            raise TypeError(
                "`scale_coefficient` must be a list, a 1-D numpy array, or a torch Tensor")
        
        if scale.ndim > 1:
            raise ValueError("`scale_coefficient` must be 1-D (or 0-D)")

        # Handle bias input
        if bias_coefficient is None:
            bias = torch.zeros_like(scale)

        elif isinstance(bias_coefficient, list) or isinstance(bias_coefficient, np.ndarray):
            bias = torch.as_tensor(bias_coefficient, dtype=torch.float32)
        elif torch.is_tensor(bias_coefficient):
            bias = bias_coefficient.to(dtype=torch.float32)
        else:
            raise TypeError(
                "`bias_coefficient` must be None, a list, a 1-D numpy array, or a torch Tensor"
            )
        if bias.ndim > 1:
            raise ValueError("`bias_coefficient` must be 1-D (or 0-D)")

        # Check consistency of dimensions
        if scale.ndim == 1 and bias.ndim == 1 and scale.shape != bias.shape:
            raise ValueError("`scale_coefficient` and `bias_coefficient` must have the same length")

        # Register scale and bias as buffers
        self.register_buffer('scale', scale)
        self.register_buffer('bias',  bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.scale.ndim == 1:
            if x.dim() != 2 or x.shape[1] != self.scale.numel():
                raise ValueError(f"Current implementation supports inputs of shape (B, N) where N must be = {self.scale.shape[0]}, but got {x.shape}")
            
            # Reshape scale and bias for broadcasting over batch
            scale_broadcast = self.scale.unsqueeze(0)  # shape (1, N)
            bias_broadcast = self.bias.unsqueeze(0)    # shape (1, N)

        else:
            # Scalar (0-D tensor) broadcasts automatically
            scale_broadcast = self.scale
            bias_broadcast = self.bias

        return x * scale_broadcast + bias_broadcast

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

        self.binary_mask_operator = None
        self.filter_operator = None

        # Build modules
        if self.binary_mask_thr_method is not None:
            if self.binary_mask_thr_method == 'quantile':
                self.binary_mask_operator = QuantileThresholdMask(
                    quantile=self.binary_mask_thrOrQuantile)
                
            elif self.binary_mask_thr_method == 'absolute':
                self.binary_mask = QuantileThresholdMask(
                    abs_thr=self.binary_mask_thrOrQuantile)
            
            elif self.binary_mask_thr_method == 'otsu':                
                raise NotImplementedError('Otsu method not implemented yet.')
            else:
                raise ValueError(
                    f"Invalid binary mask threshold method: {self.binary_mask_thr_method}. Must be 'quantile', 'absolute', or 'otsu'.")
        
        if self.filter_feature_methods is not None:
            self.filter_operator = nn.ModuleList()

            for method in self.filter_feature_methods:
                if method == 'local_variance':
                    self.filter_operator.append(LocalVarianceMap())
                elif method == 'sobel':
                    self.filter_operator.append(SobelGradient())
                elif method == 'laplacian':
                    self.filter_operator.append(LaplacianOfGaussian())
                else:
                    raise ValueError(
                        f"Invalid filter feature method: {method}. Must be 'sobel', 'local_variance', or 'laplacian'.")

        # Check number of channels against mask and methods (throw error if not matched)
        input_feature_maps = 1
        if self.binary_mask_operator is not None:
            input_feature_maps += 1

        if self.filter_operator is not None:
            input_feature_maps += len(self.filter_operator)

        if self.out_ch != input_feature_maps:
            raise ValueError(f"Number of input channels {self.in_ch} does not match number of input feature maps (image, binary mask, filters) {input_feature_maps}.")

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
        if self.binary_mask_operator is not None:
            # Apply binary mask operator
            output_tensor[:, 1, :, :] = self.binary_mask_operator(x[:, 0, :, :])
        
        # Compute 3 to N channels: filter features
        if self.filter_operator is not None:
            for i, filter_op in enumerate(self.filter_operator):
                # Apply filter operator
                output_tensor[:, 2 + i, :, :] = filter_op(x[:, 0, :, :])

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

# === Register adapters ===
@InputAdapterFactory.register
def _(cfg: Conv2dAdapterConfig) -> Conv2dResolutionChannelsAdapter:
    return Conv2dResolutionChannelsAdapter(cfg)

@InputAdapterFactory.register
def _(cfg: ResizeAdapterConfig) -> ResizeCopyChannelsAdapter:
    return ResizeCopyChannelsAdapter(cfg)

@InputAdapterFactory.register
def _(cfg: ImageMaskFilterAdapterConfig) -> ImageMaskFilterAdapter:
    return ImageMaskFilterAdapter(cfg)

@InputAdapterFactory.register
def _(cfg: ScalerAdapterConfig) -> ScalerAdapter:
    return ScalerAdapter(cfg.scale, cfg.bias)