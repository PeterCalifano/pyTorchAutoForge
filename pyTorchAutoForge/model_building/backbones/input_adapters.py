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
from pyTorchAutoForge.datasets.DatasetClasses import NormalizationType


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
    channel_sizes: tuple = (1, 3)   # [in_channels, out_channels]
    interp_method: Literal['linear', 'bilinear',
                           'bicubic', 'trilinear'] = 'bicubic'


@dataclass
class ImageMaskFilterAdapterConfig(BaseAdapterConfig):
    output_size: tuple    # [H, W]
    channel_sizes: tuple = (1, 3)
    interp_method: Literal['linear', 'bilinear',
                           'bicubic', 'trilinear'] = 'bicubic'
    binary_mask_thr_method: Literal['quantile', 'absolute',
                                    'otsu'] | None = 'quantile'  # For output channel 2
    binary_mask_thrOrQuantile: float = 0.9
    filter_feature_methods: tuple[Literal['sobel', 'local_variance', 'laplacian']] | None = (
        'sobel', )  # For output channels from 3 to N

    def __post_init__(self):
        # Validate quantile value
        if self.binary_mask_thr_method == 'quantile':
            if self.binary_mask_thrOrQuantile < 0 or self.binary_mask_thrOrQuantile > 1:
                raise ValueError(
                    f"Invalid quantile value: {self.binary_mask_thrOrQuantile}. Must be in [0, 1].")

        # Check number of channels against mask and methods
        if self.filter_feature_methods is not None:
            if len(self.filter_feature_methods) > 0 and self.channel_sizes[1] < 2 + len(self.filter_feature_methods):
                print(
                    f"\033[93mWarning: Multiple filter feature methods specified, but output channels ({self.channel_sizes[1]}) <= 2 + {len(self.filter_feature_methods)}. No filter will be applied.\033[0m")
                # Resize tuple
                # Keep only the valid methods
                self.filter_feature_methods = self.filter_feature_methods[:self.channel_sizes[1] - 1]

            # Validate filter feature methods
            for im, method in enumerate(self.filter_feature_methods):

                # self.filter_feature_methods[im] = method.lower() # TODO modify to list?
                if method.lower() not in ['sobel', 'local_variance', 'laplacian']:
                    raise ValueError(
                        f"Invalid filter feature method: {method.lower()}. Must be 'sobel', 'local_variance', or 'laplacian'.")

        if self.binary_mask_thr_method is not None:
            # Validate binary mask threshold method
            if self.binary_mask_thr_method not in ['quantile', 'absolute', 'otsu']:
                raise ValueError(
                    f"Invalid binary mask threshold method: {self.binary_mask_thr_method}. Must be 'quantile', 'absolute', or 'otsu'.")

            if self.channel_sizes[1] < 2:
                print(
                    f"\033[93mWarning: Output channels ({self.channel_sizes[1]}) < 2. No masking will be applied.\033[0m")

                # Set methods to None
                self.binary_mask_thr_method = None
                self.filter_feature_methods = None


@dataclass
class ScalerAdapterConfig(BaseAdapterConfig):
    scale: float | list[float] | np.ndarray | torch.Tensor | None = None
    bias: float | list[float] | np.ndarray | torch.Tensor | None = None
    scaling_mode: Literal['auto_compute_stats',
                          'precomputed_stats', 'batchnorm'] = 'precomputed_stats'
    input_size: int | tuple[int, int] | None = None

    # Optional data matrix for auto-compute stats mode
    data_matrix: np.ndarray | torch.Tensor | None = None
    # Type of normalization to apply
    normalization_type: NormalizationType = NormalizationType.MINMAX

    def __post_init__(self):
        # Validate configuration parameters
        if self.scaling_mode not in ['auto_compute_stats', 'precomputed_stats', 'batchnorm']:
            raise ValueError(
                f"Invalid scaling mode: {self.scaling_mode}. Must be 'auto_compute_stats', 'precomputed_stats', or 'batchnorm'.")

        if self.scaling_mode == 'auto_compute_stats':
            if self.data_matrix is None:
                raise ValueError(
                    "`data_matrix` must be provided in `auto_compute_stats` mode. It is used to compute the scale and bias coefficients.")
            if self.scale is not None or self.bias is not None:
                print(
                    '\033[93mWarning: `scale` and `bias` will be ignored in `auto_compute_stats` mode.\033[0m')

        if self.scale is None and self.scaling_mode == 'precomputed_stats':
            raise ValueError("`scale` must be provided and cannot be None.")

        if self.scaling_mode == 'batchnorm':
            if self.input_size is None:
                raise ValueError(
                    "`input_size` must be provided for batch normalization scaling mode.")

        if isinstance(self.scale, (list, np.ndarray)) and len(self.scale) == 0:
            raise ValueError(
                "`scale` must be a non-empty list or numpy array.")

        if self.bias is not None and isinstance(self.bias, (list, np.ndarray)) and len(self.bias) == 0:
            raise ValueError("`bias` must be a non-empty list or numpy array.")

        if self.normalization_type is None:
            raise ValueError("`normalization_type` must be provided.")

# ===== Adapter modules =====


class ScalerAdapter(BaseAdapter):
    def __init__(self,
                 scale_coefficient: list[float] | np.ndarray | torch.Tensor | None = None,
                 bias_coefficient: list[float] | np.ndarray | torch.Tensor | None = None,
                 input_size: int | None = None,
                 adapter_cfg: ScalerAdapterConfig | None = None,
                 data_matrix: np.ndarray | torch.Tensor | None = None) -> None:
        """
        Initializes the ScalerAdapter for input normalization or feature scaling.

        This adapter rescales input tensors by fixed scale and bias vectors or scalars.
        It supports normalization using precomputed statistics, batch normalization,
        or automatic computation of statistics from a data matrix.

        Args:
            scale_coefficient (list[float] | np.ndarray | torch.Tensor | None): 
                Multiplicative scaling factor. Can be a scalar or 1D vector.
                Required when using 'precomputed_stats' scaling mode.
            bias_coefficient (list[float] | np.ndarray | torch.Tensor | None, optional): 
                Additive bias term. Can be a scalar or 1D vector. 
                Defaults to zeros matching scale shape when None.
            input_size (int | None, optional):
                Size of input features. Required when using 'batchnorm' scaling mode.
                Defaults to None.
            adapter_cfg (ScalerAdapterConfig | None, optional):
                Configuration object for the adapter. When provided, overrides
                individual parameter settings. Defaults to None.
            data_matrix (np.ndarray | torch.Tensor | None, optional):
                Data matrix used for automatic computation of statistics in 'auto_compute_stats' mode.

        Raises:
            TypeError: If scale_coefficient or bias_coefficient has invalid type.
            ValueError: If scale and bias dimensions don't match or if required
                parameters are missing for the selected scaling mode.
        """

        super().__init__()

        if adapter_cfg is None:
            self.scaling_mode = 'precomputed_stats'
            print("\033[93mWarning: No adapter_cfg provided. Using default 'precomputed_stats' scaling mode. It is highly recommended to provide a configuration object.\033[0m")
        else:
            self.scaling_mode = adapter_cfg.scaling_mode

        match self.scaling_mode:
            case 'precomputed_stats':

                # Check input validity
                if scale_coefficient is None:
                    raise ValueError(
                        "`scale_coefficient` must be provided and cannot be None.")

                # Handle scale input
                if isinstance(scale_coefficient, list) or isinstance(scale_coefficient, np.ndarray):
                    scale = torch.as_tensor(
                        scale_coefficient, dtype=torch.float32)

                elif torch.is_tensor(scale_coefficient):
                    scale = scale_coefficient.to(dtype=torch.float32)

                elif isinstance(scale_coefficient, (int, float)):
                    # Convert scalar to 0-D tensor
                    scale = torch.tensor(scale_coefficient, dtype=torch.float32)
                else:
                    raise TypeError(
                        "`scale_coefficient` must be a list, a 1-D numpy array, or a torch Tensor")

                if scale.ndim > 1:
                    raise ValueError(
                        "`scale_coefficient` must be 1-D (or 0-D)")

                # Handle bias input
                if bias_coefficient is None:
                    bias = torch.zeros_like(scale)

                elif isinstance(bias_coefficient, list) or isinstance(bias_coefficient, np.ndarray):
                    bias = torch.as_tensor(bias_coefficient, dtype=torch.float32)

                elif torch.is_tensor(bias_coefficient):
                    bias = bias_coefficient.to(dtype=torch.float32)
                
                elif isinstance(bias_coefficient, (int, float)):
                    # Convert scalar to 0-D tensor
                    bias = torch.tensor(bias_coefficient, dtype=torch.float32)

                else:
                    raise TypeError(
                        "`bias_coefficient` must be None, a list, a 1-D numpy array, or a torch Tensor"
                    )
                
                if bias.ndim > 1:
                    raise ValueError("`bias_coefficient` must be 1-D (or 0-D)")

                # Check consistency of dimensions
                if scale.ndim == 1 and bias.ndim == 1 and scale.shape != bias.shape:
                    raise ValueError(
                        "`scale_coefficient` and `bias_coefficient` must have the same length")

                # Register scale and bias as buffers
                self.register_buffer('scale', scale)
                self.register_buffer('bias',  bias)

                # Define forward method
                self._forward_impl = self._forward_precomputed_stats

            case "batchnorm":
                if self.adapter_cfg is not None:
                    input_size_from_cfg = self.adapter_cfg.input_size

                assert input_size is not None or input_size_from_cfg is not None, \
                    "For batch normalization scaling mode, `input_size` must be provided at init or in cfg object."

                input_size = input_size if input_size is not None else input_size_from_cfg
                self.batchnorm_layer = nn.BatchNorm1d(num_features=input_size,
                                                      eps=1e-5,
                                                      momentum=0.1,
                                                      affine=True,
                                                      track_running_stats=True
                                                      )

                self._forward_impl = self.batchnorm_layer

            case "auto_compute_stats":  
                from pyTorchAutoForge.utils import torch_to_numpy, numpy_to_torch

                assert adapter_cfg is not None, \
                    "For auto-compute stats scaling mode, `adapter_cfg` must be provided."
                
                if data_matrix is not None:
                    self.data_matrix = torch_to_numpy(data_matrix)
                else:
                    if adapter_cfg.data_matrix is None:
                        raise ValueError(
                            "`data_matrix` must be provided in `auto_compute_stats` mode. It is used to compute the scale and bias coefficients.")
                    self.data_matrix = adapter_cfg.data_matrix

                assert self.data_matrix is not None, \
                    "For auto-compute stats scaling mode, `data_matrix` must be provided."

                # Check first dimension of data matrix is batch
                if self.data_matrix.ndim < 2:
                    raise ValueError(
                        "`data_matrix` must have at least 2 dimensions (batch size, features).")
                elif self.data_matrix.ndim > 2:
                    raise ValueError(
                        "`data_matrix` must be 2D (batch size, features).")
                
                if self.data_matrix.shape[0] < self.data_matrix.shape[1]:
                    print("\033[93mWarning: `data_matrix` seems to have size 1 larger than 0. Automatically transposed assuming (features, batch size).\033[0m")
                    self.data_matrix = self.data_matrix.T

                # Compute scale and bias from data matrix
                if self.adapter_cfg.normalization_type == NormalizationType.MINMAX:

                    # Define scaler 
                    from sklearn.preprocessing import MinMaxScaler
                    input_scaler = MinMaxScaler(0, 1)
                    input_scaler.fit(self.data_matrix)

                    scale = input_scaler.scale_
                    bias = input_scaler.min_

                elif self.adapter_cfg.normalization_type == NormalizationType.ZSCORE:

                    # Define scaler
                    from sklearn.preprocessing import StandardScaler
                    input_scaler = StandardScaler()
                    input_scaler.fit(self.data_matrix)

                    scale = input_scaler.var_
                    assert np.all(scale > 0), "Scale in `StandardScaler` must be positive (formally a variance)!."
                    bias = - (input_scaler.mean_ / scale)

                else:
                    raise TypeError(
                        "Adapter cfg `normalization_type` must be a valid class as enumerated by NormalizationType.")
                
                # Register scale and bias as buffers
                self.register_buffer('scale', numpy_to_torch(scale, dtype=torch.float32))
                self.register_buffer('bias', numpy_to_torch(bias, dtype=torch.float32))

                # Define forward method
                self._forward_impl = self._forward_precomputed_stats

    def _forward_precomputed_stats(self, x: torch.Tensor) -> torch.Tensor:

        if self.scale.ndim == 1:
            if x.dim() != 2 or x.shape[1] != self.scale.numel():
                raise ValueError(
                    f"Current implementation supports inputs of shape (B, N) where N must be = {self.scale.shape[0]}, but got {x.shape}")

            # Reshape scale and bias for broadcasting over batch
            scale_broadcast = self.scale.unsqueeze(0)  # shape (1, N)
            bias_broadcast = self.bias.unsqueeze(0)    # shape (1, N)

        else:
            # Scalar (0-D tensor) broadcasts automatically
            scale_broadcast = self.scale
            bias_broadcast = self.bias

        return x * scale_broadcast + bias_broadcast

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)


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
            raise ValueError(
                f"Number of input channels {self.in_ch} does not match number of input feature maps (image, binary mask, filters) {input_feature_maps}.")

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
        output_tensor = torch.zeros(
            (x.size(0), self.out_ch, x.size(2), x.size(3)), device=x.device)

        # Allocate first channel: image
        output_tensor[:, 0, :, :] = x[:, 0, :, :]

        # Compute second channel: binary mask
        if self.binary_mask_operator is not None:
            # Apply binary mask operator
            output_tensor[:, 1, :, :] = self.binary_mask_operator(
                x[:, 0, :, :])

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
    return ScalerAdapter(cfg.scale,
                         cfg.bias,
                         input_size=cfg.input_size,
                         adapter_cfg=cfg)
