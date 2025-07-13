from abc import ABC
from torchvision import models
import numpy as np
from dataclasses import dataclass, field
import kornia
import os
import optuna
import torch
from torch.nn import functional as torchFunc
from torch import nn
from pyTorchAutoForge.model_building.ModelAutoBuilder import AutoComputeConvBlocksOutput, ComputeConv2dOutputSize, ComputePooling2dOutputSize, ComputeConvBlock2dOutputSize, EnumMultiHeadOutMode, MultiHeadRegressor
from typing import Literal

from pyTorchAutoForge.utils import GetDeviceMulti
from pyTorchAutoForge.setup import BaseConfigClass
from pyTorchAutoForge.model_building.ModelMutator import ModelMutator
from pyTorchAutoForge.model_building.convolutionalBlocks import ConvolutionalBlock1d, ConvolutionalBlock2d, ConvolutionalBlock3d, FeatureMapFuser, FeatureMapFuserConfig, _feature_map_fuser_factory, fuser_type

from pyTorchAutoForge.model_building.fullyConnectedBlocks import FullyConnectedBlock
from pyTorchAutoForge.model_building.factories.block_factories import (pooling_types,
                                                                       pooling_types_1d,
                                                                       pooling_types_2d,
                                                                       pooling_types_3d,
                                                                       activ_types,
                                                                       regularizer_types,
                                                                       init_methods)


# DEVNOTE TODO change name of this file to "modelBuildingBlocks.py" and move the OLD classes to the file "modelClasses.py" for compatibility with legacy codebase

#############################################################################################################################################

# TODO complete implementation of AutoForgeModule!
class AutoForgeModule(torch.nn.Module):
    """
    AutoForgeModule Custom base class inheriting nn.Module to define a PyTorch NN model, augmented with saving/loading routines like Pytorch Lightning.

    _extended_summary_

    :param torch: _description_
    :type torch: _type_
    :raises Warning: _description_
    """

    def __init__(self, moduleName: str | None = None, enable_tracing: bool = False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Assign module name. If not provided by user, use class name
        if moduleName is None:
            self.moduleName = self.__class__.__name__
        else:
            self.moduleName = moduleName

    def save(self, 
             example_input=None, 
             target_device: torch.device | str | None = None) -> None:

        if self.enable_tracing == True and example_input is None:
            self.enable_tracing = False
            raise Warning(
                'You must provide an example input to trace the model through torch.jit.trace(). Overriding enable_tracing to False.')

        if target_device is None:
            target_device = self.device if self.device is not None else GetDeviceMulti()

    def load(self):
        pass

# %% Configuration dataclasses

@dataclass
class TemplateNetBaseConfig(BaseConfigClass):
    # General
    model_name: str = "template_network"

    # Architecture design
    regularization_layer_type: regularizer_types = 'none'  # Regularization layer type, e.g. dropout, batchnorm, etc.

    out_channels_sizes: list[int] | None = None

    # Weights initialization method
    init_method_type: init_methods = "xavier_normal"

    # Additional features
    dropout_ensemble_size: int = 1  # Set >1 if building using ensemble wrapper


@dataclass
class TemplateConvNetConfig(TemplateNetBaseConfig):

    # Generic convolutional blocks parameters
    # Pooling parameters
    pool_type: pooling_types = "MaxPool2d"

    activ_type: activ_types = "prelu"

    regularizer_type: regularizer_types = "none"
    regularizer_param: int | float = 0.0

    conv_stride: int | tuple[int, int, int] = 1
    conv_padding: int | tuple[int, int, int] = 0
    conv_dilation: int | tuple[int, int, int] = 1
    prelu_params: Literal["all", "unique"] = "unique"

    # Nominal size of input tensor. Optional to verify design can work
    reference_input_size: tuple[int, ...] | None = None

    def __post_init__(self):
        # If activation type is relu one and init method not changed set it to kaiming
        if "elu" in self.activ_type and self.init_method_type == "xavier_normal":
            self.init_method_type = "kaiming_normal"


@dataclass
class TemplateConvNet2dConfig(TemplateConvNetConfig):

    save_intermediate_features: bool = False

    kernel_sizes: list[int] | None = None
    pool_kernel_sizes: list[int] | int | None = None
    num_input_channels: int = 3  # Default is 3

    # If specified, a convolution using out_channels as this number is added
    linear_input_layer_size: int | None = None # By default, input size is in_channels
    linear_output_layer_size: int | None = None  # By default, no linear output layer --> no linear regressor
    size_skip_to_linear_output: int | None = None

    def __post_init__(self):
        # Call super post_init
        super().__post_init__()

        # Check pooling type is correct (2d)
        if not self.pool_type.endswith("2d"):
            raise TypeError(
                f"TemplateConvNet2dConfig: pool_type must be of type 'MaxPool2d', 'AvgPool2d', 'Adapt_MaxPool2d' or 'Adapt_AvgPool2d'. Found {self.pool_type}.")

        # Check config validity, throw error is not
        if self.kernel_sizes is None:
            raise ValueError(
                "TemplateConvNet2dConfig: 'kernel_sizes' cannot be None")
        if self.pool_kernel_sizes is None:
            raise ValueError(
                "TemplateConvNet2dConfig: 'pool_kernel_sizes' cannot be None")
        if self.out_channels_sizes is None:
            raise ValueError(
                "TemplateConvNet2dConfig: 'out_channels_sizes' cannot be None")

        if len(self.kernel_sizes) != len(self.out_channels_sizes):
            raise ValueError(
                "TemplateConvNet2dConfig: 'kernel_sizes' and 'out_channels_sizes' must have the same length")

        if isinstance(self.pool_kernel_sizes, list):
            if len(self.kernel_sizes) != len(self.pool_kernel_sizes):
                raise ValueError(
                    "TemplateConvNet2dConfig: 'kernel_sizes' and 'pool_kernel_sizes' must have the same length. Alternatively, pool_kernel_sizes must be scalar integer.")

        # Check if linear input layer size is specified if size_skip_to_linear_output is and check size
        if self.size_skip_to_linear_output is not None:
            
            if self.linear_input_layer_size is None:
                raise ValueError(
                    f"TemplateConvNet2dConfig: 'linear_input_layer_size' must be specified when 'size_skip_to_linear_output' is set (got None, expected >= {self.size_skip_to_linear_output})."
                )
            
            elif self.linear_input_layer_size < self.size_skip_to_linear_output:
                raise ValueError(
                    f"TemplateConvNet2dConfig: 'linear_input_layer_size' ({self.linear_input_layer_size}) must be >= 'size_skip_to_linear_output' ({self.size_skip_to_linear_output})."
                )
        else:
            self.size_skip_to_linear_output = 0 # Override to zero for actual usage

        # Automagic configuration post-processing
        # If pooling kernel size is scalar, unroll to number of layers
        if isinstance(self.pool_kernel_sizes, int):
            self.pool_kernel_sizes = [self.pool_kernel_sizes] * len(self.kernel_sizes)

        assert (isinstance(self.conv_stride, int)
                ), "conv_stride must be a scalar integer for ConvolutionalBlock2d"

        # TODO add check on conv sizes if reference_input_size is passed, to ensure kernel and pool sizes are compatible
        # convBlockOutputSize = AutoComputeConvBlocksOutput( self, kernel_sizes, pool_kernel_sizes)

@dataclass
class TemplateConvNetFeatureFuser2dConfig(TemplateConvNet2dConfig):
    num_skip_channels: list[int] = field(default_factory=lambda: [])
    merge_module_index: list[int] = field(default_factory=lambda: [])
    merge_module_type: list[fuser_type] = field(
        default_factory=lambda: [])
    
    # Optional
    num_attention_heads:list[int] = field(default_factory=lambda: [1])

    def __post_init__(self):
        # Call super post init
        super().__post_init__()

        # Check fuser modules options are not none
        if len(self.num_skip_channels) == 0:
            raise ValueError("TemplateConvNetFeatureFuser2dConfig: 'num_skip_channels' cannot be empty")
        if len(self.merge_module_index) == 0:
            raise ValueError("TemplateConvNetFeatureFuser2dConfig: 'merge_module_index' cannot be empty")
        if len(self.merge_module_type) == 0:
            raise ValueError("TemplateConvNetFeatureFuser2dConfig: 'merge_module_type' cannot be empty")

        # If merge_module_type is a single value, expand it to the number of skip channels
        if len(self.merge_module_type) == 1:
            self.merge_module_type = [self.merge_module_type[0]] * len(self.num_skip_channels)

        # Expand num_attention_heads is scalar
        if len(self.num_attention_heads) == 1:
            self.num_attention_heads = [self.num_attention_heads[0]] * len(self.num_skip_channels)

        # Check they have equal length
        if len(self.num_skip_channels) != len(self.merge_module_index):
            raise ValueError(
                "TemplateConvNetFeatureFuser2dConfig: 'num_skip_channels' and 'merge_module_index' must have the same length")

        if len(self.num_skip_channels) != len(self.merge_module_type):
            raise ValueError(
                "TemplateConvNetFeatureFuser2dConfig: 'num_skip_channels' and 'merge_module_type' must have the same length")

@dataclass
class TemplateFullyConnectedNetConfig(TemplateNetBaseConfig):

    # Architecture definition
    input_layer_size: int | None = None
    output_layer_size: int | None = None
    regularizer_param: int | float = 0.0

    activ_type: activ_types = "prelu"

    prelu_params: Literal["all", "unique"] = "unique"
    input_skip_index: list[int] | None = None

    def __post_init__(self):
        if self.out_channels_sizes is None:
            raise ValueError(
                "TemplateFullyConnectedNetConfig: 'out_channels_sizes' cannot be None")
        if self.input_layer_size is None:
            raise ValueError(
                "TemplateFullyConnectedNetConfig: 'input_layer_size' cannot be None")
                
        elif self.input_layer_size <= 0:
            raise ValueError(
                "TemplateFullyConnectedNetConfig: 'input_layer_size' must be a positive integer")

        if self.output_layer_size is None:
            # Assume last layer is given by last entry of out_channels_sizes
            self.output_layer_size = self.out_channels_sizes[-1]
            # Print warning for this in orange
            print("\033[33m[Warning] TemplateFullyConnectedNetConfig: 'output_layer_size' is None. Setting to last value of 'out_channels_sizes':",
                  self.output_layer_size, "\033[0m")

        if self.dropout_ensemble_size > 1 and (self.regularizer_param == 0.0 or self.regularization_layer_type != 'dropout'):
            raise ValueError(
                "TemplateFullyConnectedNetConfig: 'use_dropout_ensembling' is True but either 'regularizer_param' is 0.0 or 'regularization_layer_type' is not set to 'dropout'. Please set 'regularization_layer_type' to 'dropout' and provide a non-zero value for 'regularizer_param'.")

        if self.input_skip_index is not None:
            if len(self.input_skip_index) > self.output_layer_size:
                raise ValueError("TemplateFullyConnectedNetConfig: 'input_skip_index' cannot be longer than 'output_layer_size'. Please check your configuration.")

        # If activation type is a relu one and init method not changed set it to kaiming
        if "elu" in self.activ_type and self.init_method_type == "xavier_normal":
            self.init_method_type = "kaiming_normal"

# %% Special wrapper classes
# Monte-Carlo Dropout generic DNN wrapper
class DropoutEnsemblingNetworkWrapper(AutoForgeModule):
    def __init__(self,
                 model: nn.Module,
                 ensemble_size: int = 20,
                 enable_ensembling: bool = True) -> None:

        super().__init__()
        self.base_model: nn.Module = model  # Store model
        self.ensemble_size: int = ensemble_size
        self.enable_ensembling_: bool = enable_ensembling

        if not isinstance(self.base_model, nn.Module):
            raise TypeError(
                "DropoutEnsemblingNetworkWrapper: base_model must be an instance of nn.Module")

        # Outputs cached after forward
        self.last_mean: torch.Tensor | None = None
        self.last_median: torch.Tensor | None = None
        self.last_variance: torch.Tensor | None = None

    def get_last_stats(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the last mean, median and variance computed during the forward pass.
        """
        if self.last_mean is None or self.last_median is None or self.last_variance is None:
            raise ValueError(
                "DropoutEnsemblingNetworkWrapper: No forward pass has been performed yet.")

        return self.last_mean, self.last_median, self.last_variance

    def _forward_single(self, x: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.base_model(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        if self.training:
            # During training, run batched mode on different samples, as usual
            self._enable_ensembling_ = False
            self.base_model.train()
            x = self._forward_single(x)

            # Store the last mean, median and variance
            self.last_mean = x
            self.last_median = x
            self.last_variance = torch.zeros_like(x)

            return x

        # Otherwise, in eval() mode, run ensembling
        # Enable dropout by keeping the model in training mode but no grad)
        self._enable_ensembling_ = True
        self.base_model.train()

        with torch.no_grad():
            if B == 1:
                # Single input, expand to tensor in batch size
                # Output keeps the shape shape if B == 1
                x_repeated = x.expand(self.ensemble_size, -1)
                x = self._forward_single(x=x_repeated)  # [T, out]

            else:
                x = torch.stack([
                    self._forward_single(x=x) for _ in range(self.ensemble_size)
                ])  # [T, B, out]

        # Compute mean, median and variance
        # NOTE: batch dimension is always preserved, no need to do manual squeeze
        self.last_mean = x.mean(dim=0)
        self.last_median = x.median(dim=0).values
        self.last_variance = x.var(dim=0)

        if B == 1:
            # If input was a single sample, return the mean
            self.last_mean = self.last_mean.unsqueeze(0)
            self.last_median = self.last_median.unsqueeze(0)
            self.last_variance = self.last_variance.unsqueeze(0)

        return self.last_mean

# %% Template model classes for ConvNets
# TemplateConvNet2d
class TemplateConvNet2d(AutoForgeModule):
    """
    TemplateConvNet2d is a configurable 2D convolutional neural network template.

    This class builds a sequential stack of convolutional blocks based on the provided
    configuration. It supports optional linear output layers and can save intermediate
    feature maps during the forward pass.

    Args:
        cfg (TemplateConvNet2dConfig): Configuration dataclass specifying architecture parameters.

    Attributes:
        cfg (TemplateConvNet2dConfig): Stores the configuration.
        blocks (nn.ModuleList): List of convolutional blocks.
        regressor_sequential (nn.ModuleList): Optional regressor layers after convolutional blocks.
        out_channels_sizes (list[int]): Output channels for each convolutional block.
        num_of_conv_blocks (int): Number of convolutional blocks.

    Raises:
        ValueError: If required configuration fields are missing or inconsistent.

    Example:
        >>> cfg = TemplateConvNet2dConfig(
        ...     kernel_sizes=[3, 3],
        ...     pool_kernel_sizes=[2, 2],
        ...     out_channels_sizes=[16, 32],
        ...     num_input_channels=3
        ... )
        >>> model = TemplateConvNet2d(cfg)
        >>> x = torch.randn(1, 3, 64, 64)
        >>> out, features = model(x)
    """
    def __init__(self, cfg: TemplateConvNet2dConfig) -> None:
        super().__init__()

        self.cfg = cfg

        # Build architecture model
        kernel_sizes = cfg.kernel_sizes
        pool_kernel_sizes = cfg.pool_kernel_sizes

        if kernel_sizes is None or pool_kernel_sizes is None:
            raise ValueError(
                'Kernel and pooling kernel sizes must not be none')

        if isinstance(pool_kernel_sizes, list):
            if len(kernel_sizes) != len(pool_kernel_sizes):
                raise ValueError('Kernel and pooling kernel sizes must have the same length')
        else:
            raise ValueError('pool_kernel_sizes cannot be scalar')

        # Define output layer if required by config
        self.linear_output_layer: nn.Module | None

        # Model options
        self.out_channels_sizes = cfg.out_channels_sizes
        self.num_of_conv_blocks = len(kernel_sizes)

        # Additional checks
        if self.out_channels_sizes is None:
            raise ValueError(
                'TemplateConvNet2dConfig: out_channels_sizes cannot be None')

        # Model architecture
        self.blocks = nn.ModuleList()
        idLayer = 0

        # Convolutional blocks auto building
        in_channels = cfg.num_input_channels
        for ith in range(len(kernel_sizes)):

            # Get data for ith block
            kernel_size = kernel_sizes[ith]
            pool_kernel_size = pool_kernel_sizes[ith]
            out_channels = self.out_channels_sizes[ith]
            pool_type = cfg.pool_type
            activ_type_ = cfg.activ_type
            regularization_layer_type_ = cfg.regularization_layer_type
            regularizer_param_ = cfg.regularizer_param
            conv_stride_ = cfg.conv_stride

            # Convolutional blocks
            block = ConvolutionalBlock2d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         pool_kernel_size=pool_kernel_size,
                                         pool_type=pool_type,  # type:ignore
                                         activ_type=activ_type_,
                                         regularizer_type=regularization_layer_type_,
                                         regularizer_param=regularizer_param_,
                                         conv_stride=conv_stride_,  # type:ignore
                                         init_method_type=cfg.init_method_type,
                                         prelu_params=cfg.prelu_params)

            self.blocks.append(block)

            # Update data for next block
            in_channels = out_channels
            idLayer += 1

        self.regressor_sequential = nn.ModuleList()

        # TODO (PC) add support for additional vector input to output layer block
        # Specified size in cfg --> cfg.size_skip_to_linear_output size

        if cfg.linear_output_layer_size is not None:
            
            if cfg.linear_input_layer_size is None:
                cfg.linear_input_layer_size = in_channels

            if in_channels != cfg.linear_input_layer_size:
                
                # Add convolutional "expander"
                self.regressor_sequential.append(
                    nn.Conv2d(in_channels, 
                              cfg.linear_input_layer_size, 
                              1, 
                              1))

            # Fully Connected regressor layer
            self.regressor_sequential.append(nn.AdaptiveAvgPool2d((1, 1)))
            self.regressor_sequential.append(module=nn.Flatten())
            self.regressor_sequential.append(nn.Linear(in_channels,
                                                  cfg.linear_output_layer_size, bias=True))

        # Update config
        self.cfg = cfg

        # Initialize weights of layers
        self.__initialize_weights__(init_method_type=cfg.init_method_type)

    def __initialize_weights__(self,
                               init_method_type: Literal["xavier_uniform",
                                                         "kaiming_uniform",
                                                         "xavier_normal",
                                                         "kaiming_normal",
                                                         "orthogonal"] = "xavier_normal") -> None:
        """
        Initializes the weights of the model layers using the specified initialization method.

        Args:
            init_method_type (Literal["xavier_uniform", "kaiming_uniform", "xavier_normal", "kaiming_normal", "orthogonal"], optional):
            The type of weight initialization to use. Defaults to "xavier_uniform".

        Returns:
            None
        """

        # Initialize blocks calling init weight method
        for block in self.blocks:
            block.__initialize_weights__(init_method_type=init_method_type)

    def forward(self, X: torch.Tensor):
        """
        Generic forward pass for TemplateConvNet2d.
        If a tuple is provided, it expects (x, x_skips) where x_skips can be a tensor or a list of tensors.
        Processes input through convolutional blocks while optionally merging skip features.
        """
        x = X
        x_features = []
        
        # Iterate through each block in the network
        for idx, block in enumerate(self.blocks):

            x = block(x)
            if self.cfg.save_intermediate_features:
                x_features.append(x)

        # Loop through regressor_sequential
        for regr_layer in self.regressor_sequential:
            x = regr_layer(x)

        return x, x_features


# TemplateConvNet2dWithInputSkip
class FeatureMapFuserConv2dBlock(nn.Module):
    def __init__(self, 
                 fuser_module : FeatureMapFuser, 
                 conv2d_block : ConvolutionalBlock2d):
        super().__init__()
        self.fuser = fuser_module
        self.conv = conv2d_block

    def forward(self, x: torch.Tensor, skip: torch.Tensor):

        x = self.fuser(x, skip)
        x = self.conv(x)
        return x
    
    def __initialize_weights__(self,
                               init_method_type: init_methods = "xavier_normal") -> None:

        # Initialize conv calling init weight method
        self.conv.__initialize_weights__(init_method_type=init_method_type)

class TemplateConvNetFeatureFuser2d(AutoForgeModule):
    def __init__(self, cfg: TemplateConvNetFeatureFuser2dConfig) -> None:
        super().__init__()

        self.cfg = cfg

        # Build architecture model
        kernel_sizes = cfg.kernel_sizes
        pool_kernel_sizes = cfg.pool_kernel_sizes

        if kernel_sizes is None or pool_kernel_sizes is None:
            raise ValueError(
                'Kernel and pooling kernel sizes must not be none')

        if isinstance(pool_kernel_sizes, list):
            if len(kernel_sizes) != len(pool_kernel_sizes):
                raise ValueError(
                    'Kernel and pooling kernel sizes must have the same length')
        else:
            raise ValueError('pool_kernel_sizes cannot be scalar')

        # Define output layer if required by config
        self.linear_output_layer: nn.Module | None

        # Model options
        self.out_channels_sizes = cfg.out_channels_sizes
        self.num_of_conv_blocks = len(kernel_sizes)

        # Additional checks
        if self.out_channels_sizes is None:
            raise ValueError('TemplateConvNet2dConfig: out_channels_sizes cannot be None')

        if len(self.cfg.merge_module_index) == 0:
            raise ValueError('merge_module_index is empty. Please user TemplateConvNet2d if no feature map fusion is needed.')

        assert len(cfg.merge_module_index) == len(cfg.merge_module_type) == len(cfg.num_skip_channels), "merge_module_index, merge_modes, num_skip_channels must be aligned in size."

        # Model architecture
        self.blocks = nn.ModuleList()
        idLayer = 0

        # Convolutional blocks auto building
        in_channels = cfg.num_input_channels
        for ith in range(len(kernel_sizes)):

            # Get data for ith convolutional block
            kernel_size = kernel_sizes[ith]
            pool_kernel_size = pool_kernel_sizes[ith]
            out_channels = self.out_channels_sizes[ith]
            pool_type = cfg.pool_type
            activ_type_ = cfg.activ_type
            regularization_layer_type_ = cfg.regularization_layer_type
            regularizer_param_ = cfg.regularizer_param
            conv_stride_ = cfg.conv_stride

            # Determine if fuser is defined for ith index (looking up the self.cfg.merge_module_index)
            try:
                fuser_idx = self.cfg.merge_module_index.index(ith)
            except ValueError:
                fuser_idx = None 

            if fuser_idx is not None:
                # Define ith fuser module
                fuser_module = FeatureMapFuser(num_dims=4, 
                                            fuser_type=cfg.merge_module_type[fuser_idx],
                                            in_channels=in_channels,
                                            num_skip_channels=cfg.num_skip_channels[fuser_idx],
                                            num_attention_heads=cfg.num_attention_heads[fuser_idx],
                                            )
            else:
                fuser_module = FeatureMapFuser(num_dims=4,
                                               fuser_type="identity",
                                               in_channels=1)
                
            # Convolutional blocks  
            conv_block = ConvolutionalBlock2d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         pool_kernel_size=pool_kernel_size,
                                         pool_type=pool_type,  # type:ignore
                                         activ_type=activ_type_,
                                         regularizer_type=regularization_layer_type_,
                                         regularizer_param=regularizer_param_,
                                         conv_stride=conv_stride_,  # type:ignore
                                         init_method_type=cfg.init_method_type,
                                         prelu_params=cfg.prelu_params)

            # Define block to append
            self.blocks.append(FeatureMapFuserConv2dBlock(fuser_module, conv_block))

            # Update data for next block
            in_channels = out_channels
            idLayer += 1

        # Add output layers if specified in config
        self.regressor_sequential = nn.ModuleList()

        if cfg.linear_output_layer_size is not None:

           # If user did not specify it, use in_channels size as linear input size
            if cfg.linear_input_layer_size is None:
                cfg.linear_input_layer_size = in_channels
            
            if in_channels != cfg.linear_input_layer_size:
                # Add convolutional "expander"
                self.regressor_sequential.append(
                    nn.Conv2d(in_channels, 
                              cfg.linear_input_layer_size, 
                              1, 
                              1))

            # Fully Connected regressor layer
            self.regressor_sequential.append(nn.AdaptiveAvgPool2d((1, 1)))
            self.regressor_sequential.append(module=nn.Flatten())
            self.regressor_sequential.append(nn.Linear(cfg.linear_input_layer_size,
                                                  cfg.linear_output_layer_size, bias=True))

        # Initialize weights of layers
        self.__initialize_weights__(init_method_type=cfg.init_method_type)

    def __initialize_weights__(self,
                               init_method_type: init_methods = "xavier_normal") -> None:
        """
        Initializes the weights of the model layers using the specified initialization method.

        Args:
            init_method_type (Literal["xavier_uniform", "kaiming_uniform", "xavier_normal", "kaiming_normal", "orthogonal"], optional):
            The type of weight initialization to use. Defaults to "xavier_uniform".

        Returns:
            None
        """

        # Initialize blocks calling init weight method
        for block in self.blocks:
            block.__initialize_weights__(init_method_type=init_method_type)

    def forward(self, X: tuple[torch.Tensor, list[torch.Tensor]]):
        """
        Generic forward pass for TemplateConvNet2d.
        If a tuple is provided, it expects (x, x_skips) where x_skips can be a tensor or a list of tensors.
        Processes input through convolutional blocks while optionally merging skip features.
        """

        # Unpack tuple
        x, x_skips = X
        x_features = []
        
        # Iterate through each block in the network
        for idx, block in enumerate(self.blocks):
            # If x_skips is a list, get the corresponding skip connection
            x = block(x, x_skips[idx])

            if self.cfg.save_intermediate_features:
                x_features.append(x)
        
        # Loop through regressor_sequential
        for regr_layer in self.regressor_sequential:
            x = regr_layer(x)

        return x, x_features

# %% Template model classes for ConvNets
# TemplateFullyConnectedNet
class TemplateFullyConnectedNet(AutoForgeModule):
    def __init__(self, cfg: TemplateFullyConnectedNetConfig):
        super().__init__()

        # Store configuration
        self.cfg = cfg

        in_channels = cfg.input_layer_size
        output_layer_size = cfg.output_layer_size

        if self.cfg.out_channels_sizes is None:
            raise ValueError(
                'TemplateFullyConnectedNetConfig: out_channels_sizes cannot be None')

        if in_channels is None:
            raise ValueError(
                "TemplateFullyConnectedNetConfig: 'input_layer_size' cannot be None")
        if output_layer_size is None:
            raise ValueError(
                "TemplateFullyConnectedNetConfig: 'output_layer_size' cannot be None")

        # Model architecture
        self.blocks = nn.ModuleList()
        idLayer = 0

        for ith, out_channels in enumerate(self.cfg.out_channels_sizes):

            # Get data for ith block
            activ_type_ = cfg.activ_type
            regularization_layer_type_ = cfg.regularization_layer_type
            regularizer_param_ = cfg.regularizer_param

            # Build ith block
            block = FullyConnectedBlock(in_channels=in_channels,
                                        out_channels=out_channels,
                                        activ_type=activ_type_,
                                        regularizer_type=regularization_layer_type_,
                                        regularizer_param=regularizer_param_,
                                        init_method_type=cfg.init_method_type,
                                        prelu_params=cfg.prelu_params,)

            self.blocks.append(block)

            # Update data for next block
            in_channels = out_channels
            idLayer += 1

        # Add output layer
        self.blocks.append(FullyConnectedBlock(in_channels,
                                               output_layer_size,
                                               activ_type="none",
                                               regularizer_type="none",
                                               regularizer_param=0.0,
                                               init_method_type=cfg.init_method_type,
                                               prelu_params=cfg.prelu_params))
        idLayer += 1

        # Initialize weights of layers
        self.__initialize_weights__(init_method_type=cfg.init_method_type)

    def __initialize_weights__(self, init_method_type: Literal["xavier_uniform",
                                                               "kaiming_uniform",
                                                               "xavier_normal",
                                                               "kaiming_normal",
                                                               "orthogonal"] = "xavier_normal"):
        """
        Initializes the weights of the model layers using the specified initialization method.

        Args:
            init_method_type (Literal["xavier_uniform", "kaiming_uniform", "xavier_normal", "kaiming_normal", "orthogonal"], optional):
            The type of weight initialization to use. Defaults to "xavier_normal".

        Returns:
            None
        """

        # Initialize blocks calling init weight method
        for block in self.blocks:
            block.__initialize_weights__(init_method_type=init_method_type)

    def forward(self,
                X: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> torchFunc.Tensor:
        """
        Performs the forward pass for the TemplateFullyConnectedNet model.

        Args:
            X (torch.Tensor | tuple[torch.Tensor]): Input tensor or tuple containing input and optional skip connections.

        Returns:
            torch.Tensor: Output tensor after passing through all blocks.
        """

        # Initialize output
        if not torch.is_tensor(X):
            x = X[0]
            X_skips = X[1]

            # Flatten and concatenate to input X along batch dimension
            x = torch.cat((x, X_skips), dim=0)
        else:
            x = X

        # Forward through each block
        for block in self.blocks:
            x = block(x)

        return x

    @classmethod
    def build_dropout_ensemble(cls, cfg: TemplateFullyConnectedNetConfig) -> DropoutEnsemblingNetworkWrapper:
        base_model = cls(cfg)
        return DropoutEnsemblingNetworkWrapper(base_model,
                                               ensemble_size=cfg.dropout_ensemble_size,
                                               enable_ensembling=True)

# DEVNOTE Legacy class to remove
class TemplateFullyConnectedDeepNet(AutoForgeModule):
    '''
    Template class for a fully parametric Deep NN model in PyTorch. Inherits from AutoForgeModule class (nn.Module enhanced class).
    '''

    def __init__(self, cfg: TemplateFullyConnectedNetConfig) -> None:
        super().__init__()

        self.cfg = cfg
        self.model_name = cfg.model_name

        regularization_layer_type = cfg.regularization_layer_type
        dropout_probability = cfg.regularizer_param

        if regularization_layer_type == 'batchnorm':
            self.use_batchnorm = True

        elif regularization_layer_type == 'dropout':
            self.use_batchnorm = False

        elif regularization_layer_type == 'groupnorm':
            raise NotImplementedError(
                'Group normalization is not implemented yet. Please use batch normalization or dropout instead.')
        elif regularization_layer_type == 'none':
            self.dropout_probability = 0.0
            self.use_batchnorm = False
        else:
            raise ValueError(
                f"TemplateFullyConnectedDeepNet: regularization_layer_type must be 'batchnorm', 'dropout', 'groupnorm' or 'none'. Found {regularization_layer_type}.")

        # Get sizes
        input_size = cfg.input_layer_size
        out_channels_sizes = cfg.out_channels_sizes

        if out_channels_sizes is None:
            raise ValueError(
                'TemplateFullyConnectedDeepNet: out_channels_sizes cannot be None')

        if input_size is None:
            raise ValueError(
                'TemplateFullyConnectedDeepNet: input_size cannot be None')

        # Model parameters
        self.out_channels_sizes = out_channels_sizes
        self.use_batchnorm = self.use_batchnorm

        self.num_layers = len(self.out_channels_sizes)

        # Model architecture
        self.layers = nn.ModuleList()
        idLayer = 0

        # Fully Connected autobuilder
        self.layers.append(nn.Flatten())

        for i in range(idLayer, self.num_layers+idLayer-1):

            # Fully Connected layers block
            self.layers.append(
                nn.Linear(input_size, self.out_channels_sizes[i], bias=True))
            self.layers.append(nn.PReLU(self.out_channels_sizes[i]))

            # Dropout is inhibited if batch normalization
            if not self.use_batchnorm and dropout_probability > 0:
                self.layers.append(nn.Dropout(dropout_probability))

            # Add batch normalization layer if required
            if self.use_batchnorm:
                self.layers.append(nn.BatchNorm1d(
                    self.out_channels_sizes[i], eps=1E-5, momentum=0.1, affine=True))

            # Update input size for next layer
            input_size = self.out_channels_sizes[i]

        # Add output layer
        self.layers.append(
            nn.Linear(input_size, self.out_channels_sizes[-1], bias=True))

        # Initialize weights of layers
        self.__initialize_weights__()

    def __initialize_weights__(self):
        '''Weights Initialization function for layers of the model. Xavier --> layers with tanh and sigmoid, Kaiming --> layers with ReLU activation'''

        for layer in self.layers:
            # Check if layer is a Linear layer
            if isinstance(layer, nn.Linear):
                # Apply Kaiming initialization
                nn.init.kaiming_uniform_(
                    layer.weight, nonlinearity='leaky_relu')
                if layer.bias is not None:
                    # Initialize bias to zero if present
                    nn.init.constant_(layer.bias, 0)

    def forward(self, inputSample):
        # Perform forward pass iterating through all layers of DNN
        val = inputSample
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                val = layer(val)
            elif isinstance(layer, nn.PReLU):
                val = torchFunc.prelu(val, layer.weight)
            elif isinstance(layer, nn.Dropout):
                val = layer(val)
            elif isinstance(layer, nn.BatchNorm1d):
                val = layer(val)
            elif isinstance(layer, nn.Flatten):
                if len(val.shape) > 1:
                    val = layer(val)

        # Output layer
        prediction = val

        return prediction

    @staticmethod
    def build_dropout_ensemble(cls, cfg: TemplateFullyConnectedNetConfig) -> DropoutEnsemblingNetworkWrapper:
        base_model = cls(cfg)
        return DropoutEnsemblingNetworkWrapper(base_model,
                                               ensemble_size=cfg.dropout_ensemble_size,
                                               enable_ensembling=True)

# %% Image normalization classes
class NormalizeImg(nn.Module):
    def __init__(self, normaliz_factor: float = 255.0):
        super(NormalizeImg, self).__init__()

        # Register parameters as non-trainable (constant)
        self.register_buffer('normaliz_value',
                             torch.tensor(normaliz_factor))

    def forward(self, x):
        return x / self.normaliz_value  # [0, normaliz_value] -> [0, 1]

# Define the Denormalize layer
class DenormalizeImg(nn.Module):
    def __init__(self, normaliz_factor: float = 255.0):
        super(DenormalizeImg, self).__init__()

        # Register parameters as non-trainable (constant)
        self.register_buffer('normaliz_value', torch.tensor(normaliz_factor))

    def forward(self, x):
        return x * self.normaliz_value  # [0, 1] -> [0, normaliz_value]
