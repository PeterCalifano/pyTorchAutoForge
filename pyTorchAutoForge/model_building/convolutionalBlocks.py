import torch
from torch import nn
from typing import Literal

# TODO implement an optional "residual connection" feature

from pyTorchAutoForge.model_building.factories.block_factories import _activation_factory, _initialize_convblock_weights, _pooling_factory, _regularizer_factory, activ_types, pooling_types, regularizer_types, init_methods, pooling_types_1d, pooling_types_2d, pooling_types_3d

class ConvolutionalBlock1d(nn.Module):
    """
    ConvolutionalBlock1d is a configurable 1D convolutional block for PyTorch.

    This block includes a convolutional layer, optional activation, pooling, and regularization.
    All components are configurable via constructor arguments.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolving kernel.
        pool_kernel_size (int, optional): Kernel size for pooling. Defaults to 2.
        pool_type (Literal): Pooling type ("MaxPool1d", "AvgPool1d", "Adapt_MaxPool1d", "Adapt_AvgPool1d", "none").
        activ_type (Literal): Activation type ("prelu", "sigmoid", "relu", "tanh", "none").
        regularizer_type (Literal): Regularizer type ("dropout", "batchnorm", "groupnorm", "none").
        regularizer_param (float | int, optional): Parameter for regularizer (dropout probability or group count).
        conv_stride (int, optional): Stride for convolution. Defaults to 1.
        conv_padding (int, optional): Padding for convolution. Defaults to 0.
        conv_dilation (int, optional): Dilation for convolution. Defaults to 0.
        prelu_params (Literal, optional): "all" for per-channel PReLU, "unique" for shared. Defaults to "unique".
        **kwargs: Additional arguments (e.g., target_res for adaptive pooling).

    Raises:
        ValueError: If an unsupported activation, pooling, or regularizer type is provided.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool_kernel_size: int = 2,
        pool_type: pooling_types_1d = "MaxPool1d",
        activ_type: activ_types = "prelu",
        regularizer_type: regularizer_types = "none",
        regularizer_param: float | int = 0.0,
        conv_stride: int = 1,
        conv_padding: int = 0,
        conv_dilation: int = 1,
        prelu_params: Literal["all", "unique"] = "unique",
        init_method_type: init_methods = "xavier_uniform",
        **kwargs
    ):

        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=conv_stride,
            padding=conv_padding,
            dilation=conv_dilation,
        )

        # Activation selection
        self.activ: nn.Module | nn.Identity = nn.Identity()
        self.activ = _activation_factory(activ_type, out_channels, prelu_params)

        # Pooling selection
        self.pool: nn.Module | nn.Identity = nn.Identity()
        self.pool = _pooling_factory(pool_type, 
                                pool_kernel_size, 
                                target_res=kwargs.get("target_res", None))

        # Regularizer selection
        self.regularizer: nn.Module | nn.Identity = nn.Identity()
        self.regularizer = _regularizer_factory(ndims=1,
                                                regularizer_type=regularizer_type,
                                                out_channels=out_channels,
                                                regularizer_param=regularizer_param)

        # Initialize weights using specified method
        self.__initialize_weights__(init_method_type)

    def __initialize_weights__(self,
                           init_method_type: init_methods = "xavier_uniform"):
        """
        Initialize weights using specified method.
        """
        self = _initialize_convblock_weights(self, init_method_type)

    # Simple forward method
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.activ(x)
        x = self.regularizer(x)
        x = self.pool(x)
        return x


class ConvolutionalBlock2d(nn.Module):
    """
    ConvolutionalBlock2d is a configurable 2D convolutional block for PyTorch.

    This block includes a convolutional layer, optional activation, pooling, and regularization.
    All components are configurable via constructor arguments.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolving kernel.
        pool_kernel_size (int, optional): Kernel size for pooling. Defaults to 2.
        pool_type (Literal): Pooling type ("MaxPool2d", "AvgPool2d", "Adapt_MaxPool2d", "Adapt_AvgPool2d", "none").
        activ_type (Literal): Activation type ("prelu", "sigmoid", "relu", "tanh", "none").
        regularizer_type (Literal): Regularizer type ("dropout", "batchnorm", "groupnorm", "none").
        regularizer_param (float | int, optional): Parameter for regularizer (dropout probability or group count).
        conv_stride (int, optional): Stride for convolution. Defaults to 1.
        conv_padding (int, optional): Padding for convolution. Defaults to 0.
        conv_dilation (int, optional): Dilation for convolution. Defaults to 0.
        prelu_params (Literal, optional): "all" for per-channel PReLU, "unique" for shared. Defaults to "unique".
        **kwargs: Additional arguments (e.g., target_res for adaptive pooling).

    Raises:
        ValueError: If an unsupported activation, pooling, or regularizer type is provided.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool_kernel_size: int = 2,
        pool_type: pooling_types_2d = "MaxPool2d",
        activ_type: activ_types = "prelu",
        regularizer_type: regularizer_types = "none",
        regularizer_param: float | int = 0.0,
        conv_stride: int = 1,
        conv_padding: int = 0,
        conv_dilation: int = 1,
        prelu_params: Literal["all", "unique"] = "unique",
        init_method_type: init_methods = "xavier_uniform",
        **kwargs
    ):

        super().__init__()
        # Build conv layer
        self.conv: nn.Module = nn.Conv2d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         stride=conv_stride,
                                         padding=conv_padding,
                                         dilation=conv_dilation)

        # Activation selection
        self.activ: nn.Module | nn.Identity = nn.Identity()
        self.activ = _activation_factory(activ_type, out_channels, prelu_params)

        # Pooling selection
        self.pool: nn.Module | nn.Identity = nn.Identity()
        self.pool = _pooling_factory(pool_type, 
                                     pool_kernel_size, 
                                     target_res=kwargs.get("target_res", None))

        # Regularizer selection
        self.regularizer: nn.Module | nn.Identity = nn.Identity()
        self.regularizer = _regularizer_factory(ndims=2, 
                                                regularizer_type=regularizer_type, 
                                                out_channels=out_channels,
                                                regularizer_param=regularizer_param)

        # Initialize weights using specified method
        self.__initialize_weights__(init_method_type)

    def __initialize_weights__(self,
                           init_method_type: init_methods = "xavier_uniform"):
        """
        Initialize weights using specified method.
        """
        self = _initialize_convblock_weights(self, init_method_type)

    # Simple forward method
    def forward(self, x):

        x = self.conv(x)
        x = self.activ(x)
        x = self.regularizer(x)
        x = self.pool(x)

        return x

class ConvolutionalBlock3d(nn.Module):
    """
    ConvolutionalBlock3d is a configurable 3D convolutional block for PyTorch.

    This block includes a convolutional layer, optional activation, pooling, and regularization.
    All components are configurable via constructor arguments.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int or tuple): Size of the convolving kernel.
        pool_kernel_size (int or tuple, optional): Kernel size for pooling. Defaults to 2.
        pool_type (Literal): Pooling type ("MaxPool3d", "AvgPool3d", "Adapt_MaxPool3d", "Adapt_AvgPool3d", "none").
        activ_type (Literal): Activation type ("prelu", "sigmoid", "relu", "tanh", "none").
        regularizer_type (Literal): Regularizer type ("dropout", "batchnorm", "groupnorm", "none").
        regularizer_param (float | int, optional): Parameter for regularizer (dropout probability or group count).
        conv_stride (int or tuple, optional): Stride for convolution. Defaults to 1.
        conv_padding (int or tuple, optional): Padding for convolution. Defaults to 0.
        conv_dilation (int or tuple, optional): Dilation for convolution. Defaults to 0.
        prelu_params (Literal, optional): "all" for per-channel PReLU, "unique" for shared. Defaults to "unique".
        **kwargs: Additional arguments (e.g., target_res for adaptive pooling).

    Raises:
        ValueError: If an unsupported activation, pooling, or regularizer type is provided.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        pool_kernel_size: int | tuple[int, int, int] = 2,
        pool_type: pooling_types_3d = "MaxPool3d",
        activ_type: activ_types = "prelu",
        regularizer_type: regularizer_types = "none",
        regularizer_param: float | int = 0.0,
        conv_stride: int | tuple[int, int, int] = 1,
        conv_padding: int | tuple[int, int, int] = 0,
        conv_dilation: int | tuple[int, int, int] = 1,
        prelu_params: Literal["all", "unique"] = "unique",
        init_method_type: init_methods = "xavier_uniform",
        **kwargs
    ):

        super().__init__()

        # Build conv layer
        self.conv: nn.Module = nn.Conv3d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         stride=conv_stride,
                                         padding=conv_padding,
                                         dilation=conv_dilation)

        # Activation selection
        self.activ: nn.Module | nn.Identity = nn.Identity()
        self.activ = _activation_factory(
            activ_type, out_channels, prelu_params)

        # Pooling selection
        self.pool: nn.Module | nn.Identity = nn.Identity()
        self.pool = _pooling_factory(pool_type,
                                     pool_kernel_size,
                                     target_res=kwargs.get("target_res", None))

        # Regularizer selection
        self.regularizer: nn.Module | nn.Identity = nn.Identity()
        self.regularizer = _regularizer_factory(ndims=3, 
                                    regularizer_type=regularizer_type, 
                                    out_channels=out_channels,
                                    regularizer_param=regularizer_param)
                
        # Initialize weights of all blocks 
        self.__initialize_weights__(init_method_type)

    def __initialize_weights__(self,
                           init_method_type: Literal["xavier_uniform",
                                                     "kaiming_uniform",
                                                     "xavier_normal",
                                                     "kaiming_normal",
                                                     "orthogonal"] = "xavier_uniform"):
        """
        Initialize weights using specified method.
        """
        self = _initialize_convblock_weights(self, init_method_type)

    # Simple forward method
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.activ(x)
        x = self.regularizer(x)
        x = self.pool(x)
        return x


# TODO
class ConvolutionalBlockNd(nn.Module):
    def __init__(self):
        raise NotImplementedError(
            'Torch does not have NDim convolutions by default. Implementation needs to rely on custom code or existing modules. TODO')
