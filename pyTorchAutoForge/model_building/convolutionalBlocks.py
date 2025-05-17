import torch
from torch import nn
from typing import Literal


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
        pool_type: Literal["MaxPool1d", "AvgPool1d",
                           "Adapt_MaxPool1d", "Adapt_AvgPool1d", "none"] = "MaxPool1d",
        activ_type: Literal["prelu", "sigmoid",
                            "relu", "tanh", "none"] = "prelu",
        regularizer_type: Literal["dropout",
                                  "batchnorm", "groupnorm", "none"] = "none",
        regularizer_param: float | int = 0.0,
        conv_stride: int = 1,
        conv_padding: int = 0,
        conv_dilation: int = 1,
        prelu_params: Literal["all", "unique"] = "unique",
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
        match activ_type.lower():
            case "prelu":
                num_p = out_channels if prelu_params == "all" else 1
                self.activ = nn.PReLU(num_p)

            case "relu": self.activ = nn.ReLU()
            case "sigmoid": self.activ = nn.Sigmoid()
            case "tanh": self.activ = nn.Tanh()
            case "none": self.activ = nn.Identity()
            case _: raise ValueError(f"Unsupported activation: {activ_type}")

        # Pooling selection
        self.pool: nn.Module | nn.Identity = nn.Identity()
        match pool_type.lower():
            case "maxpool1d": self.pool = nn.MaxPool1d(pool_kernel_size, stride=1)
            case "avgpool1d": self.pool = nn.AvgPool1d(pool_kernel_size, stride=1)

            case pt if pt.startswith("adapt_"):

                target = kwargs.get("target_res")

                if target is None:
                    raise ValueError(
                        "target_res required for adaptive pooling")
                self.pool = nn.AdaptiveMaxPool1d(
                    target) if pt == "adapt_maxpool1d" else nn.AdaptiveAvgPool1d(target)

            case "none": self.pool = nn.Identity()
            case _: raise ValueError(f"Unsupported pool: {pool_type}")

        # Regularizer selection
        self.regularizer: nn.Module | nn.Identity = nn.Identity()
        match regularizer_type.lower():
            case "dropout":
                assert 0 < regularizer_param < 1
                self.regularizer = nn.Dropout1d(regularizer_param)
            case "batchnorm": self.regularizer = nn.BatchNorm1d(out_channels)
            case "groupnorm":
                assert regularizer_param > 0
                self.regularizer = nn.GroupNorm(
                    int(regularizer_param), out_channels)

            case "none": self.regularizer = nn.Identity()
            case _: raise ValueError(f"Unsupported regularizer: {regularizer_type}")

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
        pool_type: Literal["MaxPool2d", "AvgPool2d",
                           "Adapt_MaxPool2d", "Adapt_AvgPool2d", "none"] = "MaxPool2d",
        activ_type: Literal["prelu", "sigmoid",
                            "relu", "tanh", "none"] = "prelu",
        regularizer_type: Literal["dropout",
                                  "batchnorm", "groupnorm", "none"] = "none",
        regularizer_param: float | int = 0.0,
        conv_stride: int = 1,
        conv_padding: int = 0,
        conv_dilation: int = 1,
        prelu_params: Literal["all", "unique"] = "unique",
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

        if activ_type.lower() == "prelu":
            prelu_out = out_channels if prelu_params == "all" else 1
            self.activ = nn.PReLU(prelu_out)

        elif activ_type.lower() == "relu":
            self.activ = nn.ReLU()

        elif activ_type.lower() == "sigmoid":
            self.activ = nn.Sigmoid()

        elif activ_type.lower() == "tanh":
            self.activ = nn.Tanh()

        # TODO add more fancy activations
        elif activ_type.lower() == "none":
            pass
        else:
            raise ValueError(f"Unsupported activation type: {activ_type}")

        # Pooling selection
        self.pool: nn.Module | nn.Identity = nn.Identity()
        if pool_type.lower() == "maxpool2d":
            self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=1)

        elif pool_type.lower() == "avgpool2d":
            self.pool = nn.AvgPool2d(kernel_size=pool_kernel_size, stride=1)

        elif pool_type.lower().startswith("adapt"):

            # Expect pool_type to be "adapt_maxpool2d" or "adapt_avgpool2d"
            # Use target_res argument for output size (must be provided)
            target_res = kwargs.get("target_res", None)

            if target_res is None:
                raise ValueError(
                    "target_res (H, W) must be provided for adaptive pooling")

            if pool_type.lower() == "adapt_maxpool2d":
                self.pool = nn.AdaptiveMaxPool2d(target_res)

            elif pool_type.lower() == "adapt_avgpool2d":
                self.pool = nn.AdaptiveAvgPool2d(target_res)

            else:
                raise ValueError(
                    f"Unsupported adaptive pool type: {pool_type}")

        elif pool_type.lower() == "none":
            pass
        else:
            raise ValueError(f"Unsupported pool type: {pool_type}")

        # Regularizer selection
        self.regularizer: nn.Module | nn.Identity = nn.Identity()

        if regularizer_type.lower() == "dropout":
            # Use regularizer_param as dropout probability
            assert (regularizer_param > 0 and regularizer_param <
                    1), "Invalid dropout probability: must be in (0,1)"
            self.regularizer = nn.Dropout2d(regularizer_param)

        elif regularizer_type.lower() == "batchnorm":
            self.regularizer = nn.BatchNorm2d(out_channels)

        elif regularizer_type.lower() == "groupnorm":
            assert (regularizer_param !=
                    0), "Invalid group norms count: must be > 0"

            # Use regularizer_param as num_groups
            self.regularizer = nn.GroupNorm(
                int(regularizer_param), out_channels)

        elif regularizer_type.lower() == "none":
            pass

        else:
            raise ValueError(
                f"Unsupported regularizer type: {regularizer_type}")

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
        pool_type: Literal["MaxPool3d", "AvgPool3d",
                           "Adapt_MaxPool3d", "Adapt_AvgPool3d", "none"] = "MaxPool3d",
        activ_type: Literal["prelu", "sigmoid",
                            "relu", "tanh", "none"] = "prelu",
        regularizer_type: Literal["dropout",
                                  "batchnorm", "groupnorm", "none"] = "none",
        regularizer_param: float | int = 0.0,
        conv_stride: int | tuple[int, int, int] = 1,
        conv_padding: int | tuple[int, int, int] = 0,
        conv_dilation: int | tuple[int, int, int] = 1,
        prelu_params: Literal["all", "unique"] = "unique",
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
        match activ_type.lower():
            case "prelu":
                num_p = out_channels if prelu_params == "all" else 1
                self.activ = nn.PReLU(num_p)
            case "relu": self.activ = nn.ReLU()
            case "sigmoid": self.activ = nn.Sigmoid()
            case "tanh": self.activ = nn.Tanh()
            case "none": self.activ = nn.Identity()
            case _:
                raise ValueError(f"Unsupported activation: {activ_type}")

        # Pooling selection
        self.pool: nn.Module | nn.Identity = nn.Identity()

        match pool_type.lower():
            case "maxpool3d": self.pool = nn.MaxPool3d(pool_kernel_size, stride=1)
            case "avgpool3d": self.pool = nn.AvgPool3d(pool_kernel_size, stride=1)
            case pt if pt.startswith("adapt_"):

                target = kwargs.get("target_res")

                if target is None:
                    raise ValueError(
                        "target_res required for adaptive pooling")

                self.pool = nn.AdaptiveMaxPool3d(
                    target) if pt == "adapt_maxpool3d" else nn.AdaptiveAvgPool3d(target)

            case "none": self.pool = nn.Identity()
            case _:
                raise ValueError(f"Unsupported pool: {pool_type}")

        # Regularizer selection
        self.regularizer: nn.Module | nn.Identity = nn.Identity()

        match regularizer_type.lower():
            case "dropout":
                assert 0 < regularizer_param < 1
                self.regularizer = nn.Dropout3d(regularizer_param)
            case "batchnorm": self.regularizer = nn.BatchNorm3d(out_channels)
            case "groupnorm":
                assert regularizer_param > 0
                self.regularizer = nn.GroupNorm(
                    int(regularizer_param), out_channels)
            case "none": self.regularizer = nn.Identity()
            case _:
                raise ValueError(
                    f"Unsupported regularizer: {regularizer_type}")

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
        raise NotImplementedError('Torch does not have NDim convolutions by default. Implementation needs to rely on custom code or existing modules. TODO')