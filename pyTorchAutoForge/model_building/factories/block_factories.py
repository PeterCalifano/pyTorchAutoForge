from multiprocessing import pool
import torch.nn as nn
from typing import Literal

# %% Type aliases for activation and regularizer types
activ_types = Literal["prelu", "leakyrelu", "relu", "elu", "selu",
                      "gelu", "swish", "softplus", "sigmoid", "tanh", "none"] 

regularizer_types = Literal["dropout", "batchnorm", "groupnorm",
    "layernorm", "instancenorm", "none"]

pooling_types = Literal[
    "MaxPool1d", "AvgPool1d",
    "MaxPool2d", "AvgPool2d",
    "MaxPool3d", "AvgPool3d",
    "Adapt_MaxPool1d", "Adapt_AvgPool1d",
    "Adapt_MaxPool2d", "Adapt_AvgPool2d",
    "Adapt_MaxPool3d", "Adapt_AvgPool3d",
    "none"
]

pooling_types_1d = Literal[
    "MaxPool1d", "AvgPool1d",
    "Adapt_MaxPool1d", "Adapt_AvgPool1d"
]

pooling_types_2d = Literal[
    "MaxPool2d", "AvgPool2d",
    "Adapt_MaxPool2d", "Adapt_AvgPool2d"
]

pooling_types_3d = Literal[
    "MaxPool3d", "AvgPool3d",
    "Adapt_MaxPool3d", "Adapt_AvgPool3d"
]

init_methods = Literal[
    "xavier_uniform", "kaiming_uniform",
    "xavier_normal", "kaiming_normal",
    "orthogonal"
]

# %% Layer initialization methods
def _initialize_convblock_weights(block,
                                  init_method_type: init_methods = "xavier_uniform"):
    """
    Initialize weights using specified method. Assumes the input module has a "conv" attribute. 
    """
    match init_method_type.lower():
        # type:ignore
        case "xavier_uniform": nn.init.xavier_uniform_(block.conv.weight)
        # type:ignore
        case "kaiming_uniform": nn.init.kaiming_uniform_(block.conv.weight)
        case "xavier_normal": nn.init.xavier_normal_(block.conv.weight)
        case "kaiming_normal": nn.init.kaiming_normal_(block.conv.weight)
        case "orthogonal": nn.init.orthogonal_(block.conv.weight)
        case _: raise ValueError(f"Unsupported initialization method: {init_method_type}")

    # Initialize biases to zero
    if block.conv.bias is not None:
        nn.init.zeros_(block.conv.bias)

def _initialize_fcnblock_weights(block,
                                 init_method_type: init_methods = "xavier_uniform"):
    """
    Initializes the weights of the linear layer using the specified initialization method.

    Args:
        init_method_type (str): The initialization method to use.
        One of "xavier_uniform", "kaiming_uniform", "xavier_normal",
        "kaiming_normal", or "orthogonal".
    """
    match init_method_type.lower():
        # type:ignore
        case "xavier_uniform": nn.init.xavier_uniform_(block.linear.weight)
        # type:ignore
        case "kaiming_uniform": nn.init.kaiming_uniform_(block.linear.weight)
        case "xavier_normal": nn.init.xavier_normal_(block.linear.weight)
        case "kaiming_normal": nn.init.kaiming_normal_(block.linear.weight)
        case "orthogonal": nn.init.orthogonal_(block.linear.weight)
        case _: raise ValueError(f"Unsupported initialization method: {init_method_type}")

    # Initialize biases to zero
    if block.linear.bias is not None:
        nn.init.zeros_(block.linear.bias)

# %% Block factories
def _activation_factory(activ_type: activ_types,
                        out_channels: int, 
                        prelu_params: str) -> nn.Module:
    """
    Factory function to create activation modules based on the specified type.

    Args:
        activ_type (str): The activation type to use. Supported values are
            "prelu", "leakyrelu", "relu", "elu", "selu", "gelu", "swish",
            "softplus", "sigmoid", "tanh", "none".
        out_channels (int): Number of output channels, used for certain activations like PReLU.
        prelu_params (str): If "all", use separate PReLU parameters per channel; if "unique", use one parameter.

    Raises:
        ValueError: If an unsupported activation type is provided.

    Returns:
        nn.Module: The corresponding activation module.
    """
    match activ_type.lower():
        case "prelu":
            num_p = out_channels if prelu_params == "all" else 1
            return nn.PReLU(num_p)
        case "leakyrelu":
            # you can expose a slope parameter if you like, e.g. 0.01
            return nn.LeakyReLU(negative_slope=0.01, inplace=True)
        case "relu":
            return nn.ReLU(inplace=True)
        case "elu":
            # ELU with alpha=1.0 by default
            return nn.ELU(alpha=1.0, inplace=True)
        case "selu":
            # SELU is self-normalizing
            return nn.SELU(inplace=True)
        case "gelu":
            # Gaussian Error Linear Unit
            return nn.GELU()
        case "swish":
            # PyTorch 1.7+ has SiLU which is a form of Swish
            return nn.SiLU(inplace=True)
        case "softplus":
            # smooth approximation to ReLU
            return nn.Softplus()
        case "sigmoid":
            return nn.Sigmoid()
        case "tanh":
            return nn.Tanh()
        case "none":
            return nn.Identity()
        case _:
            raise ValueError(f"Unsupported activation type: {activ_type}")


def _pooling_factory(pool_type: pooling_types,
                     kernel_size: int | tuple[int, int] | tuple[int, int, int],
                     stride: int | tuple[int] | tuple[int, int] | tuple[int, int, int] | None = None,
                     padding: int | tuple[int] | tuple[int, int] | tuple[int, int, int] = 0,
                     target_res: int | tuple[int] | None = None
                     ) -> nn.MaxPool1d | nn.AvgPool1d | nn.MaxPool2d | nn.AvgPool2d | nn.MaxPool3d | nn.AvgPool3d | nn.AdaptiveMaxPool1d | nn.AdaptiveAvgPool1d | nn.AdaptiveMaxPool2d | nn.AdaptiveAvgPool2d | nn.AdaptiveMaxPool3d | nn.AdaptiveAvgPool3d | nn.Identity:
    """
    Factory function to create pooling layers for convolutional blocks.

    Supports 1D, 2D, and 3D pooling layers.

    Args:
        pool_type (str): The pooling type. Supported values are:
            - "MaxPool1d", "AvgPool1d"
            - "MaxPool2d", "AvgPool2d"
            - "MaxPool3d", "AvgPool3d"
            - "Adapt_MaxPool1d", "Adapt_AvgPool1d"
            - "Adapt_MaxPool2d", "Adapt_AvgPool2d"
            - "Adapt_MaxPool3d", "Adapt_AvgPool3d"
            - "none"
        kernel_size (int or tuple[int]): Kernel size for the pooling layer.
            Not used for adaptive pooling.
        stride (int or tuple[int], optional): Stride for the pooling layer.
            Defaults to 1 if not provided.
        padding (int or tuple[int], optional): Padding for the pooling layer.
            Defaults to 0.
        target_res (int or tuple[int], optional): Target output size for adaptive pooling.
            Required if pool_type is adaptive.

    Returns:
        nn.Module: The corresponding pooling layer.

    Raises:
        ValueError: If an unsupported pooling type is provided or target_res is required but not provided.
    """
    pool_type_lower: str = pool_type.lower()
    default_stride = stride if stride is not None else kernel_size

    match pool_type_lower:
        case "maxpool1d":
            return nn.MaxPool1d(kernel_size, stride=default_stride, padding=padding)
        case "avgpool1d":
            return nn.AvgPool1d(kernel_size, stride=default_stride, padding=padding)
        case "maxpool2d":
            return nn.MaxPool2d(kernel_size, stride=default_stride, padding=padding)
        case "avgpool2d":
            return nn.AvgPool2d(kernel_size, stride=default_stride, padding=padding)
        case "maxpool3d":
            return nn.MaxPool3d(kernel_size, stride=default_stride, padding=padding)
        case "avgpool3d":
            return nn.AvgPool3d(kernel_size, stride=default_stride, padding=padding)
        
        case pt if pt.startswith("adapt_"):
            if target_res is None:
                raise ValueError("target_res is required for adaptive pooling")
            if pool_type_lower in ("adapt_maxpool1d", "adapt_maxpool_1d"):
                return nn.AdaptiveMaxPool1d(target_res)
            elif pool_type_lower in ("adapt_avgpool1d", "adapt_avgpool_1d"):
                return nn.AdaptiveAvgPool1d(target_res)
            elif pool_type_lower in ("adapt_maxpool2d", "adapt_maxpool_2d"):
                return nn.AdaptiveMaxPool2d(target_res)
            elif pool_type_lower in ("adapt_avgpool2d", "adapt_avgpool_2d"):
                return nn.AdaptiveAvgPool2d(target_res)
            elif pool_type_lower in ("adapt_maxpool3d", "adapt_maxpool_3d"):
                return nn.AdaptiveMaxPool3d(target_res)
            elif pool_type_lower in ("adapt_avgpool3d", "adapt_avgpool_3d"):
                return nn.AdaptiveAvgPool3d(target_res)
            else:
                raise ValueError(f"Unsupported adaptive pooling type: {pool_type}")
            
        case "none":
            return nn.Identity()
        case _:
            raise ValueError(f"Unsupported pooling type: {pool_type}")


def _regularizer_factory(ndims: int,
                         regularizer_type: regularizer_types,
                         out_channels: int,
                         regularizer_param: float | int | None = None
                         ) -> nn.Module:
    """
    Factory function to create a regularization module for both convolutional
    and fully-connected blocks.

    Args:
        regularizer_type (str): The regularizer type. Supported values are:
            "dropout", "batchnorm", "groupnorm", "layernorm", "instancenorm", "none".
        out_channels (int): Number of output channels or features.
        ndims (int): Number of dimensions of the block.
            Use 1 for fully-connected/1D convolutional blocks,
            2 for 2D convolutional blocks, and 3 for 3D convolutional blocks.
        regularizer_param (float | int | None): Parameter for the regularizer.
            Required for:
                - dropout (a float in (0, 1))
                - groupnorm (a positive integer for the number of groups)
            Ignored for other types; can be left as None.

    Returns:
        nn.Module: The corresponding regularization layer.

    Raises:
        ValueError: If an unsupported regularizer type is provided or required parameters are missing or invalid.
    """
    reg_type = regularizer_type.lower()

    if reg_type == "dropout":
        if regularizer_param is None:
            raise ValueError(
                "Dropout regularizer requires a dropout rate (float in (0,1))")
        if not (0 < regularizer_param < 1):
            raise ValueError("Dropout rate must be in the range (0,1)")
        if ndims == 1:
            return nn.Dropout(regularizer_param)
        elif ndims == 2:
            return nn.Dropout2d(regularizer_param)
        elif ndims == 3:
            return nn.Dropout3d(regularizer_param)
        else:
            raise ValueError(
                f"Unsupported number of dimensions for dropout: {ndims}")

    elif reg_type == "batchnorm":
        if ndims == 1:
            return nn.BatchNorm1d(out_channels)
        elif ndims == 2:
            return nn.BatchNorm2d(out_channels)
        elif ndims == 3:
            return nn.BatchNorm3d(out_channels)
        else:
            raise ValueError(
                f"Unsupported number of dimensions for batch normalization: {ndims}")

    elif reg_type == "groupnorm":
        if regularizer_param is None:
            raise ValueError("GroupNorm requires a group count (positive integer)")
        if regularizer_param <= 0 or not float(regularizer_param).is_integer():
            raise ValueError("Group count must be a positive integer")
        if out_channels % int(regularizer_param) != 0:
            raise ValueError(f"out_channels ({out_channels}) must be divisible by group count ({int(regularizer_param)})")

        return nn.GroupNorm(int(regularizer_param), out_channels)

    elif reg_type == "layernorm":
        # For layernorm, regularizer_param is not used.
        return nn.LayerNorm(out_channels)

    elif reg_type == "instancenorm":
        if ndims == 1:
            return nn.InstanceNorm1d(out_channels, affine=True)
        elif ndims == 2:
            return nn.InstanceNorm2d(out_channels, affine=True)
        elif ndims == 3:
            return nn.InstanceNorm3d(out_channels, affine=True)
        else:
            raise ValueError(
                f"Unsupported number of dimensions for instance normalization: {ndims}")

    elif reg_type == "none":
        return nn.Identity()

    else:
        raise ValueError(f"Unsupported regularizer: {regularizer_type}")
