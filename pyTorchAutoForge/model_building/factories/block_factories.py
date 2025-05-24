import torch.nn as nn
from typing import Literal

# TODO add other initialization methods
# %% Layer initialization methods
def _initialize_convblock_weights(block,
                                  init_method_type: Literal["xavier_uniform",
                                                            "kaiming_uniform",
                                                            "xavier_normal",
                                                            "kaiming_normal",
                                                            "orthogonal"] = "xavier_uniform"):
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
                                 init_method_type: Literal["xavier_uniform",
                                                           "kaiming_uniform",
                                                           "xavier_normal",
                                                           "kaiming_normal",
                                                           "orthogonal"] = "xavier_uniform"):
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
def _activation_factory(activ_type, out_channels, prelu_params):
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
def _pooling_factory(pool_type, 
                     kernel_size, 
                     stride=None, 
                     padding=0, 
                     target_res=None) -> nn.MaxPool1d | nn.AvgPool1d | nn.MaxPool2d | nn.AvgPool2d | nn.MaxPool3d | nn.AvgPool3d | nn.AdaptiveMaxPool1d | nn.AdaptiveAvgPool1d | nn.AdaptiveMaxPool2d | nn.AdaptiveAvgPool2d | nn.AdaptiveMaxPool3d | nn.AdaptiveAvgPool3d | nn.Identity:
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
        kernel_size (int or tuple): Kernel size for the pooling layer.
            Not used for adaptive pooling.
        stride (int or tuple, optional): Stride for the pooling layer. Defaults to 1 if not provided.
        padding (int or tuple, optional): Padding for the pooling layer. Defaults to 0.
        target_res (int, tuple, optional): Target output size for adaptive pooling. Required if pool_type is adaptive.

    Returns:
        nn.Module: The corresponding pooling layer.
    
    Raises:
        ValueError: If an unsupported pooling type is provided or target_res is required but not provided.
    """
    pool_type_lower = pool_type.lower()
    default_stride = stride if stride is not None else 1

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
                raise ValueError(
                    f"Unsupported adaptive pooling type: {pool_type}")
        case "none":
            return nn.Identity()
        case _:
            raise ValueError(f"Unsupported pooling type: {pool_type}")
    
def _regularizer_factory():
    pass