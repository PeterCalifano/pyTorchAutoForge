from dataclasses import dataclass
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
        self.activ = _activation_factory(
            activ_type, out_channels, prelu_params)

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
        self.activ = _activation_factory(
            activ_type, out_channels, prelu_params)

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


# %% Feature map fusion blocks
# EXPERIMENTAL
fuser_type = Literal["feature_add", "channel_concat", "multihead_attention"]

@dataclass
class FeatureMapFuserConfig():
    """
    Configuration for a feature-map fusion operation.

    Attributes:
        fuser_type: 'feature_add', 'channel_concat', or 'multihead_attention'
        in_channels: number of channels in x
        num_skip_channels: channels in skip tensor (required for 'concat')
        num_attention_heads: number of heads for attention (only for 'attn')
    """
    in_channels: int
    num_skip_channels: int
    num_dims : int = 4
    fuser_type: fuser_type = "channel_concat"
    num_attention_heads: int | None = None


class FeatureMapFuser(nn.Module):
    """
    FeatureMapFuser _summary_

    _extended_summary_

    :param nn: _description_
    :type nn: _type_
    """

    def __init__(self,
                 num_dims: int,  # Number of dims of the input tensor
                 fuser_type: fuser_type,
                 in_channels: int,
                 num_skip_channels: int | None = None,
                 num_attention_heads: int | None = None,
                 **kwargs):

        super().__init__()
        self.fuser_type = fuser_type

        # Build the actual operator and assign it to self.fuse
        if fuser_type == 'feature_add':
            import torch.nn.functional as F

            # Preselect the appropriate fusion function at initialization so that the returned
            # function contains no conditionals during forward.
            num_dims = kwargs.get("ndims", 4)

            if num_dims == 2:  # 1D inputs: [B, L]
                mode = kwargs.get("mode", "linear")

                def _feature_add_fuser(x: torch.Tensor, skip: torch.Tensor):
                    out: torch.Tensor = x + \
                        F.interpolate(
                            skip, size=x.shape[1:], mode=mode, align_corners=True, antialias=False)
                    return out

            elif num_dims == 4:  # 2D inputs: [B,C,H,W]
                mode = kwargs.get("mode", "bicubic")

                def _feature_add_fuser(x: torch.Tensor, skip: torch.Tensor):
                    out: torch.Tensor = x + \
                        F.interpolate(
                            skip, size=x.shape[1:], mode=mode, align_corners=True, antialias=False)
                    return out

            elif num_dims == 5:  # 3D inputs: [B,C,D,H,W]
                mode = kwargs.get("mode", "trilinear")

                def _feature_add_fuser(x: torch.Tensor, skip: torch.Tensor):
                    out: torch.Tensor = x + \
                        F.interpolate(
                            skip, size=x.shape[2:], mode=mode, align_corners=True, antialias=False)
                    return out

            else:
                raise ValueError(
                    "Invalid input dimensions. Expected a 4D tensor (for 2D inputs) or a 5D tensor (for 3D inputs).")

            self.fuser = _feature_add_fuser

        elif fuser_type == 'channel_concat':
            assert num_skip_channels is not None, "num_skip_channels arg is required for concat"

            if num_dims == 4:
                self.proj: nn.Conv2d | nn.Conv3d = nn.Conv2d(in_channels=in_channels + num_skip_channels,
                                                             out_channels=in_channels,
                                                             kernel_size=1)
            elif num_dims == 5:
                self.proj = nn.Conv3d(in_channels=in_channels + num_skip_channels,
                                      out_channels=in_channels,
                                      kernel_size=1)
            else:
                raise ValueError(
                    f"Invalid num_dims for channel_concat fuser. Expected 4 (2D) or 5 (3D), found {num_dims}.")

            def _channel_concat_fuser(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
                """
                Concatenate the input tensor and skip tensor along the channel dimension,
                then project the result to the original number of channels.
                """
                # Concatenate along channel dimension
                concatenated: torch.Tensor = torch.cat([x, skip], dim=1)
                # Project back to original number of channels
                out: torch.Tensor = self.proj(concatenated)
                return out

            self.fuser = _channel_concat_fuser

        elif fuser_type == 'multihead_attention':
            assert num_attention_heads is not None, "num_attention_heads arg is required for attention fuser"

            # Retrieve additional arguments for multi-head attention from kwargs
            dropout = kwargs.get("dropout", 0.0)
            bias = kwargs.get("bias", True)
            add_bias_kv = kwargs.get("add_bias_kv", False)
            add_zero_attn = kwargs.get("add_zero_attn", False)
            kdim = kwargs.get("kdim", in_channels)
            vdim = kwargs.get("vdim", in_channels)

            # Define multi-head attention
            self._attention_fuser = nn.MultiheadAttention(embed_dim=in_channels,
                                                          num_heads=num_attention_heads,
                                                          dropout=dropout,
                                                          bias=bias,
                                                          add_bias_kv=add_bias_kv,
                                                          add_zero_attn=add_zero_attn,
                                                          kdim=kdim,
                                                          vdim=vdim,
                                                          batch_first=False)

            def _attn_fuse(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:

                b, c, h, w = x.shape

                # Flatten to [sequence_len, batch, embed]
                x_flat = x.flatten(2).permute(2, 0, 1)
                skip_flat = skip.flatten(2).permute(2, 0, 1)
                fused_flat, _ = self._attention_fuser(
                    x_flat, skip_flat, skip_flat)

                # Reshape back to [batch, channel, h, w]
                fused_flat: torch.Tensor = fused_flat.permute(
                    1, 2, 0).reshape(b, c, h, w)

                return fused_flat

            self.fuser = _attn_fuse

        else:
            raise ValueError(f"Unknown fuse mode: {fuser_type}")

    def forward(self, x: torch.Tensor, skip_features: torch.Tensor) -> torch.Tensor:
        # Call fuser module
        fused_x: torch.Tensor = self.fuser(x, skip_features)
        return fused_x


# FeatureMapFuser factory
def _feature_map_fuser_factory(config: FeatureMapFuserConfig,
                               **kwargs
                               ) -> FeatureMapFuser:
    """
    Factory function to create a FeatureMapFuser instance based on the provided configuration.

    This factory reads from a FeatureMapFuserConfig instance and instantiates a
    FeatureMapFuser that fuses skip connections with the main input by using one of three methods:
    feature addition, channel concatenation, or multi-head attention. The choice of fuser is determined by
    config.fuser_type and the provided num_dims parameter. This abstraction allows customization of
    the fusion operation while supporting additional mode arguments and multi-head attention parameters.

    Args:
        config (FeatureMapFuserConfig): 
            Configuration for the feature map fuser. Must contain the following attributes:
                - in_channels: Number of input channels.
                - num_skip_channels: Number of channels in the skip tensor (required for channel concatenation).
                - fuser_type: The type of fusion operation ("feature_add", "channel_concat", or "multihead_attention").
                - num_attention_heads: Number of attention heads (only required for multi-head attention).
        num_dims (int): 
            The dimensionality of the input tensor. Expected values:
                - 2 for 1D inputs,
                - 4 for 2D inputs (e.g., images),
                - 5 for 3D inputs (e.g., volumetric data).
            Default is 4.
        **kwargs: 
            Additional keyword arguments for the fuser. For example:
                - mode: Upsampling mode for 'feature_add' (e.g., "linear", "bicubic", or "trilinear").
                - dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim: Advanced settings for multi-head attention.

    Returns:
        FeatureMapFuser: A configured feature map fuser instance that fuses an input tensor with skip features
        according to the specified fusion method.
    """
    return FeatureMapFuser(num_dims=config.num_dims,
                           fuser_type=config.fuser_type,
                           in_channels=config.in_channels,
                           num_skip_channels=config.num_skip_channels,
                           num_attention_heads=config.num_attention_heads,
                           **kwargs)
