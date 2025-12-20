from enum import Enum
import numpy as np
from torch import nn, cat
from numpy.typing import NDArray

conv_size_autocomp_input_types = tuple[int,...] | list[int] | NDArray[np.integer]

def Infer_head_first_layer_size(head: nn.Module) -> int:
    """
    Infer the input size of the head's first learnable layer by scanning for the first
    module that exposes `in_features` or `in_channels`.
    """

    for module in head.modules():
        if module is head:
            continue
        if hasattr(module, "in_features"):
            return int(getattr(module, "in_features"))  # type: ignore[arg-type]
        if hasattr(module, "in_channels"):
            return int(getattr(module, "in_channels"))  # type: ignore[arg-type]
        
    raise ValueError("Unable to infer head first layer input size; no layer with in_features/in_channels found.")

# %% Auxiliary functions for output size computation
def ComputeConv2dOutputSize(input_size: conv_size_autocomp_input_types, 
                            kernel_size: int = 3,  
                            stride_size: int = 1, 
                            padding_size: int = 0) -> tuple[int, int]:
    """
    Compute output height and width of a 2D convolutional layer.

    Args:
        input_size: Sequence (height, width) of the input tensor.
        kernel_size: Size of the convolution kernel. Defaults to 3.
        stride_size: Stride of the convolution. Defaults to 1.
        padding_size: Zero-padding added to both sides. Defaults to 0.

    Returns:
        A tuple (out_height, out_width) with the computed spatial dimensions.
    """
    return int(((input_size[0] + 2*padding_size - (kernel_size-1)-1) / stride_size) + 1), int(((input_size[1] + 2*padding_size - (kernel_size-1)-1) / stride_size) + 1)

def ComputePooling2dOutputSize(inputSize: conv_size_autocomp_input_types, 
                               kernelSize: int = 2, 
                               strideSize: int = 2, 
                               paddingSize: int = 0) -> tuple[int, int]:
    """
    Compute output height and width of a 2D pooling layer (max/avg).

    Args:
        inputSize: Sequence (height, width) of the input tensor.
        kernelSize: Size of the pooling kernel. Defaults to 2.
        strideSize: Stride of the pooling. Defaults to 2.
        paddingSize: Padding added to the input. Defaults to 0.

    Returns:
        A tuple (out_height, out_width) with the computed spatial dimensions.
    """
    return int(((inputSize[0] + 2*paddingSize - (kernelSize-1)-1) / strideSize) + 1), int(((inputSize[1] + 2*paddingSize - (kernelSize-1)-1) / strideSize) + 1)

# ConvBlock 2D and flatten sizes computation (SINGLE BLOCK)
def ComputeConvBlock2dOutputSize(input_size: conv_size_autocomp_input_types, 
                               out_channels_size: int,
                               conv2d_kernel_size: int = 3, 
                               pooling_kernel_size: int = 2,
                               conv_stride_size: int = 1, 
                               pooling_stride_size: int | None = None,
                               conv2d_padding_size: int = 0, 
                               pooling_padding_size: int = 0) -> tuple[tuple[int, int], int]:
    """
    Compute the spatial output size and flattened feature count for a single ConvBlock (Conv2d -> Pool).

    Args:
        input_size: Sequence (height, width) of the input to the ConvBlock.
        out_channels_size: Number of output channels produced by the Conv2d layer.
        conv2d_kernel_size: Conv2d kernel size. Defaults to 3.
        pooling_kernel_size: Pooling kernel size. Defaults to 2.
        conv_stride_size: Conv2d stride. Defaults to 1.
        pooling_stride_size: Pooling stride. If None, defaults to pooling_kernel_size.
        conv2d_padding_size: Conv2d padding. Defaults to 0.
        pooling_padding_size: Pooling padding. Defaults to 0.

    Returns:
        A tuple:
            - (out_height, out_width): Spatial size after ConvBlock.
            - flattened_size: Number of features after flattening (height * width * out_channels_size).

    Raises:
        ValueError: If pooling kernel is larger than the Conv2d output spatial dimensions.
    """
    if pooling_stride_size is None:
        pooling_stride_size = pooling_kernel_size

    # Compute output size of Conv2d and Pooling2d layers
    conv2d_outsize = ComputeConv2dOutputSize(input_size,
          conv2d_kernel_size, 
          conv_stride_size, 
          conv2d_padding_size)

    if conv2d_outsize[0] < pooling_kernel_size or conv2d_outsize[1] < pooling_kernel_size:
        raise ValueError('Pooling kernel size is larger than output size of Conv2d layer. Please check configuration validity.')

    conv_block_output_size = ComputePooling2dOutputSize(conv2d_outsize,
        pooling_kernel_size,
          pooling_stride_size,
            pooling_padding_size)

    # Compute total number of features after ConvBlock as required for the fully connected layers
    conv2d_flattened_output_size = conv_block_output_size[0] * \
        conv_block_output_size[1] * out_channels_size

    return conv_block_output_size, conv2d_flattened_output_size

### AutoComputeConvBlocksOutput
def AutoComputeConvBlocksOutput(first_input_size: int | list[int] | tuple[int, int], 
                                out_channels_sizes: conv_size_autocomp_input_types,
                                kernel_sizes: conv_size_autocomp_input_types, 
                                pooling_kernel_sizes: conv_size_autocomp_input_types | None = None,
                                conv_stride_sizes: conv_size_autocomp_input_types | None = None,
                                pooling_stride_sizes: conv_size_autocomp_input_types | None = None,
                                conv2d_padding_sizes: conv_size_autocomp_input_types | None = None,
                                pooling_padding_sizes: conv_size_autocomp_input_types | None = None) -> tuple[tuple[int, int], list[int], list[tuple[int,int]]]:
    """
    Compute outputs for a sequence of ConvBlock layers.

    Args:
        first_input_size: Initial input size. If int, treated as square (height==width). If tuple/list, expects (height, width).
        out_channels_sizes: Sequence of output channel counts for each ConvBlock.
        kernel_sizes: Sequence of conv kernel sizes for each ConvBlock.
        pooling_kernel_sizes: Sequence of pooling kernel sizes. If None, defaults to 1 for each block.
        conv_stride_sizes: Sequence of conv strides. If None, defaults to 1 for each block.
        pooling_stride_sizes: Sequence of pooling strides. If None, defaults to pooling_kernel_sizes.
        conv2d_padding_sizes: Sequence of conv paddings. If None, defaults to 0 for each block.
        pooling_padding_sizes: Sequence of pooling paddings. If None, defaults to 0 for each block.

    Returns:
        A tuple:
            - last_map_size: (height, width) after the final ConvBlock.
            - flattened_sizes: List of flattened feature sizes for each ConvBlock.
            - intermediated_maps_sizes: List of (height, width) for each ConvBlock.
    """
    if isinstance(first_input_size, int):
        first_input_size = [first_input_size, first_input_size]
    elif isinstance(first_input_size, tuple):
        first_input_size = list(first_input_size)
    else:
        raise TypeError("Invalid input size format.")

    # Handle None defaults
    if pooling_kernel_sizes is None:
        pooling_kernel_sizes = list(np.ones(len(kernel_sizes)))

    if conv_stride_sizes is None:
        conv_stride_sizes = list(np.ones(len(kernel_sizes)))

    if pooling_stride_sizes is None:
        pooling_stride_sizes = pooling_kernel_sizes.copy() if isinstance(pooling_kernel_sizes, list) else list(pooling_kernel_sizes)

    if conv2d_padding_sizes is None:
        conv2d_padding_sizes = list(np.zeros(len(kernel_sizes)))

    if pooling_padding_sizes is None:
        pooling_padding_sizes = list(np.zeros(len(kernel_sizes)))

    # Loop over input lists 
    flattened_sizes = []
    intermediated_maps_sizes = []
    # Initialize to the current input size so that an empty kernel_sizes yields a meaningful output
    conv_block_map_output_size = (first_input_size[0], first_input_size[1])

    for idL in range(len(kernel_sizes)):

        conv_block_map_output_size, flattened_feats = ComputeConvBlock2dOutputSize(input_size=first_input_size,
            out_channels_size=out_channels_sizes[idL], 
            conv2d_kernel_size=kernel_sizes[idL], 
            pooling_kernel_size=pooling_kernel_sizes[idL],
            conv_stride_size=conv_stride_sizes[idL], 
            pooling_stride_size=pooling_stride_sizes[idL],
            conv2d_padding_size=conv2d_padding_sizes[idL], 
            pooling_padding_size=pooling_padding_sizes[idL]
        )

        print((f'Output size of ConvBlock ID: {idL}: {conv_block_map_output_size}. Output channels: {out_channels_sizes[idL]}, flattened features size: {flattened_feats}'))

        # Get size from previous convolutional block
        first_input_size[0] = conv_block_map_output_size[0]
        first_input_size[1] = conv_block_map_output_size[1]

        # Compute intermediate sizes and flattened sizes
        intermediated_maps_sizes.append(conv_block_map_output_size)
        flattened_sizes.append(flattened_feats)

    return conv_block_map_output_size, flattened_sizes, intermediated_maps_sizes

# %% MultiHeadRegressor class implementation
class EnumMultiHeadOutMode(Enum):
    Concatenate = 0
    Append = 1
    Sum = 2  # TODO not implemented yet
    Average = 3  # TODO not implemented yet

# TODO (PC) move to modelBuildingBlocks module (or maybe a new one, it is already quite large)
class HeadRegressor(nn.Module):
    """
    Base class for regression heads that manages a collection of sub-head modules and output packing.

    Args:
        model_heads: One of nn.Module, nn.ModuleList or nn.ModuleDict containing head modules.
        output_mode: EnumMultiHeadOutMode value controlling how multiple heads' outputs are combined.

    Attributes:
        output_mode: Selected output packing mode.
        heads: nn.ModuleList containing the provided head modules.
        pack_output: Callable used to pack outputs according to output_mode.

    Raises:
        TypeError: If model_heads is not nn.Module, nn.ModuleList, or nn.ModuleDict.
        NotImplementedError: If an unsupported output_mode is provided.
    """

    def __init__(self, model_heads: nn.ModuleList | nn.ModuleDict | nn.Module,
                 output_mode: EnumMultiHeadOutMode = EnumMultiHeadOutMode.Concatenate):
        
        super(HeadRegressor, self).__init__()

        self.output_mode = output_mode
        self.heads = nn.ModuleList()

        # Define function to pack output depending on the output_mode
        if self.output_mode == EnumMultiHeadOutMode.Concatenate:
            self.pack_output = self._pack_output_concat

        elif self.output_mode == EnumMultiHeadOutMode.Append:
            self.pack_output = self._pack_output_append

        else:
            raise NotImplementedError(
                f"Output mode {self.output_mode} not implemented yet >.<")

        if isinstance(model_heads, nn.ModuleList):
            # Unpack list and append to heads module List
            for module in model_heads:
                self.heads.append(module)

        elif isinstance(model_heads, nn.ModuleDict):

            # Unpack dictionary and append to heads module List
            for key, module in model_heads.items():
                self.heads.append(module)

        elif isinstance(model_heads, nn.Module):
            self.heads.append(model_heads)

        else:
            raise TypeError("model_heads must be nn.ModuleList, nn.ModuleDict or nn.Module")

    # Methods
    def _pack_output_append(self, predictions: list):
        """
        Return list of predictions (append mode).

        Args:
            predictions: List of tensors produced by each head.

        Returns:
            The input list of predictions.
        """
        return predictions

    def _pack_output_concat(self, predictions: list):
        """
        Concatenate predictions along channel dimension (concat mode).

        Args:
            predictions: List of tensors produced by each head.

        Returns:
            A single tensor obtained by concatenating predictions along dim=1.
        """
        # Concatenate along 2nd dimension
        return cat(tensors=predictions, dim=1)
    

class CascadedHeadRegressor(HeadRegressor):
    def __init__(self, model_heads: nn.ModuleList | nn.ModuleDict | nn.Module,
                 output_mode: EnumMultiHeadOutMode = EnumMultiHeadOutMode.Concatenate,
                 concat_dim: int = 1,
                 *args,
                 **kwargs):
        
        # Initialize nn.Module base class
        super().__init__(model_heads=model_heads, output_mode=output_mode)
        self.concat_dim = concat_dim

    def forward(self, X):

        # Perform forward pass for each head and append to list
        predictions = []  # TODO this should be initializer statically based on output specifications

        for head in self.heads:
            predictions.append(head(X))

            # Concatenate input with prediction for next head
            X = cat(tensors=(X, predictions[-1]), dim=self.concat_dim)

        return self.pack_output(predictions)
    
    
class MultiHeadRegressor(HeadRegressor):
    def __init__(self, model_heads: nn.ModuleList | nn.ModuleDict | nn.Module,
                 output_mode: EnumMultiHeadOutMode = EnumMultiHeadOutMode.Concatenate,
                 *args,
                 **kwargs):

        # Initialize nn.Module base class
        super().__init__(model_heads=model_heads, output_mode=output_mode)

    def forward(self, X):

        # Perform forward pass for each head and append to list
        predictions = []  # TODO this should be initializer statically based on output specifications

        for head in self.heads:
            predictions.append(head(X))

        return self.pack_output(predictions)


# %% ModelAutoBuilder class implementation (# DEVNOTE TBD, old idea, not sure it was a good one)
class ModelAutoBuilder():
    def __init__(self, cfg):
        self.cfg = cfg

    def build(self):
        pass  # TODO
