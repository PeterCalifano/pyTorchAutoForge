from enum import Enum
import numpy as np
from sympy import NDimArray
from torch import nn, cat
# Auxiliar functions
from typing import Any
from numpy.typing import NDArray

conv_size_autocomp_input_types = tuple[int,...] | list[int] | NDArray[np.integer]

# %% Auxiliary functions for output size computation
def ComputeConv2dOutputSize(input_size: conv_size_autocomp_input_types, 
                            kernel_size: int = 3,  
                            stride_size: int = 1, 
                            padding_size: int = 0) -> tuple[int, int]:
    """
    Compute the output size and number of feature maps (channels) of a 2D convolutional layer.

    Parameters:
        inputSize (Union[list, np.array, torch.Tensor]): The input size, which must be a list, numpy array, or torch tensor with 2 elements: [height, width].
        kernelSize (int, optional): The size of the convolutional kernel. Default is 3.
        strideSize (int, optional): The stride of the convolution. Default is 1.
        paddingSize (int, optional): The amount of zero-padding added to both sides of the input. Default is 0.

    Returns:
        tuple: A tuple containing the height and width of the output feature map.
    """
    return int(((input_size[0] + 2*padding_size - (kernel_size-1)-1) / stride_size) + 1), int(((input_size[1] + 2*padding_size - (kernel_size-1)-1) / stride_size) + 1)


def ComputePooling2dOutputSize(inputSize: conv_size_autocomp_input_types, 
                               kernelSize: int = 2, 
                               strideSize: int = 2, 
                               paddingSize: int = 0) -> tuple[int, int]:
    """
    Compute the output size and number of feature maps (channels, i.e., volume) of a 2D max/avg pooling layer.

    Parameters:
    inputSize (Union[list, np.array, tensor]): Input size with 2 elements [height, width].
    kernelSize (int, optional): Size of the pooling kernel. Default is 2.
    strideSize (int, optional): Stride size of the pooling operation. Default is 2.
    paddingSize (int, optional): Padding size added to the input. Default is 0.

    Returns:
    tuple: A tuple containing the height and width of the output size.
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
    Computes the output size and number of feature maps (channels, i.e., volume) of a ConvBlock layer.

    Args:
        input_size (Union[list, tuple, np.ndarray, torch.Tensor]): Input size with 2 elements [height, width].
        out_channels_size (int): Number of output channels for the Conv2d layer.
        conv2d_kernel_size (int, optional): Size of the Conv2d kernel. Default is 3.
        pooling_kernel_size (int, optional): Size of the pooling kernel. Default is 2.
        conv_stride_size (int, optional): Stride size for the Conv2d layer. Default is 1.
        pooling_stride_size (int or None, optional): Stride size for the pooling layer. If None, defaults to pooling_kernel_size.
        conv2d_padding_size (int, optional): Padding size for the Conv2d layer. Default is 0.
        pooling_padding_size (int, optional): Padding size for the pooling layer. Default is 0.

    Returns:
        tuple:
            - tuple[int, int]: Output size [height, width] after ConvBlock.
            - int: Flattened output size (height * width * out_channels_size).
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

# TODO function requiring extensive rework


def AutoComputeConvBlocksOutput(self, 
                                kernel_sizes: conv_size_autocomp_input_types, 
                                pooling_kernel_size: conv_size_autocomp_input_types | None = None):
    """
    Automatically compute the output size of a series of ConvBlock layers.

    Args:
        kernelSizes (list): A list of kernel sizes for each convolutional layer.
        poolingKernelSize (list, optional): A list of pooling kernel sizes for each convolutional layer. 
                                            If None, defaults to a list of ones with the same length as kernelSizes.

    Returns:
        list: The output size of the last ConvBlock layer in the format [height, width].
    """
    # NOTE: stride and padding are HARDCODED in this version
    outputMapSize = [self.patchSize, self.patchSize]

    if pooling_kernel_size is None:
        pooling_kernel_size = list(np.ones(len(kernel_sizes)))

    assert (self.numOfConvLayers == len(
            kernel_sizes) == len(pooling_kernel_size))

    for idL in range(self.numOfConvLayers):

        convBlockOutputSize = ComputeConvBlock2dOutputSize(input_size=outputMapSize,
                                                            out_channels_size=self.outChannelsSizes[idL], conv2d_kernel_size=kernel_sizes[idL], 
                                                            pooling_kernel_size=pooling_kernel_size[idL],
                                                            conv_stride_size=1, 
                                                            pooling_stride_size=pooling_kernel_size[idL],
                                                            conv2d_padding_size=0, 
                                                            pooling_padding_size=0)

        print(('Output size of ConvBlock ID: {ID}: {outSize}').format(
            ID=idL, outSize=convBlockOutputSize))
        
        # Get size from previous convolutional block
        outputMapSize[0] = convBlockOutputSize[0][0]
        outputMapSize[1] = convBlockOutputSize[0][1]

    return convBlockOutputSize

# %% MultiHeadRegressor class implementation
class EnumMultiHeadOutMode(Enum):
    Concatenate = 0
    Append = 1
    Sum = 2  # TODO not implemented yet
    Average = 3  # TODO not implemented yet


# TODO (PC) move to modelBuildingBlocks module (or maybe a new one, it is already quite large)
class MultiHeadRegressor(nn.Module):
    def __init__(self, model_heads: nn.ModuleList | nn.ModuleDict | nn.Module,
                 output_mode: EnumMultiHeadOutMode = EnumMultiHeadOutMode.Concatenate,
                 *args,
                 **kwargs):

        # Initialize nn.Module base class
        super(MultiHeadRegressor, self).__init__()
        self.heads = nn.ModuleList()
        self.output_mode = output_mode

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

        # Define function to pack output depending on the output_mode
        if self.output_mode == EnumMultiHeadOutMode.Concatenate:
            self.pack_output = self._pack_output_concat

        elif self.output_mode == EnumMultiHeadOutMode.Append:
            self.pack_output = self._pack_output_append

        else:
            raise NotImplementedError(
                f"Output mode {self.output_mode} not implemented yet >.<")


    def _pack_output_append(self, predictions: list):
        """
        Packs the predictions into a list. Used when the output mode is set to Append.
        """
        return predictions
   
    def _pack_output_concat(self, predictions: list):
        """
        Packs the predictions into a single tensor by concatenating them along the second dimension.
        Used when the output mode is set to Concatenate.
        """
        # Concatenate along 2nd dimension
        return cat(tensors=predictions, dim=1)

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
