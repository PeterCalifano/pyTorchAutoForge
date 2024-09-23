from typing import Union
import numpy as np
import torch

# Auxiliar functions


def ComputeConv2dOutputSize(inputSize: Union[list, np.array, torch.tensor], kernelSize=3, strideSize=1, paddingSize=0):
    """
    Compute the output size and number of feature maps (channels) of a 2D convolutional layer.

    Parameters:
        inputSize (Union[list, np.array, torch.tensor]): The input size, which must be a list, numpy array, or torch tensor with 2 elements: [height, width].
        kernelSize (int, optional): The size of the convolutional kernel. Default is 3.
        strideSize (int, optional): The stride of the convolution. Default is 1.
        paddingSize (int, optional): The amount of zero-padding added to both sides of the input. Default is 0.

    Returns:
        tuple: A tuple containing the height and width of the output feature map.
    """
    return int((inputSize[0] + 2*paddingSize - (kernelSize-1)-1) / strideSize + 1), int((inputSize[1] + 2*paddingSize - (kernelSize-1)-1) / strideSize + 1)


def ComputePooling2dOutputSize(inputSize: Union[list, np.array, torch.tensor], kernelSize=2, strideSize=2, paddingSize=0):
    """
    Compute the output size and number of feature maps (channels, i.e., volume) of a 2D max/avg pooling layer.

    Parameters:
    inputSize (Union[list, np.array, torch.tensor]): Input size with 2 elements [height, width].
    kernelSize (int, optional): Size of the pooling kernel. Default is 2.
    strideSize (int, optional): Stride size of the pooling operation. Default is 2.
    paddingSize (int, optional): Padding size added to the input. Default is 0.

    Returns:
    tuple: A tuple containing the height and width of the output size.
    """
    return int(((inputSize[0] + 2*paddingSize - (kernelSize-1)-1) / strideSize) + 1), int(((inputSize[1] + 2*paddingSize - (kernelSize-1)-1) / strideSize) + 1)

# ConvBlock 2D and flatten sizes computation (SINGLE BLOCK)


def ComputeConvBlockOutputSize(inputSize: Union[list, np.array, torch.tensor], outChannelsSize: int,
                               convKernelSize: int = 3, poolingkernelSize: int = 2,
                               convStrideSize: int = 1, poolingStrideSize: int = None,
                               convPaddingSize: int = 0, poolingPaddingSize: int = 0):
        

    # TODO: modify interface to use something like a dictionary with the parameters, to make it more fexible and avoid the need to pass all the parameters
    '''Compute output size and number of features maps (channels, i.e. volume) of a ConvBlock layer.
       Input size must be a list, numpy array or a torch tensor with 2 elements: [height, width].'''

    if poolingStrideSize is None:
        poolingStrideSize = poolingkernelSize

    # Compute output size of Conv2d and Pooling2d layers
    conv2dOutputSize = ComputeConv2dOutputSize(
        inputSize, convKernelSize, convStrideSize, convPaddingSize)

    if conv2dOutputSize[0] < poolingkernelSize or conv2dOutputSize[1] < poolingkernelSize:
        raise ValueError(
            'Pooling kernel size is larger than output size of Conv2d layer. Check configuration.')

    convBlockOutputSize = ComputePooling2dOutputSize(
        conv2dOutputSize, poolingkernelSize, poolingStrideSize, poolingPaddingSize)

    # Compute total number of features after ConvBlock as required for the fully connected layers
    conv2dFlattenOutputSize = convBlockOutputSize[0] * \
        convBlockOutputSize[1] * outChannelsSize

    return convBlockOutputSize, conv2dFlattenOutputSize


def AutoComputeConvBlocksOutput(self, kernelSizes: list, poolingKernelSize: list = None):
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

    if poolingKernelSize is None:
        poolingKernelSize = list(np.ones(len(kernelSizes)))

    assert (self.numOfConvLayers == len(
            kernelSizes) == len(poolingKernelSize))

    for idL in range(self.numOfConvLayers):

        convBlockOutputSize = ComputeConvBlockOutputSize(outputMapSize, self.outChannelsSizes[idL], kernelSizes[idL], poolingKernelSize[idL],
                                                                              convStrideSize=1, poolingStrideSize=poolingKernelSize[idL],
                                                                              convPaddingSize=0, poolingPaddingSize=0)

        print(('Output size of ConvBlock ID: {ID}: {outSize}').format(
            ID=idL, outSize=convBlockOutputSize))
        # Get size from previous convolutional block
        outputMapSize[0] = convBlockOutputSize[0][0]
        outputMapSize[1] = convBlockOutputSize[0][1]

    return convBlockOutputSize


# %% ModelAutoBuilder class implementation
class ModelAutoBuilder():
    def __init__(self):
        raise NotImplementedError('ModelAutoBuilder class is not implemented yet')
        pass
