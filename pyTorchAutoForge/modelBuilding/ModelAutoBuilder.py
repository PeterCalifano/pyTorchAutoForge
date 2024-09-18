from typing import Union    
import numpy as np
import torch

# Auxiliar functions
def ComputeConv2dOutputSize(inputSize: Union[list, np.array, torch.tensor], kernelSize=3, strideSize=1, paddingSize=0):
    '''Compute output size and number of features maps (channels, i.e. volume) of a 2D convolutional layer. 
       Input size must be a list, numpy array or a torch tensor with 2 elements: [height, width].'''
    return int((inputSize[0] + 2*paddingSize - (kernelSize-1)-1) / strideSize + 1), int((inputSize[1] + 2*paddingSize - (kernelSize-1)-1) / strideSize + 1)


def ComputePooling2dOutputSize(inputSize: Union[list, np.array, torch.tensor], kernelSize=2, strideSize=2, paddingSize=0):
    '''Compute output size and number of features maps (channels, i.e. volume) of a 2D max/avg pooling layer. 
       Input size must be a list, numpy array or a torch tensor with 2 elements: [height, width].'''
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


# %% ModelAutoBuilder class implementation
class ModelAutoBuilder():
    def __init__(self):
        raise NotImplementedError('ModelAutoBuilder class is not implemented yet')
        pass
