
# Import modules
import pyTorchAutoForge

# Test definition of ConvBlock
outChannelsSizes = [16, 32, 75, 15]

# %% Test computation of output size of Conv2d using default settings
patchSize = [7, 7]
convKernelSize = 3
convStrideSize = 1
convPaddingSize = 0
conv2dOutputSize = pyTorchAutoForge.ComputeConv2dOutputSize(
    patchSize, convKernelSize, convStrideSize, convPaddingSize)


def test_ComputePooling2dOutputSize():
    # %% Test computation of output size of Pooling2d using default settings
    poolingkernelSize = 2
    poolingStrideSize = 1
    poolingOutputSize = pyTorchAutoForge.ComputePooling2dOutputSize(
        [5, 5], poolingkernelSize, poolingStrideSize)
    
    print('Output size of Conv2d:', conv2dOutputSize)
    print('Output size of Pooling2d:', poolingOutputSize)

    # ADD ASSERTS
    
def test_ComputeConvBlockOutputSize():
    # %% Test computation of number of features after ConvBlock using default settings
    convBlockOutputSize = pyTorchAutoForge.ComputeConvBlockOutputSize(
        [7, 7], outChannelsSizes[0])

    print('Output size of ConvBlock:', convBlockOutputSize)

    outputMapSize = [7, 7]

    convkernelSizes = [3, 3]
    convStrideSize = [1, 1]
    convPaddingSize = [0, 0]
    poolingkernelSize = [2, 2]
    poolingStrideSize = [1, 1]
    poolingPaddingSize = [0, 0]

    print('\n')
    # Test recursive computation for all defined ConvBlocks (PASSED)
    for idBlock in range(2):

        convBlockOutputSize = pyTorchAutoForge.ComputeConvBlockOutputSize(outputMapSize, outChannelsSizes[idBlock],
                                                                          convkernelSizes[idBlock], poolingkernelSize[idBlock],
                                                                          convStrideSize[idBlock], poolingStrideSize[idBlock],
                                                                          convPaddingSize[idBlock], poolingPaddingSize[idBlock])
        print(('Output size of ConvBlock ID: {ID}: {outSize}').format(
            ID=idBlock, outSize=convBlockOutputSize))

        # Get size from previous convolutional block
        outputMapSize[0] = convBlockOutputSize[0][0]
        outputMapSize[1] = convBlockOutputSize[0][1]

        # ADD ASSERTS