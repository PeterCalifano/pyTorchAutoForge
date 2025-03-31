
# Import modules
from pyTorchAutoForge.model_building.ModelAutoBuilder import ComputeConv2dOutputSize, ComputePooling2dOutputSize, ComputeConvBlockOutputSize

# Test definition of ConvBlock
outChannelsSizes = [16, 32, 75, 15]

# %% Test computation of output size of Conv2d using default settings
patchSize = [7, 7]
convKernelSize = 3
convStrideSize = 1
convPaddingSize = 0
conv2dOutputSize = ComputeConv2dOutputSize(
    patchSize, convKernelSize, convStrideSize, convPaddingSize)


def test_ComputePooling2dOutputSize():
    # %% Test computation of output size of Pooling2d using default settings
    poolingkernelSize = 2
    poolingStrideSize = 1
    poolingOutputSize = ComputePooling2dOutputSize(
        [5, 5], poolingkernelSize, poolingStrideSize)
    
    print('Output size of Conv2d:', conv2dOutputSize)
    print('Output size of Pooling2d:', poolingOutputSize)

    # Add asserts with expected value 
    assert poolingOutputSize == (4, 4), "The output size of Pooling2d is incorrect."



def test_ComputeConvBlockOutputSize():
    # %% Test computation of number of features after ConvBlock using default settings
    convBlockOutputSize = ComputeConvBlockOutputSize(
        [7, 7], outChannelsSizes[0])

    print('Output size of ConvBlock:', convBlockOutputSize)

    # ADD ASSERTS
    assert convBlockOutputSize == [[5, 5]], "The output size of ConvBlock is incorrect."

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

        convBlockOutputSize = ComputeConvBlockOutputSize(outputMapSize, outChannelsSizes[idBlock],
                                                                          convkernelSizes[idBlock], poolingkernelSize[idBlock],
                                                                          convStrideSize[idBlock], poolingStrideSize[idBlock],
                                                                          convPaddingSize[idBlock], poolingPaddingSize[idBlock])
        print(('Output size of ConvBlock ID: {ID}: {outSize}').format(
            ID=idBlock, outSize=convBlockOutputSize
        ))

        # Get size from previous convolutional block
        outputMapSize[0] = convBlockOutputSize[0][0]
        outputMapSize[1] = convBlockOutputSize[0][1]

        