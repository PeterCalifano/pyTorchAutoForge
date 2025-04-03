# Import modules
from pyTorchAutoForge.model_building.ModelAutoBuilder import ComputeConv2dOutputSize, ComputePooling2dOutputSize, ComputeConvBlockOutputSize

# Test definition of ConvBlock
outChannelsSizes = [16, 32, 75, 15]

# %% Test computation of output size of Conv2d using default settings
def test_ComputeConv2dOutputSize():
    # %% Test computation of output size of Conv2d using default settings
    patchSize = [7, 7]
    convKernelSize = 3
    convStrideSize = 1
    convPaddingSize = 0
    conv2dOutputSize = ComputeConv2dOutputSize(
        patchSize, convKernelSize, convStrideSize, convPaddingSize)

    print('Output size of Conv2d:', conv2dOutputSize)

    # Add asserts with expected value
    assert conv2dOutputSize == (5, 5), "The output size of ComputeConv2dOutputSize is incorrect."

def test_ComputePooling2dOutputSize():
    # %% Test computation of output size of Pooling2d using default settings
    poolingkernelSize = 2
    poolingStrideSize = 1
    poolingOutputSize = ComputePooling2dOutputSize(
        [5, 5], poolingkernelSize, poolingStrideSize)
    
    print('Output size of Pooling2d:', poolingOutputSize)

    # Add asserts with expected value 
    assert poolingOutputSize == (4, 4), "The output size of ComputePooling2dOutputSize is incorrect."

def test_ComputeConvBlockOutputSize_singleBlock():
    # %% Test computation of number of features after ConvBlock using default settings
    convBlockOutputSize, flattenedOutput = ComputeConvBlockOutputSize(
        [7, 7], outChannelsSizes[0])

    assert convBlockOutputSize == (2, 2), "The 2d output size of ConvBlock (conv. + pooling) is incorrect."
    assert flattenedOutput == 1 * convBlockOutputSize[0] * convBlockOutputSize[1] * outChannelsSizes[0], "The flattened output size of ConvBlock is incorrect."


def test_ComputeConvBlockOutputSize_multiBlock():
    outputMapSize = [7, 7]
    convkernelSizes = [3, 3]
    convStrideSize = [1, 1]
    convPaddingSize = [0, 0]
    poolingkernelSize = [2, 2]
    poolingStrideSize = [1, 1]
    poolingPaddingSize = [0, 0]
    
    for idBlock in range(2):
        convBlockOutputSize, flattenedOutput = ComputeConvBlockOutputSize(
            outputMapSize,
            outChannelsSizes[idBlock],
            convkernelSizes[idBlock],
            poolingkernelSize[idBlock],
            convStrideSize[idBlock],
            poolingStrideSize[idBlock],
            convPaddingSize[idBlock],
            poolingPaddingSize[idBlock]
        )

        print(f'Output size of ConvBlock ID: {idBlock}: {convBlockOutputSize}')

        # Add asserts for each block
        if idBlock == 0:
            assert convBlockOutputSize == (4, 4), "First ConvBlock output size is incorrect."
            assert flattenedOutput == outChannelsSizes[idBlock] * 4 * 4, "First ConvBlock flattened output is incorrect."
        elif idBlock == 1:
            assert convBlockOutputSize == (1, 1), "Second ConvBlock output size is incorrect."
            assert flattenedOutput == outChannelsSizes[idBlock] * 1 * 1, "Second ConvBlock flattened output is incorrect."

        outputMapSize[0] = convBlockOutputSize[0]
        outputMapSize[1] = convBlockOutputSize[1]

def test_AutoComputeConvBlocksOutput():
    print("Warning: This test is incomplete. Please implement the test logic here.")
    pass