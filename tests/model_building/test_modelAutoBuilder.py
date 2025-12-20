import pytest
import torch.nn as nn

from pyTorchAutoForge.model_building.ModelAutoBuilder import (
    Infer_head_first_layer_size,
    ComputeConv2dOutputSize,
    ComputePooling2dOutputSize,
    ComputeConvBlock2dOutputSize,
    AutoComputeConvBlocksOutput,
)


def test_infer_head_first_layer_size_linear_chain():
    head = nn.Sequential(
        nn.ReLU(),
        nn.Linear(5, 2),
        nn.Linear(2, 1),
    )

    assert Infer_head_first_layer_size(head) == 5


def test_infer_head_first_layer_size_conv_branch():
    head = nn.Sequential(
        nn.BatchNorm2d(3),
        nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
    )

    assert Infer_head_first_layer_size(head) == 3


def test_infer_head_first_layer_size_raises_when_missing_metadata():
    head = nn.Sequential(
        nn.ReLU(),
        nn.BatchNorm1d(4),
    )

    with pytest.raises(ValueError):
        Infer_head_first_layer_size(head)


OUT_CHANNEL_SIZES = [16, 32, 75, 15]

# %% Test computation of output size of Conv2d using default settings


def test_ComputeConv2dOutputSize():
    # %% Test computation of output size of Conv2d using default settings
    patchSize = [7, 7]
    convKernelSize = 3
    convStrideSize = 1
    convPaddingSize = 0
    conv2dOutputSize = ComputeConv2dOutputSize(
        patchSize, convKernelSize, convStrideSize, convPaddingSize)

    # Add asserts with expected value
    assert conv2dOutputSize == (
        5, 5), "The output size of ComputeConv2dOutputSize is incorrect."


def test_ComputePooling2dOutputSize():
    # %% Test computation of output size of Pooling2d using default settings
    poolingkernelSize = 2
    poolingStrideSize = 1
    poolingOutputSize = ComputePooling2dOutputSize(
        [5, 5], poolingkernelSize, poolingStrideSize)

    # Add asserts with expected value
    assert poolingOutputSize == (
        4, 4), "The output size of ComputePooling2dOutputSize is incorrect."


def test_ComputeConvBlockOutputSize_singleBlock():
    # %% Test computation of number of features after ConvBlock using default settings
    convBlockOutputSize, flattenedOutput = ComputeConvBlock2dOutputSize(
        [7, 7], OUT_CHANNEL_SIZES[0])

    assert convBlockOutputSize == (
        2, 2), "The 2d output size of ConvBlock (conv. + pooling) is incorrect."
    assert flattenedOutput == 1 * convBlockOutputSize[0] * convBlockOutputSize[1] * \
        OUT_CHANNEL_SIZES[0], "The flattened output size of ConvBlock is incorrect."


def test_ComputeConvBlockOutputSize_multiBlock():
    outputMapSize = [7, 7]
    convkernelSizes = [3, 3]
    convStrideSize = [1, 1]
    convPaddingSize = [0, 0]
    poolingkernelSize = [2, 2]
    poolingStrideSize = [1, 1]
    poolingPaddingSize = [0, 0]

    for idBlock in range(2):
        convBlockOutputSize, flattenedOutput = ComputeConvBlock2dOutputSize(
            outputMapSize,
            OUT_CHANNEL_SIZES[idBlock],
            convkernelSizes[idBlock],
            poolingkernelSize[idBlock],
            convStrideSize[idBlock],
            poolingStrideSize[idBlock],
            convPaddingSize[idBlock],
            poolingPaddingSize[idBlock]
        )

        # Add asserts for each block
        if idBlock == 0:
            assert convBlockOutputSize == (
                4, 4), "First ConvBlock output size is incorrect."
            assert flattenedOutput == OUT_CHANNEL_SIZES[idBlock] * \
                4 * 4, "First ConvBlock flattened output is incorrect."
        elif idBlock == 1:
            assert convBlockOutputSize == (
                1, 1), "Second ConvBlock output size is incorrect."
            assert flattenedOutput == OUT_CHANNEL_SIZES[idBlock] * \
                1 * 1, "Second ConvBlock flattened output is incorrect."

        outputMapSize[0] = convBlockOutputSize[0]
        outputMapSize[1] = convBlockOutputSize[1]


def test_AutoComputeConvBlocksOutput():
    final_output_size, flattened_sizes, intermediate_sizes = AutoComputeConvBlocksOutput(
        first_input_size=7,
        out_channels_sizes=OUT_CHANNEL_SIZES[:2],
        kernel_sizes=[3, 3],
        pooling_kernel_sizes=[2, 2],
        conv_stride_sizes=[1, 1],
        pooling_stride_sizes=[1, 1],
        conv2d_padding_sizes=[0, 0],
        pooling_padding_sizes=[0, 0],
    )

    assert intermediate_sizes == [(4, 4), (1, 1)]
    assert flattened_sizes == [OUT_CHANNEL_SIZES[0] * 4 * 4, OUT_CHANNEL_SIZES[1]]
    assert final_output_size == (1, 1)


def test_ComputeConvBlockOutputSize_raises_for_invalid_pooling():
    with pytest.raises(ValueError):
        ComputeConvBlock2dOutputSize(
            input_size=[2, 2],
            out_channels_size=8,
            conv2d_kernel_size=3,
            pooling_kernel_size=2,
        )
