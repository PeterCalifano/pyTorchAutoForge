from pyTorchAutoForge.model_building.modelBuildingFunctions import *
import torch

from pyTorchAutoForge.model_building.modelBuildingBlocks import ResizeCopyChannelsAdapter

# Block builders
def test_convolutional_block_builder():
    pass


def test_linear_block_builder():
    pass


def test_block_builder():
    # Define a simple block configuration
    block_config = BlockConfig(
        type=block_type.conv,
        layers=[
            ConvLayerConfig(in_channels=3, out_channels=16,
                            typename='conv2d', kernel_size=3, stride=1, padding=1),
            NormLayerConfig(num_features=16, typename='batchnorm2d'),
            ActivationLayerConfig('relu', inplace=True),
            PoolingLayerConfig()
        ]
    )

    # Build the block
    block = build_block(block_config)
    print(block)


def test_resizeCopyAdapter():

    # Example Sequential model integration
    # input_size = 224  # Example input size
    input_size = 512  # Example input size
    output_channels = 3  # Set to match feature extractor input channels
    resAdapter = nn.Sequential(
        # For grayscale input repeated to 3 channels
        ResizeCopyChannelsAdapter([input_size, input_size], [1, output_channels])
    )

    # Test with a grayscale image Tensor
    # Example grayscale Tensor with size 512x512
    x = torch.rand(1, 1, 512, 512)
    resized_copied_image = resAdapter(x)

    # Check output size
    assert resized_copied_image.size() == torch.Size(
        [1, 3, input_size, input_size])

    # Check values of the output Tensor
    assert resized_copied_image.shape == torch.Size(
        [1, 3, input_size, input_size])



if __name__ == "__main__":
    # Test set
    test_block_builder()
    test_convolutional_block_builder()
    test_linear_block_builder()
    
    print("All tests passed!")