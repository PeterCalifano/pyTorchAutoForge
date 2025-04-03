from pyTorchAutoForge.model_building.modelBuildingFunctions import *


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

if __name__ == "__main__":
    # Test set
    test_block_builder()
    test_convolutional_block_builder()
    test_linear_block_builder()
    
    print("All tests passed!")