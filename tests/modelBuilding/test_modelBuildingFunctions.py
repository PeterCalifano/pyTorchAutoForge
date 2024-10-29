from pyTorchAutoForge.modelBuilding.modelBuildingFunctions import *


def test_activation_builder():
    # Test building of each layer
    activation_name = 'relu'
    relu = build_activation_layer(activation_name, show_defaults=True)
    print(relu)

    activation_name = 'lrelu'
    negative_slope = 0.05
    lrelu = build_activation_layer(
        activation_name, show_defaults=True, negative_slope=negative_slope)
    print(lrelu)

    activation_name = 'prelu'
    num_parameters = 5
    prelu = build_activation_layer(
        activation_name, num_parameters=num_parameters)
    print(prelu)

    activation_name = 'sigmoid'
    sigmoid = build_activation_layer(activation_name)
    print(sigmoid)

    activation_name = 'tanh'
    tanh = build_activation_layer(activation_name)
    print(tanh)

    # Assert equality
    assert isinstance(relu, nn.ReLU)
    assert isinstance(lrelu, nn.LeakyReLU)
    assert lrelu.negative_slope == negative_slope
    assert isinstance(prelu, nn.PReLU)
    assert prelu.num_parameters == num_parameters
    assert isinstance(sigmoid, nn.Sigmoid)
    assert isinstance(tanh, nn.Tanh)


def test_convolutional_builder():
    pass


def test_convolutional_block_builder():
    pass


def test_normalization_builder():
    pass


if __name__ == "__main__":
    test_activation_builder()
    test_convolutional_builder()
    test_convolutional_block_builder()
    test_normalization_builder()
    print("All tests passed")
