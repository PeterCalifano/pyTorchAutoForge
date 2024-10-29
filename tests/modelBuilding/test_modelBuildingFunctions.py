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
    # Test building each normalization layer
    dict_key = 'BatchNorm2d'
    num_features = 10
    batch_norm = build_normalization_layer(
        dict_key, show_defaults=True, num_features=num_features)
    print(batch_norm)

    dict_key = 'LayerNorm'
    normalized_shape = [10, 10]
    layer_norm = build_normalization_layer(
        dict_key, show_defaults=True, normalized_shape=normalized_shape)
    print(layer_norm)

    dict_key = 'InstanceNorm2d'
    num_features = 10
    instance_norm = build_normalization_layer(
        dict_key, show_defaults=True, num_features=num_features)
    print(instance_norm)

    dict_key = 'GroupNorm'
    num_groups = 2
    num_channels = 10
    group_norm = build_normalization_layer(
        dict_key, show_defaults=True, num_groups=num_groups, num_channels=num_channels)
    print(group_norm)

    # Assert equality
    assert isinstance(batch_norm, nn.BatchNorm2d)
    assert batch_norm.num_features == num_features
    assert isinstance(layer_norm, nn.LayerNorm)
    assert list(layer_norm.normalized_shape) == normalized_shape
    assert isinstance(instance_norm, nn.InstanceNorm2d)
    assert instance_norm.num_features == num_features
    assert isinstance(group_norm, nn.GroupNorm)
    assert group_norm.num_groups == num_groups
    assert group_norm.num_channels == num_channels


if __name__ == "__main__":
    test_activation_builder()
    test_normalization_builder()
    test_convolutional_builder()
    test_convolutional_block_builder()
    print("All tests passed")
