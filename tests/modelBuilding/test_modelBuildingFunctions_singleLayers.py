from pyTorchAutoForge.model_building.modelBuildingFunctions import *

# Single layer builders
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
    # Test building each convolutional layer
    dict_key = 'Conv1d'
    in_channels_1d, out_channels_1d, kernel_size_1d = (5, 5, (3,))
    conv1d = build_convolutional_layer(
        dict_key, show_defaults=True, in_channels=in_channels_1d, out_channels=out_channels_1d, kernel_size=kernel_size_1d)
    print(conv1d)

    dict_key = 'Conv2d'
    in_channels_2d, out_channels_2d, kernel_size_2d = (3, 6, (3, 3))
    conv2d = build_convolutional_layer(
        dict_key, show_defaults=True, in_channels=in_channels_2d, out_channels=out_channels_2d, kernel_size=kernel_size_2d)
    print(conv2d)

    dict_key = 'Conv3d'
    in_channels_3d, out_channels_3d, kernel_size_3d = (3, 6, (3, 3, 3))
    conv3d = build_convolutional_layer(
        dict_key, show_defaults=True, in_channels=in_channels_3d, out_channels=out_channels_3d, kernel_size=kernel_size_3d)
    print(conv3d)

    # Assert equality
    assert isinstance(conv1d, nn.Conv1d)
    assert conv1d.in_channels == in_channels_1d
    assert conv1d.out_channels == out_channels_1d
    assert conv1d.kernel_size == kernel_size_1d

    assert isinstance(conv2d, nn.Conv2d)
    assert conv2d.in_channels == in_channels_2d
    assert conv2d.out_channels == out_channels_2d
    assert conv2d.kernel_size == kernel_size_2d

    assert isinstance(conv3d, nn.Conv3d)
    assert conv3d.in_channels == in_channels_3d
    assert conv3d.out_channels == out_channels_3d
    assert conv3d.kernel_size == kernel_size_3d

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

def test_pooling_builder():

    # Show optional defaults for AdaptiveMaxPool2d and create it with output size 3x3
    adaptive_pooling_layer = build_pooling_layer('AdaptiveMaxPool2d', show_defaults=True, output_size=(3, 3))
    print(adaptive_pooling_layer)

    # Create MaxPool1d with a specific kernel size and default settings for other parameters
    pooling_layer = build_pooling_layer('MaxPool1d', kernel_size=3)
    print(pooling_layer)

    # Check instances of the pooling layers
    assert isinstance(adaptive_pooling_layer, nn.AdaptiveMaxPool2d)
    assert adaptive_pooling_layer.output_size == (3, 3)
    assert isinstance(pooling_layer, nn.MaxPool1d)
    assert pooling_layer.kernel_size == 3

def test_linear_builder():
    # Test building each linear layer
    dict_key = 'Linear'
    in_features, out_features = (10, 5)
    linear = build_linear_layer(dict_key, show_defaults=True, in_features=in_features, out_features=out_features)
    print(linear)

    # Assert equality
    assert isinstance(linear, nn.Linear)
    assert linear.in_features == in_features
    assert linear.out_features == out_features 

if __name__ == "__main__":
    test_activation_builder()
    test_normalization_builder()
    test_convolutional_builder()
    test_pooling_builder()
    test_linear_builder()
    print("All tests passed")
