import torch
from torchvision import models
from pyTorchAutoForge.model_building.backbones.efficient_net import EfficientNetConfig, FeatureExtractorFactory, EfficientNetBackbone


def test_efficientnet_backbone_basic():
    # Create a dummy configuration
    cfg = EfficientNetConfig(
        model_name='b0',
        input_resolution=(224, 224),
        pretrained=True,
        output_size=10,
        remove_classifier=True,
        device='cpu',
        input_channels=3,
        output_type='last',
    )

    # Create the backbone

    backbone_last = FeatureExtractorFactory(cfg)
    print(backbone_last)

    # Try inference with a dummy input
    dummy_input = torch.randn(2, 3, 224, 224)  # Batch size of 2, 3 channels, 224x224 image

    # Forward pass
    output = backbone_last(dummy_input)
    print("Output type:", type(output))
    print("Output shape:", output.shape)

    # Adjust based on your model's output size
    assert output.shape == (2, 10)

def test_efficientnet_backbone_intermediatefeats():
    # Create configuration for spill_features
    cfg = EfficientNetConfig(
        model_name='b0',
        input_resolution=(224, 224),
        pretrained=True,
        output_size=10,
        remove_classifier=True,
        device='cpu',
        input_channels=3,
        output_type='spill_features',
    )

    # Create the backbone
    backbone_features = FeatureExtractorFactory(cfg)
    print(backbone_features)

    # Try inference with a dummy input
    # Batch size of 2, 3 channels, 224x224 image
    dummy_input = torch.randn(2, 3, 224, 224)

    # Forward pass
    output_list = backbone_features(dummy_input)
    print("Output type:", type(output_list))
    print("Output shape:", len(output_list))
    for i, output in enumerate(output_list):
        print(f"Output {i} shape:", output.shape)

    # Add asserts of expected output shapes
    assert isinstance(output_list, list)
    assert len(output_list) == 11  # Number of layers in EfficientNetB0
    # Adjust based on your model's output size
    assert output_list[-1].shape == (2, 10)


# Manual run
if __name__ == "__main__":
    test_efficientnet_backbone_basic()
    test_efficientnet_backbone_intermediatefeats()