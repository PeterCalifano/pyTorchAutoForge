import torch, torchsummary
from torch import nn
import torchvision.models as models
from pyTorchAutoForge.optimization import FreezeModel

def test_FreezeModel():
    
    # Load EfficientNet model from torchvision
    feature_extractor = models.efficientnet_b0(weights=True)

    # Remove last Linear classifier and dropout layer (Classifier nn.Sequential module)
    feature_extractor = nn.Sequential(*list(feature_extractor.children())[:-1])

    # Freeze model parameters
    feature_extractor = FreezeModel(feature_extractor)

    # Print model summary
    #torchsummary.summary(feature_extractor, (3, 224, 224))

    # Check if model parameters are frozen
    for param in feature_extractor.parameters():
        assert param.requires_grad == False

    return 0

if __name__ == '__main__':
    test_FreezeModel()