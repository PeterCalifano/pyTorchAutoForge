
import sys
import os

# Add path to sys
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../../operative/operative-develop/src')))

# Import modules
from tailoredModels import FullDiskDataset, ShiftDiskTransform, NormalizeImage, inputResolutionAdapter, CentroidRangeDeepMSEloss
import matplotlib.pyplot as plt
import torch

from torch import nn
from torchsummary import summary
# Utils for dataset management, storing pairs of (sample, label)
from torch.utils.data import DataLoader
import numpy as np
from typing import Union
import cv2 as ocv

from pyTorchAutoForge.optimization.ModelTrainingManager import ModelTrainingManager, ModelTrainingManagerConfig, TaskType, TrainModel, ValidateModel
from pyTorchAutoForge.datasets import DataloaderIndex, Dataset, DataLoader
import torchvision.models as models

from pyTorchAutoForge.utils import GetDevice
from pyTorchAutoForge.modelBuilding import ModelAutoBuilder, TemplateDeepNet, TemplateConvNet

import matplotlib
matplotlib.use('agg')  # or 'Qt5Agg'
plt.ion()

DEBUG = False


class ModelMutatorBNtoGN():
    def __init__(self, model: nn.Module, numOfGroups: int = 32) -> None:
        self.numOfGroups = numOfGroups
        self.model = model

    def MutateBNtoGN(self) -> nn.Module:
        """Mutates all BatchNorm layers to GroupNorm layers in the model"""

        self.replace_batchnorm_with_groupnorm(self.model, self.numOfGroups)

        return self.model

    def replace_batchnorm_with_groupnorm(self, module, numOfGroups):
        """Recursively replaces BatchNorm2d layers with GroupNorm, adjusting num_groups dynamically."""
        for name, layer in module.named_children():
            if isinstance(layer, nn.BatchNorm2d):
                num_channels = layer.num_features
                # Check if divisible by num_groups
                if num_channels % self.numOfGroups != 0:
                    num_groups = self.find_divisible_groups(num_channels)
                else:
                    num_groups = self.numOfGroups
                # Replace BN with GroupNorm
                setattr(module, name, nn.GroupNorm(num_groups, num_channels))
            else:
                # Recurse to find all BatchNorm layers
                self.replace_batchnorm_with_groupnorm(layer, numOfGroups)

    def find_divisible_groups(self, num_channels):
        """Finds an appropriate number of groups for GroupNorm that divides num_channels."""
        # Start with 32, or reduce it if it doesn’t divide num_channels
        for groups in [16, 8, 4, 2]:  # Attempt standard group sizes
            if num_channels % groups == 0:
                return groups
        raise ValueError(f"Could not find a suitable number of groups for {num_channels} channels.")


def main():

    # Define adapter model to bring resolution down to feature_extractor input size
    resAdapter = inputResolutionAdapter([224, 224], [1, 3])
    print("Adapter model: \n", resAdapter)

    # Get model backbone from torchvision
    feature_extractor = models.efficientnet_b1(
        weights=True)  # Load model without weights
    # Remove last Linear classifier ()
    #feature_extractor = nn.Sequential(*list(feature_extractor.children())[:-1])

    # TEST: Mutate all BatchNorm layers to GroupNorm
    feature_extractor = (ModelMutatorBNtoGN(feature_extractor, 32)).MutateBNtoGN()

    print("EfficientNet B model modified to use GN: \n", feature_extractor)
    efficientNet_output_size = 1280

    # Define DNN regressor for centroid prediction

    headCentroid_config = {
        'input_size': efficientNet_output_size,
        'useBatchNorm': False,
        'alphaDropCoeff': 0.0,
        'alphaLeaky': 0.0,
        'outChannelsSizes': [256, 128, 32, 16, 2]
    }

    centroidRegressor = TemplateDeepNet(headCentroid_config)
    print("Centroid regressor model: \n", headCentroid_config)

    model = nn.Sequential(
        resAdapter,
        feature_extractor,
        centroidRegressor
    )

    print('\n MODEL AFTER MUTATION')
    print(model)

    # Move model to device
    #summary(model, (1, 1024, 1024))


if __name__ == '__main__':
    main()