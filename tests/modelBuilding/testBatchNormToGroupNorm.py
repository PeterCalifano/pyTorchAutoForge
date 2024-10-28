
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
from pyTorchAutoForge.modelBuilding import ModelMutatorBNtoGN

import matplotlib
matplotlib.use('agg')  # or 'Qt5Agg'
plt.ion()

DEBUG = False

def main():

    # Define adapter model to bring resolution down to feature_extractor input size
    resAdapter = inputResolutionAdapter([224, 224], [1, 3])
    print("Adapter model: \n", resAdapter)

    # Get model backbone from torchvision
    feature_extractor = models.efficientnet_b0(
        weights=True)  # Load model without weights
    # Remove last Linear classifier ()
    feature_extractor = nn.Sequential(*list(feature_extractor.children())[:-1])

    # TEST: Mutate all BatchNorm layers to GroupNorm
    feature_extractor = (ModelMutatorBNtoGN(32, feature_extractor)).MutateBNtoGN()

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