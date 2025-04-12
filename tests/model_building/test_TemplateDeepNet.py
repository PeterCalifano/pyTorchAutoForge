import matplotlib
from tailoredModels import *
from datasetHandling import *
from definitionFcns import LoadDataset
from pyTorchAutoForge.api.torch import SaveModel, LoadModel
from pyTorchAutoForge.model_building import ModelAutoBuilder, TemplateDeepNet, TemplateConvNet
from pyTorchAutoForge.utils import GetDevice
import torchvision.models as models
import cv2 as ocv
from typing import Union
import numpy as np
import mlflow
from torchvision import transforms
from torchvision import datasets  # Import vision default datasets from torchvision
from torch.utils.data import DataLoader
from torchsummary import summary
from torch import nn
import json
import torch
import matplotlib.pyplot as plt
from tailoredModels import FullDiskDataset, ShiftDiskTransform, NormalizeImage, conv2dResolutionAdapter, CentroidRangeDeepMSEloss
import sys
import os

# Add path to sys
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../src')))

# Import modules
# Utils for dataset management, storing pairs of (sample, label)


matplotlib.use('agg')  # or 'Qt5Agg'
plt.ion()

DEBUG = False


def main():
    # TODO
    # raise NotImplementedError("This script is not yet ready for use")
    # Script setup
    device = GetDevice()

    # Define adapter model to bring resolution down to feature_extractor input size
    resAdapter = conv2dResolutionAdapter([224, 224], [1, 3])
    print("Adapter model: \n", resAdapter)

    # cnn_config = {
    #    'kernelSizes': [],
    #    'poolkernelSizes': [],
    #    'useBatchNorm': False,
    #    'alphaDropCoeff': 0,
    #    'alphaLeaky': 0.0,
    #    'outChannelsSizes': [128, 64, 32, 3]
    # }
    # convResAdapter = TemplateConvNet(cnn_config)

    # Define optimizer object
    initial_lr = 5E-3
    numOfEpochs = 200
    batch_size = 50

    # Get model backbone from torchvision
    feature_extractor = models.efficientnet_b0(
        weights=True)  # Load model without weights

    # Remove last Linear classifier ()
    feature_extractor = nn.Sequential(*list(feature_extractor.children())[:-1])

    # NOTE: expected input size is 240x240, 3 channels
    # Expected output for EfficientNet-B1 feature extractor, the shape of the extracted features before the final classification layer
    # will be approximately (batch_size, 1280, 7, 7) (which represents the spatial dimensions and channels of the last convolutional block)
    # NOTE: but... does using only last layer output implies using only higher level features?
    print("EfficientNet B model: \n", feature_extractor)
    efficientNet_output_size = 1280

    # Define DNN regressor for centroid and rangbestValidationLosse prediction
    regressor_config = {
        'input_size': efficientNet_output_size,
        'useBatchNorm': False,
        'alphaDropCoeff': 0,
        'alphaLeaky': 0.0,
        'outChannelsSizes': [256, 128, 32, 3]
    }

    headCentroid_config = {
        'input_size': efficientNet_output_size,
        'useBatchNorm': False,
        'alphaDropCoeff': 0.0,
        'alphaLeaky': 0.0,
        'outChannelsSizes': [256, 128, 32, 16, 2]
    }

    centroidRegressor = TemplateDeepNet(headCentroid_config)
    print("Centroid regressor model: \n", headCentroid_config)
    # NOTE: why torch summary does not see parameters in regressor?
    # Assembly model

    headRange_config = {
        'input_size': efficientNet_output_size,
        'useBatchNorm': False,
        'alphaDropCoeff': 0.0,
        'alphaLeaky': 0.0,
        'outChannelsSizes': [512, 256, 32, 1]
    }

    multiHead = CascadedDeepRegressor({"head_centroid": headCentroid_config,
                                       "head_range": headRange_config})

    multiHead.to(device)

    model = nn.Sequential(
        resAdapter,
        feature_extractor,
        multiHead
    )

    # Move model to device
    # DEVNOTE: does it work on the whole model? --> not working
    model.to(device)

    # Check if whole model is on device
    if not (next(model.parameters()).is_cuda):
        raise Exception("Model is not entirely on CUDA device")

    summary(model, (1, 1024, 1024))

    # Load dataset # TODO: try to define a general-purpose class to automate dataset retrieval/loading/split
    # TODO: load labels of centroid and range (in Moon radii)
    # Datasets are defined in folder with name: Dataset_<type>_hash
    # Images in /images, labels in /labels

    # Load dataset
    datasetID = 2  # 250 images for faster loading
    hostname = os.uname().nodename
    datasetsRootFolder = "dataset_gen/output"

    dataDict = LoadDataset(datasetID, datasetsRootFolder, hostname)

    # Define datasets
    train_dataset = FullDiskDataset(
        dataDict, transform_shift=None, transform_normalize=NormalizeImage())  # Do not apply transform
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    # Get input sample
    sampleBatch = next(iter(train_loader))
    inputSample = sampleBatch[0]

    try:
        model.eval()
        # Test saving of TRACED model
        SaveTorchModel(model, "testdata/test_model_traced",
                       saveAsTraced=True, exampleInput=inputSample, targetDevice=device)

        # Test loading of TRACED model
        loadedModel = LoadTorchModel("testdata/test_model_traced.pt", device)
    except Exception as e:
        # Print error message limiting to 300 characters
        print(f"Error message: {str(e)[:300]}")


if __name__ == "__main__":
    main()
