''' Script created by PeterC to test evaluation of loaded model - 10-06-2024 '''
import onnx 
import torch
from torch import nn 

# Import modules
import sys, os
# Append paths of custom modules
sys.path.append(os.path.join('/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch'))
sys.path.append(os.path.join('/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch/LimbBasedNavigationAtMoon'))

import PyTorch.pyTorchAutoForge.pyTorchAutoForge.pyTorchAutoForge as pyTorchAutoForge # Custom torch tools
import limbPixelExtraction_CNN_NN

import datetime
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader # Utils for dataset management, storing pairs of (sample, label)
from torchvision import datasets # Import vision default datasets from torchvision
from torchvision.transforms import ToTensor # Utils


def main():

    # Define inputs
    exportPath = './ExportedModelsToONNx'
    modelSavePath = './checkpoints/HorizonPixCorrector_CNN_run8'
    tracedModelSavePath = '/home/peterc/devDir/MachineLearning_PeterCdev'

    modelName = 'trainedModel_' + pyTorchAutoForge.AddZerosPadding(0, 4)
    tracedModelName = 'trainedTracedModel075.pt'

    datasetSavePath = modelSavePath + 'sampleDatasetToONNx'
    batch_size = 16

    device = pyTorchAutoForge.GetDevice()

    # NOTE: these settings must be the same as the saved model. Current version does not check for this.
    outChannelsSizes = [16, 32, 75, 15]
    kernelSizes = [3, 1]

    modelEmpty = limbPixelExtraction_CNN_NN.HorizonExtractionEnhancerCNN(outChannelsSizes, kernelSizes)

    # Load torch model and define loss function
    trainedModel = pyTorchAutoForge.LoadTorchModel(modelEmpty, modelName, modelSavePath).to(device)

    trainedTracedModel = pyTorchAutoForge.LoadTorchModel(None, tracedModelName, tracedModelSavePath, True).to(device)

    lossFcn = pyTorchAutoForge.CustomLossFcn(pyTorchAutoForge.MoonLimbPixConvEnhancer_LossFcn)

    # Load sample dataset
    sampleData = pyTorchAutoForge.LoadTorchDataset(datasetSavePath)
    sampleDataset  = DataLoader(sampleData, batch_size, shuffle=True)

    # Test model and get sample inputs
    examplePrediction, exampleLosses, inputSampleList = pyTorchAutoForge.EvaluateModel(sampleDataset, trainedModel, lossFcn)
    
    examplePrediction, exampleLosses, inputSampleList = pyTorchAutoForge.EvaluateModel(sampleDataset, trainedTracedModel, lossFcn)

    pyTorchAutoForge.SaveTorchModel(trainedTracedModel.to('cpu'), os.path.join(tracedModelSavePath, 'trainedTracedModel075_cpu'), True, inputSampleList[0])

    # %% TEST TORCH MATLAB WRAPPER
    torchWrapper = pyTorchAutoForge.TorchModel_MATLABwrap(tracedModelName, tracedModelSavePath)

    testPrediction = torchWrapper.forward((inputSampleList[0].cpu()).numpy())

    print('Test output:', testPrediction)

if __name__ == '__main__':
    main()