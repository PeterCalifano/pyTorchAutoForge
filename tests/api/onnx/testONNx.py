# Script created by PeterC to test ONNx related codes (mainly from Torch) - 10-06-2024
import onnx 
import torch
from torch import nn 

# Import modules
import sys, os
# Append paths of custom modules
sys.path.append(os.path.join('/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch/customTorchTools'))
sys.path.append(os.path.join('/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch/LimbBasedNavigationAtMoon'))

from pyTorchAutoForge.api.onnx import ExportTorchModelToONNx
import datetime
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader # Utils for dataset management, storing pairs of (sample, label)
from torchvision import datasets # Import vision default datasets from torchvision
from torchvision.transforms import ToTensor # Utils

# Define a simple test class
class SimpleModel(nn.Module):
    def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)
       
inputDummySample = torch.randn(1, 10)

# Define export path and model name
exportPath = '.'
modelName = 'test_model_export'

def test_conversions_deprectated():

    # Instantiate the model and create a dummy input
    test_model = SimpleModel()

    # Convert to ONNx format and save
    modelONNx, pathToModel, convertedModel = ExportTorchModelToONNx(
        test_model, inputDummySample, exportPath, modelName, onnx_version=13)


# Main program to run tests manually
def main():
    test_conversions_deprectated()

    # Remove test artifacts
    if os.path.isfile(os.path.join(exportPath, modelName + '.onnx')):
        os.remove(os.path.join(exportPath, modelName + '.onnx'))

if __name__ == '__main__':
    main()