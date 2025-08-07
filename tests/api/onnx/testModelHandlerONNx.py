# Script created by PeterC to test ONNx related codes (mainly from Torch) - 10-06-2024
import onnx 
import torch
from torch import nn 

# Import modules
import sys, os
import datetime
import numpy as np


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


# Main program to run tests manually
def main():

    raise NotImplementedError("Test to implement.")

    # Remove test artifacts
    if os.path.isfile(os.path.join(exportPath, modelName + '.onnx')):
        os.remove(os.path.join(exportPath, modelName + '.onnx'))

if __name__ == '__main__':
    main()