from pyTorchAutoForge.utils.utils import GetDevice
from pyTorchAutoForge.modelBuilding.modelClasses import torchModel
from pyTorchAutoForge.api.torch import LoadTorchModel
import numpy as np
import torch
from torch import nn

from typing import Union
from dataclasses import dataclass

@dataclass 
class WrapperConfig():

    # Default wrapper configuration
    DEBUG_MODE: bool = False
    device = GetDevice()
    pass

# %% MATLAB wrapper class for Torch models evaluation - 11-06-2024 # TODO: update class
class TorchModelMATLABwrapper():
    '''Class to wrap a trained PyTorch model for evaluation in MATLAB'''

    def __init__(self, trainedModel: Union[str, nn.Module, torchModel], 
                 wrapperConfig: WrapperConfig = WrapperConfig()) -> None:
        '''Constructor for TorchModelMATLABwrapper'''

        # Initialize using configuration class
        self.DEBUG_MODE = wrapperConfig.DEBUG_MODE
        self.device = wrapperConfig.device
        self.enable_warning = True # To disable warning for batch size

        # Load model as traced 
        if isinstance(trainedModel, str):

            # Get model name splitting from path
            trainedModelName = trainedModel.split('/')[-1].split('.')[0]
            trainedModelPath = trainedModel.split(trainedModelName)[0] + ".pt" # To verify

            self.trainedModel = LoadTorchModel(None, trainedModelName, trainedModelPath, loadAsTraced=True).to(self.device)
            (self.trainedModel).eval()  # Set model in evaluation mode

        # TODO: Implement code to print model summary and information to MATLAB (reduces likelihood of errors in loading model)

    def forward(self, inputSample: Union[np.array, torch.tensor], numBatches: int = None):
        '''Forward method to perform inference on N sample input using loaded trainedModel. Batch size assumed as 1 if not given.'''
        
        # Set batch size if not provided
        if numBatches != None and len(inputSample.shape) > 2:
            # Assume first dimension of array is batch size
            numBatches = inputSample.shape[0]
        else:
            if self.enable_warning:
                Warning('Batch size not provided and input is two-dimensional. Assuming batch size of 1.')
                self.enable_warning = False
            numBatches = 1

        # Check input type and convert to torch.tensor if necessary
        if inputSample is np.array and inputSample.dtype != np.float32:
            Warning('Converting input to np.float32 from', inputSample.dtype)
            inputSample = np.float32(inputSample)

        elif inputSample is torch.tensor and inputSample.dtype != torch.float32:
            Warning('Converting input to torch.float32 from', inputSample.dtype)
            inputSample = inputSample.float()

        # Convert numpy array into torch.tensor for model inference
        X = torch.tensor(inputSample).reshape(numBatches, -1) if inputSample is np.array else inputSample.reshape(numBatches, -1)

        # ########### DEBUG ######################:
        if self.DEBUG_MODE:
            print('Input sample shape: ', X.shape, 'on device: ', self.device)
            print('Evaluating model using batch input: ', X) # Add truncation for large arrays
        ############################################
    
        # TODO: Add check on input shape before attempting inference

        # Perform inference using model
        try:
            Y = self.trainedModel(X.to(self.device))
        except Exception as e:
            raise Exception('Error during model inference: ', e)

        # ########### DEBUG ######################:
        if self.DEBUG_MODE:
            print('Model prediction: ', Y)
        ############################################

        return Y.detach().cpu().numpy()  # Move to cpu and convert to numpy before returning


# Test function definition for pytest
def test_TorchModelMATLABwrapper():
    pass