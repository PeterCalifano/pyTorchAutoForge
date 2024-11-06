from pyTorchAutoForge.utils.utils import GetDevice
from pyTorchAutoForge.modelBuilding.modelClasses import torchModel
from pyTorchAutoForge.api.torch import LoadTorchModel
import numpy as np
import torch, os
from torch import nn
from typing import Union
from dataclasses import dataclass

@dataclass 
class MatlabWrapperConfig():
    # Default wrapper configuration
    DEBUG_MODE: bool = False
    device = GetDevice()
    input_shape_validation: list = None # None for no validation
    

# %% MATLAB wrapper class for Torch models evaluation - 11-06-2024 # TODO: update class
class TorchModelMATLABwrapper():
    '''Class to wrap a trained PyTorch model for evaluation in MATLAB'''

    def __init__(self, trainedModel: Union[str, nn.Module, torchModel], 
                 wrapperConfig: MatlabWrapperConfig = MatlabWrapperConfig()) -> None:
        '''Constructor for TorchModelMATLABwrapper'''

        # Initialize using configuration class
        self.DEBUG_MODE = wrapperConfig.DEBUG_MODE
        self.device = wrapperConfig.device
        self.enable_warning = True # To disable warning for batch size
        self.input_shape_validation = wrapperConfig.input_shape_validation

        # Load model as traced 
        if isinstance(trainedModel, str):

            # Get extension of model file
            filepath_noext, extension = os.path.splitext(str(trainedModel))

            # Check if extension is provided
            if extension != '.pt' and extension == '.pth':
                raise ValueError('Please provide a .pt file. This function only supports traced models at current stage and cannot load .pth state dict.')
            elif extension != '.pt' and extension == '':
                print('No extension provided. Assuming .pt extension for model file.')
                trainedModelPath = trainedModel + ".pt"  # Assume .pt extension
            elif extension == '.pt':
                trainedModelPath = trainedModel
            else:
                raise ValueError('Invalid model file extension. Please provide a .pt file. This function only supports traced models at current stage.')

            # Load model and set to eval()
            self.trainedModel = LoadTorchModel(None, trainedModelPath, loadAsTraced=True).to(self.device)
            (self.trainedModel).eval()

        # Print model data
        if self.DEBUG_MODE:
            print("The following model has been loaded and will be used in forward() call: \n", self.trainedModel)

    def forward(self, inputSample: Union[np.ndarray, torch.Tensor], numBatches: int = None):
        '''Forward method to perform inference on N sample input using loaded trainedModel. Batch size assumed as 1 if not given.'''
        
        if self.DEBUG_MODE:
            print('Input sample shape: ', X.shape, 'on device: ', self.device)
        
        if self.input_shape_validation is not None:
            if inputSample.shape != self.input_shape_validation:
                raise ValueError(f'Input shape {X.shape} does not match the expected shape: {self.input_shape_validation}')
            
        try:
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
            if isinstance(inputSample, np.ndarray) and inputSample.dtype != np.float32:
                Warning('Converting input to np.float32 from', inputSample.dtype)
                inputSample = torch.from_numpy(np.float32(inputSample))

            elif isinstance(inputSample, torch.Tensor) and inputSample.dtype != torch.float32:
                Warning('Converting input to torch.float32 from', inputSample.dtype)
                inputSample = inputSample.float()


            # Reshape according to batch dimension
            X = inputSample.reshape(numBatches, -1)

        except Exception as e:
            max_chars = 400  # Define the max length you want to print
            print( f"\nError during input preprocessing: {str(e)[:max_chars]}...")
            return 1
        
        # Perform inference using model
        try:
            Y = self.trainedModel(X.to(self.device))
        except Exception as e:        
            max_chars = 400  # Define the max length you want to print
            print(f"\nError during model inference: {str(e)[:max_chars]}...")
            return 1

        # ########### DEBUG ######################:
        if self.DEBUG_MODE:
            print('Model prediction: ', Y)
        ############################################

        return Y.detach().cpu().numpy()  # Move to cpu and convert to numpy before returning



