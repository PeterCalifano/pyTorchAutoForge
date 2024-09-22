from pyTorchAutoForge.utils.utils import GetDevice
from pyTorchAutoForge.api.torch import LoadTorchModel
import numpy as np
import torch

from typing import Union

# %% MATLAB wrapper class for Torch models evaluation - 11-06-2024
class TorchModelMATLABwrapper():
    '''Class to wrap a trained PyTorch model for evaluation in MATLAB'''

    def __init__(self, trainedModelPath: str, trainedModelName: Union[str, None] = None, DEBUG_MODE: bool = False) -> None:
        '''Constructor for TorchModelMATLABwrapper'''

        # Get available device
        self.device = GetDevice()

        # Load model state and state
        if trainedModelName is None:
            # TODO: Implement model name extraction from path
            raise NotImplementedError('Current version is not able to separate model name from path. Please provide model name as second argument.')

        self.trainedModel = LoadTorchModel(None, trainedModelName, trainedModelPath, loadAsTraced=True).to(self.device)
        (self.trainedModel).eval()  # Set model in evaluation mode

        # Set debug mode flag
        self.DEBUG_MODE = DEBUG_MODE

    def forward(self, inputSample: Union[np.array, torch.tensor], numBatches: int = 1):
        '''Forward method to perform inference on N sample input using loaded trainedModel. Batch size assumed as 1 if not given.'''

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
            print('Evaluating model using batch input: ', X)
        ############################################

        # Perform inference using model
        Y = self.trainedModel(X.to(self.device))

        # ########### DEBUG ######################:
        if self.DEBUG_MODE:
            print('Model prediction: ', Y)
        ############################################

        return Y.detach().cpu().numpy()  # Move to cpu and convert to numpy before returning
