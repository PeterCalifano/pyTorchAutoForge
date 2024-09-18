from pyTorchAutoForge.utils.utils import GetDevice
from pyTorchAutoForge.api.torch import LoadTorchModel
import numpy as np
import torch

# %% MATLAB wrapper class for Torch models evaluation - 11-06-2024
class TorchModelMATLABwrapper():
    def __init__(self, trainedModelName: str, trainedModelPath: str) -> None:
        # Get available device
        self.device = GetDevice()

        # Load model state and state
        trainedModel = LoadTorchModel(
            None, trainedModelName, trainedModelPath, loadAsTraced=True)
        trainedModel.eval()  # Set model in evaluation mode
        self.trainedModel = trainedModel.to(self.device)

    def forward(self, inputSample: np.array, numBatches: int = 1):
        '''Forward method to perform inference for ONE sample input using trainedModel'''
        if inputSample.dtype is not np.float32:
            inputSample = np.float32(inputSample)

        print('Performing formatting of input array to pass to model...')

        # TODO: check the input is exactly identical to what the model receives using EvaluateModel() loading from dataset!
        # Convert numpy array into torch.tensor for model inference
        X = torch.tensor(inputSample).reshape(numBatches, -1)

        # ########### DEBUG ######################:
        print('Evaluating model using batch input: ', X)
        ############################################

        # Perform inference using model
        Y = self.trainedModel(X.to(self.device))

        # ########### DEBUG ######################:
        print('Model prediction: ', Y)
        ############################################

        return Y.detach().cpu().numpy()  # Move to cpu and convert to numpy
