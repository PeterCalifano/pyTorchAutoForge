import inspect, torch
from typing import Union

# TODO (PC) complete rework of this module is needed
# %% Class to define a custom loss function for training, validation and testing - 01-06-2024
# NOTE: Function EvalLossFcn must be implemented using Torch operations to work!


class CustomLossFcn_old(torch.nn.Module):
    '''Custom loss function based class, instantiated by specifiying a loss function (callable object) and optionally, a dictionary containing parameters required for the evaluation'''

    def __init__(self, EvalLossFcn: callable, lossParams: dict = None) -> None:
        '''Constructor for CustomLossFcn class'''
        super(CustomLossFcn_old, self).__init__()  # Call constructor of nn.Module

        if len((inspect.signature(EvalLossFcn)).parameters) >= 2:
            self.LossFcnObj = EvalLossFcn
        else:
            raise ValueError(
                'Custom EvalLossFcn must take at least two inputs: inputVector, labelVector')

        # Store loss function parameters dictionary
        self.lossParams = lossParams

    # def setTrainingMode(self):
    #    self.lossParams = lossParams

    # def setEvalMode(self):

    def forward(self, predictVector, labelVector):
        ''''Forward pass method to evaluate loss function on input and label vectors using EvalLossFcn'''
        lossBatch = self.LossFcnObj(
            predictVector, labelVector, self.lossParams)

        if isinstance(lossBatch, torch.Tensor):
            assert (lossBatch.dim() == 0)
        elif isinstance(lossBatch, dict):
            assert (lossBatch.get('lossValue').dim() == 0)
        else:
            raise ValueError(
                'EvalLossFcn must return a scalar loss value (torch tensor) or a dictionary with a "lossValue" key')

        return lossBatch


class CustomLossFcn(torch.nn.Module):
    '''Custom loss function based class, instantiated by specifiying a loss function (callable object) and optionally, a dictionary containing parameters required for the evaluation'''

    def __init__(self) -> None:
        super().__init__()  # Call constructor of nn.Module
        
    def forward(self, predictedVector, labelVector):
        ''''Forward pass method to evaluate loss function on input and label vectors using EvalLossFcn'''
        lossBatch = self.LossFcnObj( predictedVector, labelVector, self.lossParams)

        if isinstance(lossBatch, torch.Tensor):
            assert (lossBatch.dim() == 0)
        elif isinstance(lossBatch, dict):
            assert (lossBatch.get('lossValue').dim() == 0)
        else:
            raise ValueError(
                'EvalLossFcn must return a scalar loss value (torch tensor) or a dictionary with a "lossValue" key')

        return lossBatch