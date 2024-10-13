from typing import Optional, Any, Union, IO
import torch
import mlflow
import os
import sys
import traceback
from torch import nn
import numpy as np
from dataclasses import dataclass, asdict, fields, Field, MISSING

from pyTorchAutoForge.datasets import DataloaderIndex, DataLoader
from pyTorchAutoForge.utils.utils import GetDevice, AddZerosPadding, GetSamplesFromDataset
from pyTorchAutoForge.api.torch import SaveTorchModel
from pyTorchAutoForge.optimization import CustomLossFcn

from typing import Callable, Optional
import torch.optim as optim
from pyTorchAutoForge.evaluation import ResultsPlotter

@dataclass
class ModelEvaluatorConfig():
    device = GetDevice()
    # TODO


class ModelEvaluator():
    """_summary_
    """
    def __init__(self, model: Union[nn.Module], lossFcn: Union[nn.Module, CustomLossFcn],
                 dataLoaderIndex: DataloaderIndex, evalFunction: Callable = None, 
                 plotter: Optional[ResultsPlotter] = None) -> None:
        """_summary_

        Args:
            model (Union[nn.Module]): _description_
            lossFcn (Union[nn.Module, CustomLossFcn]): _description_
            dataLoaderIndex (DataloaderIndex): _description_
            evalFunction (Callable, optional): _description_. Defaults to None.
        """

        self.model = model
        self.lossFcn = lossFcn
        self.validationDataloader = dataLoaderIndex.getValidationLoader()
        self.trainingDataloaderSize = len(self.validationDataloader)
        self.evalFunction = evalFunction
        self.device = GetDevice()

        self.stats = {}
        self.plotter = plotter

    def evaluateRegressor(self) -> dict:
        '''DEVNOTE: loss function averaging assumes that the batch_loss is not an average but a sum of losses'''
        self.model.eval()

        # Backup the original batch size (TODO: TBC if it is useful)
        original_dataloader = self.validationDataloader

        # Temporarily initialize a new dataloader for validation
        newBathSizeTmp = 2 * self.validationDataloader.batch_size  # TBC how to set this value

        tmpdataloader = DataLoader(
            original_dataloader.dataset,
            batch_size=newBathSizeTmp,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=0
        )

        dataset_size = len(tmpdataloader.dataset)

        # Get one sample from the dataset
        sample = GetSamplesFromDataset(tmpdataloader.dataset, 1)

        # Allocate torch tensors to store errors
        residuals = torch.zeros(dataset_size, sample[1].shape[1])

        # Perform model evaluation on all batches
        idAllocator = 0
        total_loss = 0.0
        with torch.no_grad():

            for X, Y in tmpdataloader:

                X, Y = X.to(self.device), Y.to(self.device)

                # Get batch size
                batchSize = X.shape[0]

                # Perform forward pass
                Y_hat = self.model(X)

                if self.evalFunction is not None:
                    # Compute errors per component
                    errorPerComponent = self.evalFunction(Y_hat, Y)
                else:
                    # Assume that error is computed as difference between prediction and target
                    errorPerComponent = Y_hat - Y

                if self.lossFcn is not None:
                    # Compute loss for ith batch
                    batch_loss = self.lossFcn(Y_hat, Y)
                    # Accumulate loss
                    total_loss += batch_loss.get('lossValue') if isinstance(batch_loss, dict) else batch_loss.item()

                # Store residuals
                allocRange = np.arange( idAllocator, idAllocator + batchSize + 1, dtype=int)
                residuals[allocRange, :] = errorPerComponent

                idAllocator += batchSize

            if self.lossFcn is not None:
                # Compute average loss value
                avg_loss = total_loss/dataset_size

            # Compute statistics
            avgResiduals = torch.mean(torch.abs(residuals), dim=0)
            stdResiduals = torch.std(torch.abs(residuals), dim=0)
            medianResiduals = torch.median(torch.abs(residuals), dim=0)
            maxResiduals = torch.max(torch.abs(residuals), dim=0)

        # Pack data into dict
        self.stats = {}
        self.stats['prediction_err'] = residuals.to('cpu').numpy()
        self.stats['average_prediction_err'] = avgResiduals.to('cpu').numpy()
        self.stats['median_prediction_err'] = medianResiduals.to('cpu').numpy()
        self.stats['max_prediction_err'] = maxResiduals.to('cpu').numpy()

        if self.lossFcn is not None:
            self.stats['avg_loss'] = avg_loss

        # Print statistics
        print('Avg residuals: ', avgResiduals)
        print('Std residuals: ', stdResiduals)
        print('Median residuals: ', medianResiduals)
        print('Max residuals: ', maxResiduals)

        return self.stats

    def plotResults(self) -> None:
        if self.plotter is not None:
            self.plotter.histPredictionErrors(self.stats)
        

    




