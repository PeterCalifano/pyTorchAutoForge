from typing import Optional, Any, Union, IO
import torch
import mlflow
import os
import sys
import traceback
from torch import NoneType, nn
import numpy as np
from dataclasses import dataclass, asdict, fields, Field, MISSING

from pyTorchAutoForge.datasets import DataloaderIndex
from torch.utils.data import DataLoader
from pyTorchAutoForge.utils.utils import GetDevice, AddZerosPadding, GetSamplesFromDataset
from pyTorchAutoForge.api.torch import SaveTorchModel
from pyTorchAutoForge.optimization import CustomLossFcn

from collections.abc import Callable
import torch.optim as optim
from pyTorchAutoForge.evaluation import ResultsPlotter

@dataclass
class ModelEvaluatorConfig():
    device = GetDevice()
    # TODO


class ModelEvaluator():
    """_summary_
    """
    def __init__(self, model: nn.Module, lossFcn: nn.Module | CustomLossFcn,
                 dataLoader: DataLoader, device: str = 'cpu', evalFunction: Callable | None = None, 
                 plotter: ResultsPlotter = None) -> None:
        """_summary_

        Args:
            model (Union[nn.Module]): _description_
            lossFcn (Union[nn.Module, CustomLossFcn]): _description_
            dataLoaderIndex (DataloaderIndex): _description_
            evalFunction (Callable, optional): _description_. Defaults to None.
        """

        self.lossFcn = lossFcn
        self.validationDataloader = dataLoader
        self.trainingDataloaderSize = len(self.validationDataloader)
        self.evalFunction = evalFunction
        self.device = device

        self.model = model.to(self.device)

        self.stats = {}
        self.plotter = plotter

    def evaluateRegressor(self) -> dict:
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
        numOfBatches = len(tmpdataloader)
        residuals = None

        # Perform model evaluation on all batches
        total_loss = 0.0
        print('\nEvaluating model on validation dataset...\n')
        with torch.no_grad():

            for batch_idx, (X, Y) in enumerate(tmpdataloader):

                X, Y = X.to(self.device), Y.to(self.device)

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
                if residuals is None:
                    residuals = errorPerComponent
                else:
                    residuals = torch.cat((residuals, errorPerComponent), dim=0)

                # Print progress
                current_batch = batch_idx + 1
                progress = f"Evaluating: Batch {batch_idx+1}/{numOfBatches}"
                # Print progress on the same line
                sys.stdout.write('\r' + progress)
                sys.stdout.flush()

            print('\n')
            if self.lossFcn is not None:
                # Compute average loss value
                avg_loss = total_loss/dataset_size

            # Compute statistics
            meanResiduals = torch.mean(residuals, dim=0)
            avgResidualsErr = torch.mean(torch.abs(residuals), dim=0)
            stdResiduals = torch.std(torch.abs(residuals), dim=0)
            medianResiduals, _ = torch.median(torch.abs(residuals), dim=0)
            maxResiduals, _ = torch.max(torch.abs(residuals), dim=0)

        # Pack data into dict
        self.stats = {}
        self.stats['prediction_err'] = residuals.to('cpu').numpy()
        self.stats['average_prediction_err'] = avgResidualsErr.to('cpu').numpy()
        self.stats['median_prediction_err'] = medianResiduals.to('cpu').numpy()
        self.stats['max_prediction_err'] = maxResiduals.to('cpu').numpy()
        self.stats['mean_prediction_err'] = meanResiduals.to('cpu').numpy()

        if self.lossFcn is not None:
            self.stats['avg_loss'] = avg_loss

        # Print statistics
        print('Mean of residuals: ', meanResiduals)
        print('Std of residuals: ', stdResiduals)
        print('Median of residuals: ', medianResiduals)
        print('Max of residuals: ', maxResiduals)

        return self.stats

    def plotResults(self, entriesNames: list = None, units: list = None,
                    unit_scalings: Union[dict, list, np.ndarray, float, int] = None, colours: list = None, num_of_bins: int = 100) -> None:
        if self.plotter is not None:
            self.plotter.histPredictionErrors(self.stats, entriesNames=entriesNames, units=units, unit_scalings=unit_scalings, colours=colours, num_of_bins=num_of_bins)
        else:
            Warning('No plotter object provided. Cannot plot results.')
        

    




