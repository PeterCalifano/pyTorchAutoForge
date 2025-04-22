import torch
import sys
from torch import nn
import numpy as np
from dataclasses import dataclass

from pyTorchAutoForge.utils import torch_to_numpy
from pyTorchAutoForge.datasets import DataloaderIndex
from torch.utils.data import DataLoader
from pyTorchAutoForge.utils.utils import GetDevice
from pyTorchAutoForge.optimization import CustomLossFcn

from collections.abc import Callable
from pyTorchAutoForge.evaluation import ResultsPlotterHelper


@dataclass
class ModelEvaluatorConfig():
    device = GetDevice()
    # TODO

class ModelEvaluator():
    """
    ModelEvaluator _summary_

    _extended_summary_
    """

    def __init__(self, model: nn.Module, 
                 lossFcn: nn.Module | CustomLossFcn,
                 dataLoader: DataLoader, 
                 device: str = 'cpu', 
                 evalFunction: Callable | None = None,
                 plotter: ResultsPlotterHelper | None = None,
                 make_plot_predict_vs_target: bool = False) -> None:
        """
            # TODO
        """

        self.loss_fcn = lossFcn
        self.validationDataloader : DataLoader = dataLoader
        self.trainingDataloaderSize : int = len(self.validationDataloader)
        self.eval_function = evalFunction
        self.device = device
        self.make_plot_predict_vs_target = make_plot_predict_vs_target

        self.model = model.to(self.device)
        self.stats : dict = {}
        self.plotter = plotter

    def evaluateRegressor(self) -> dict:
        self.model.eval()

        # Backup the original batch size (TODO: TBC if it is useful)
        original_dataloader = self.validationDataloader

        if self.validationDataloader is None:
            raise ValueError('Validation dataloader is None. Cannot evaluate model.')
        
        if self.validationDataloader.batch_size is None:
            raise ValueError('Batch size of dataloader is None. Cannot evaluate model.')
        
        # Temporarily initialize a new dataloader for validation
        new_batch_size_tmp = 2 * int(self.validationDataloader.batch_size)  # TBC how to set this value

        # Create a new dataloader with the same dataset but doubled batch size for speed
        tmp_dataloader = DataLoader(
            original_dataloader.dataset,
            batch_size=new_batch_size_tmp,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=0
        )

        dataset_size = len(tmp_dataloader.dataset)
        num_batches = len(tmp_dataloader)
        batch_size : int = tmp_dataloader.batch_size
        residuals = None

        # Perform model evaluation on all batches
        total_loss = 0.0
        print('\nEvaluating model on validation dataset...\n')
        with torch.no_grad():

            if self.make_plot_predict_vs_target:
                # Initialize arrays for storing predicted values and target values
                self.predicted_values = np.zeros((dataset_size, 1))
                self.target_values = np.zeros((dataset_size, 1))

            for batch_idx, (X, Y) in enumerate(tmp_dataloader):

                X, Y = X.to(self.device), Y.to(self.device)

                # Perform forward pass
                Y_hat = self.model(X)
                
                if self.make_plot_predict_vs_target:
                    # Store predicted values in array for plotting
                    self.predicted_values[batch_idx * batch_size: (batch_idx + 1) * batch_size] = torch_to_numpy(Y_hat)
                    self.target_values[batch_idx * batch_size: (batch_idx + 1) * batch_size] = torch_to_numpy(Y)

                if self.eval_function is not None:
                    # Compute errors per component
                    error_per_component = self.eval_function(Y_hat, Y)
                else:
                    # Assume that error is computed as difference between prediction and target
                    error_per_component = Y_hat - Y

                if self.loss_fcn is not None:
                    # Compute loss for ith batch
                    batch_loss = self.loss_fcn(Y_hat, Y)

                    # Accumulate loss
                    if isinstance(batch_loss, dict):
                        total_loss += batch_loss.get('loss_value') 
                    else:
                        total_loss += batch_loss.item()
    
                # Store residuals
                if residuals is None:
                    residuals = error_per_component
                else:
                    residuals = torch.cat((residuals, error_per_component), dim=0)

                # Print progress
                progress_info = f"\033[93mEvaluating: Batch {batch_idx+1}/{num_batches}\033[0m"
                # Print progress on the same line
                print(progress_info, end='\r')

            print('\n')
            if self.loss_fcn is not None:
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

        if self.loss_fcn is not None:
            self.stats['avg_loss'] = avg_loss

        # Print statistics
        print('\033[95mModel evaluation statistics:\033[0m')
        print(' Mean of residuals: ', meanResiduals)
        print(' Std of residuals: ', stdResiduals)
        print(' Median of residuals: ', medianResiduals)
        print(' Max of residuals: ', maxResiduals)

        return self.stats

    def plotResults(self, entriesNames: list | None = None, 
                    units: list | None = None, 
                    unit_scalings: dict | tuple | float | None = None, 
                    colours: list | None = None, 
                    num_of_bins: int = 100) -> None:
        
        if self.plotter is not None:
            self.plotter.histPredictionErrors(self.stats, 
                                              entriesNames=entriesNames, 
                                              units=units,
                                              unit_scalings=unit_scalings, 
                                              colours=colours, 
                                              num_of_bins=num_of_bins)
        else:
            print('\033[93m' + 'Warning: No plotter object provided. Cannot plot histograms of prediction errors.' + '\033[0m')

        # Predictions vs targets scatter plot
        if self.make_plot_predict_vs_target:
            import matplotlib.pyplot as plt
            import math

            # Make plot of predicted values vs target values
            n_samples, n_outputs = self.predicted_values.shape

            # Make grid layout
            n_cols = int(math.ceil(math.sqrt(n_outputs)))
            n_rows = int(math.ceil(n_outputs / n_cols))

            fig, axes = plt.subplots(n_rows, n_cols,
                                    figsize=(4*n_cols, 4*n_rows),
                                    squeeze=False)

            # For each output dim, make scatter + identity line plot
            targets = self.target_values
            preds = self.predicted_values

            for id_output in range(n_outputs):
                idrow = id_output // n_cols
                idcol = id_output % n_cols

                # Select axis
                ax = axes[idrow][idcol]

                # Scatter of target vs predicted
                ax.scatter(targets[:, id_output], preds[:, id_output], alpha=0.6, edgecolors='none')

                # Draw perfect mean prediction line
                mn = min(targets[:, id_output].min(), preds[:, id_output].min())
                mx = max(targets[:, id_output].max(), preds[:, id_output].max())
                ax.plot([mn, mx], [mn, mx], linestyle='--')

                ax.set_xlabel('Target')
                ax.set_ylabel('Predicted')
                ax.set_title(f'Output #{id_output}')

            # Remove empty subplots
            for j in range(n_outputs, n_rows*n_cols):
                idrow = j // n_cols
                idcol = j % n_cols
                axes[idrow][idcol].axis('off')

            plt.tight_layout()

            # Show if not tmux
            if sys.stdout.isatty():
                plt.show()
            else:
                # Save to file
                plt.savefig('predictions_vs_targets.png', dpi=300, bbox_inches='tight')
                print('Saved predictions vs targets plot to "predictions_vs_targets.png"')
