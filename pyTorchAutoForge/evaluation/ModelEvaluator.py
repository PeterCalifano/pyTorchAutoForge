import torch
import sys
from torch import nn
import numpy as np
from dataclasses import dataclass

from pyTorchAutoForge.utils import torch_to_numpy, numpy_to_torch
from pyTorchAutoForge.datasets import DataloaderIndex
from torch.utils.data import DataLoader
from pyTorchAutoForge.utils.utils import GetDevice
from pyTorchAutoForge.optimization import CustomLossFcn

from collections.abc import Callable
from pyTorchAutoForge.evaluation import ResultsPlotterHelper
from numpy.typing import NDArray

@dataclass
class ModelEvaluatorConfig():
    device = GetDevice()
    # TODO

class ModelEvaluator():
    """
    A class for evaluating PyTorch models.

    This class provides functionality for evaluating regression models, 
    computing statistics, and visualizing results such as prediction errors 
    and predictions vs targets.

    Attributes:
        model (nn.Module): The PyTorch model to evaluate.
        loss_fcn (nn.Module | CustomLossFcn): Loss function used for evaluation.
        validationDataloader (DataLoader): DataLoader providing validation data.
        device (str): Device to run the evaluation on ('cpu' or 'cuda').
        eval_function (Callable | None): Custom evaluation function.
        plotter (ResultsPlotterHelper | None): Helper for plotting results.
        make_plot_predict_vs_target (bool): Whether to plot predictions vs targets.
        output_scale_factors (NDArray | torch.Tensor | None): Scaling factors for outputs.
        stats (dict): Dictionary to store evaluation statistics.
        predicted_values (np.ndarray | None): Predicted values for plotting.
        target_values (np.ndarray | None): Target values for plotting.
    """

    def __init__(self, model: nn.Module, 
                 lossFcn: nn.Module | CustomLossFcn,
                 dataLoader: DataLoader, 
                 device: str = 'cpu', 
                 evalFunction: Callable | None = None,
                 plotter: ResultsPlotterHelper | None = None,
                 make_plot_predict_vs_target: bool = False,
                 output_scale_factors: NDArray[np.generic] | torch.Tensor | None = None) -> None:

        self.loss_fcn = lossFcn
        self.validationDataloader : DataLoader = dataLoader
        self.trainingDataloaderSize : int = len(self.validationDataloader)
        self.eval_function = evalFunction
        self.device = device
        self.output_scale_factors = numpy_to_torch(output_scale_factors).to(self.device) if output_scale_factors is not None else None

        self.make_plot_predict_vs_target = make_plot_predict_vs_target
        self.predicted_values : np.ndarray | None = None
        self.target_values: np.ndarray | None = None 

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
        
        # Create a new dataloader with the same dataset but doubled batch size for speed
        tmp_dataloader = DataLoader(
            original_dataloader.dataset,
            batch_size=original_dataloader.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=False,
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
            for batch_idx, (X, Y) in enumerate(tmp_dataloader):
                X, Y = X.to(self.device), Y.to(self.device)

                # Perform forward pass
                Y_hat = self.model(X)
                
                if self.make_plot_predict_vs_target:

                    if self.predicted_values is None or self.target_values is None:
                        # Initialize arrays for storing predicted values and target values
                        self.predicted_values = np.zeros((dataset_size, *Y_hat.shape[1:]))
                        self.target_values = np.zeros((dataset_size, *Y_hat.shape[1:]))

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
                        if "loss_value" not in batch_loss.keys():
                            raise ValueError("Loss function must return a dictionary with 'loss_value' key or a torchTensor.")
                        
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
            else:
                avg_loss = None

            if residuals is None:
                raise ValueError('Residuals are None. Something has gone wrong during evaluation. Cannot compute statistics.')

            # Scale residuals according to scale factors
            if self.output_scale_factors is not None:
                residuals = residuals * self.output_scale_factors

            # Compute statistics
            mean_residual = torch.mean(residuals, dim=0)
            median_residual, _ = torch.median(residuals, dim=0)
            avg_abs_residual = torch.mean(torch.abs(residuals), dim=0)
            std_residual = torch.std(residuals, dim=0)
            median_abs_residual, _ = torch.median(torch.abs(residuals), dim=0)
            max_abs_residual, _ = torch.max(torch.abs(residuals), dim=0)

        # Move data to numpy
        residuals = torch_to_numpy(residuals)
        mean_residual = torch_to_numpy(mean_residual)
        median_residual = torch_to_numpy(median_residual)
        avg_abs_residual = torch_to_numpy(avg_abs_residual)
        median_abs_residual = torch_to_numpy(median_abs_residual)
        max_abs_residual = torch_to_numpy(max_abs_residual)

        quantile95_residual = np.percentile(residuals, 0.95, axis=0)

        # Pack data into dict
        # TODO replace with dedicated object!
        self.stats = {}
        self.stats['prediction_err']             = residuals
        self.stats['average_abs_prediction_err'] = avg_abs_residual
        self.stats['median_abs_prediction_err']  = median_abs_residual
        self.stats['max_abs_prediction_err']     = max_abs_residual
        self.stats['mean_prediction_err']        = mean_residual 
        self.stats['median_prediction_err']      = median_residual 
        self.stats['std_prediction_err']         = std_residual
        self.stats['quantile95_prediction_err']  = quantile95_residual 
        self.stats['num_samples']                = dataset_size

        if self.loss_fcn is not None:
            self.stats['avg_loss'] = avg_loss

        # Print statistics
        print('\033[95mModel evaluation statistics:\033[0m')
        print(' Mean of prediction errors: ', mean_residual)
        print(' Median of prediction errors: ', median_residual)
        print(' Std of prediction errors: ', std_residual)
        print(' Quantile 95 of prediction errors', quantile95_residual)
        print(' Median of abs prediction errors: ', median_abs_residual)
        print(' Max of abs prediction errors: ', max_abs_residual)

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
