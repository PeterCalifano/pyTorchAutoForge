from matplotlib.ticker import MultipleLocator
import torch
import sys
import os
from torch import nn
import numpy as np
from dataclasses import dataclass

from pyTorchAutoForge.utils import torch_to_numpy, numpy_to_torch
from pyTorchAutoForge.datasets import DataloaderIndex
from torch.utils.data import DataLoader
from pyTorchAutoForge.utils import GetDevice, GetDeviceMulti
from pyTorchAutoForge.optimization import CustomLossFcn

from collections.abc import Callable
from pyTorchAutoForge.evaluation import ResultsPlotterHelper
from numpy.typing import NDArray
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns


@dataclass
class ModelEvaluatorConfig():
    device : torch.device | str | None = None
    # TODO

# TODO (PC) rework this class. Not general enough, hint types are to review, constrain more.


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
                 plotter: ResultsPlotterHelper,
                 device: torch.device | str | None = None,
                 custom_eval_fcn: Callable | None = None,
                 make_plot_predict_vs_target: bool = False,
                 output_scale_factors: NDArray[np.generic] | torch.Tensor | None = None,
                 augmentation_module: nn.Module | None = None,
                 indep_variable_boxplot: np.ndarray | None = None,
                 indep_variable_boxplot_label: str = "Independent variable #0",
                 indep_variable_dependenceplot: np.ndarray | None = None,
                 indep_variable_dependenceplot_label: str = "Independent variable #0") -> None:

        # Evaluator attributes
        self.loss_fcn = lossFcn
        self.validationDataloader: DataLoader = dataLoader
        self.custom_eval_function = custom_eval_fcn
        self.device = device if device is not None else GetDeviceMulti()
        self.model = model.to(self.device)
        self.stats: dict = {}
        self.plotter = plotter
        self.augmentation_module = augmentation_module

        if not isinstance(self.plotter, ResultsPlotterHelper):
            raise TypeError(
                "plotter must be an instance of ResultsPlotterHelper")

        # Store variables for additional plots
        self.indep_variable_boxplot = indep_variable_boxplot
        self.indep_variable_boxplot_label = indep_variable_boxplot_label
        self.indep_variable_dependenceplot = indep_variable_dependenceplot
        self.indep_variable_dependenceplot_label = indep_variable_dependenceplot_label

        # Flag to determine plots
        self.make_plot_predict_vs_target = make_plot_predict_vs_target

        # Storage attributes
        self.predicted_values: np.ndarray | None = None
        self.target_values: np.ndarray | None = None

        # Determine scale factors
        self.output_scale_factors: NDArray[np.generic] | torch.Tensor | None = None

        if output_scale_factors is not None:
            print("Using provided output scale factors for stats computation...")
            self.output_scale_factors = numpy_to_torch(
                output_scale_factors).to(self.device)

        elif self.plotter is not None:
            if self.plotter.unit_scalings is not None:

                scalings_ = np.asarray(
                    [self.plotter.unit_scalings[key] for key in self.plotter.unit_scalings])

                print(
                    "Using output scale factors in plotter object for stats computation...")
                self.output_scale_factors = numpy_to_torch(
                    scalings_).to(self.device)
        else:
            print("No output scale factors provided. Using default scale factors of 1.0.")

        if self.plotter is not None and output_scale_factors is not None:
            if self.plotter.unit_scalings is not None:
                print('\033[93mWarning: Overriding unit scalings in plotter with output scale factors as they would result in double application when plotting. Modify input settings to remove this warning.\033[0m')
                # Override plotter.unit_scalings to 1.0
                self.plotter.unit_scalings = {
                    k: 1.0 for k in self.plotter.unit_scalings.keys()}

        # Define base output folder
        self.base_output_folder = self.plotter.output_folder

        # Make directory if not exists
        os.makedirs(self.base_output_folder, exist_ok=True)

    def evaluateRegressor(self) -> dict:
        """
        evaluateRegressor _summary_

        _extended_summary_

        :raises ValueError: _description_
        :raises ValueError: _description_
        :raises ValueError: _description_
        :raises ValueError: _description_
        :return: _description_
        :rtype: dict
        """

        self.model.eval()

        # Backup the original batch size (TODO: TBC if it is useful)
        original_dataloader = self.validationDataloader

        if self.validationDataloader is None:
            raise ValueError(
                'Validation dataloader is None. Cannot evaluate model.')

        if self.validationDataloader.batch_size is None:
            raise ValueError(
                'Batch size of dataloader is None. Cannot evaluate model.')

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
        batch_size: int = tmp_dataloader.batch_size
        residuals = None

        # Perform model evaluation on all batches
        total_loss = 0.0
        print('\nEvaluating model on validation dataset...\n')
        with torch.no_grad():
            for batch_idx, (X, Y) in enumerate(tmp_dataloader):
                X, Y = X.to(self.device), Y.to(self.device)

                # Optional augmentation module
                if self.augmentation_module is not None:
                    Y = Y * self.output_scale_factors

                    # Apply augmentations
                    X, Y = self.augmentation_module(X, Y)

                    # Rescale labels down to [0,1] # DOUBT this breaks everything? Why?
                    #Y = Y / torch.Tensor(self.output_scale_factors).to(self.device)


                # Perform forward pass
                Y_hat = self.model(X)

                if self.make_plot_predict_vs_target:

                    if self.predicted_values is None or self.target_values is None:
                        # Initialize arrays for storing predicted values and target values
                        self.predicted_values = np.zeros(
                            (dataset_size, *Y_hat.shape[1:]))
                        self.target_values = np.zeros(
                            (dataset_size, *Y_hat.shape[1:]))

                    # Store predicted values in array for plotting
                    self.predicted_values[batch_idx * batch_size: (
                        batch_idx + 1) * batch_size] = torch_to_numpy(Y_hat)
                    self.target_values[batch_idx * batch_size: (
                        batch_idx + 1) * batch_size] = torch_to_numpy(Y)

                if self.custom_eval_function is not None:
                    # Compute errors per component
                    error_per_component = self.custom_eval_function(Y_hat, Y)
                else:
                    # Assume that error is computed as difference between prediction and target
                    error_per_component = Y_hat - Y

                if self.loss_fcn is not None:
                    # Compute loss for ith batch
                    batch_loss = self.loss_fcn(Y_hat, Y)

                    # Accumulate loss
                    if isinstance(batch_loss, dict):
                        if "loss_value" not in batch_loss.keys():
                            raise ValueError(
                                "Loss function must return a dictionary with 'loss_value' key or a torchTensor.")

                        total_loss += batch_loss.get('loss_value')
                    else:
                        total_loss += batch_loss.item()

                # Store residuals
                if residuals is None:
                    residuals = error_per_component
                else:
                    residuals = torch.cat(
                        (residuals, error_per_component), dim=0)

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
                raise ValueError(
                    'Residuals are None. Something has gone wrong during evaluation. Cannot compute statistics.')

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
        std_residual = torch_to_numpy(std_residual)

        quantile68_residual = np.quantile(np.abs(residuals), 0.68, axis=0)
        quantile95_residual = np.quantile(np.abs(residuals), 0.95, axis=0)
        quantile99p7_residual = np.quantile(np.abs(residuals), 0.997, axis=0)

        # Pack data into dict
        # TODO replace with dedicated object!
        self.stats = {}
        # Errors with sign
        self.stats['prediction_err'] = residuals
        self.stats['mean_prediction_err'] = mean_residual
        self.stats['median_prediction_err'] = median_residual
        self.stats['std_prediction_err'] = std_residual
        # Absolute errors and quantiles
        self.stats['average_abs_prediction_err'] = avg_abs_residual
        self.stats['median_abs_prediction_err'] = median_abs_residual
        self.stats['quant68_abs_prediction_err'] = quantile68_residual
        self.stats['quant95_abs_prediction_err'] = quantile95_residual
        self.stats['quant99p7_abs_prediction_err'] = quantile99p7_residual
        self.stats['max_abs_prediction_err'] = max_abs_residual
        self.stats['num_samples'] = dataset_size

        error_labels = [f"Output {i}" for i in range(residuals.shape[1])]

        if self.plotter.entriesNames is not None:
            error_labels = self.plotter.entriesNames

        if self.plotter.units is not None:
            error_labels = [f"{label} ({unit})" for label, unit in zip(
                error_labels, self.plotter.units)]

        # TODO (PC) come back here when changed plotter. Some fields are assigned dynamically and mypy fails to detect those
        self.stats["error_labels"] = tuple(error_labels)[:residuals.shape[1]]

        if self.loss_fcn is not None:
            self.stats['avg_loss'] = avg_loss

        self.printAndSaveStats(self.stats,
                               output_folder=self.base_output_folder)

        return self.stats

    def printAndSaveStats(self, stats: dict, output_folder: str = "."):
        """
        Prints evaluation statistics in a Markdown table and saves them as a JSON file.

        Args:
            stats (dict): Dictionary containing evaluation statistics, including error metrics and labels.
            output_folder (str, optional): Directory to save the statistics file. Defaults to ".".

        """

        # Build output table of stats
        num_entries = len(stats["error_labels"])
        vector_stats = {k: v for k, v in stats.items()
                        if hasattr(v, "__len__") and len(v) == num_entries}

        # Build DataFrame: rows=vector_stats keys, cols=labels
        df = pd.DataFrame.from_dict(
            vector_stats, orient="index", columns=stats["error_labels"])

        # Print Markdown table (rounded)
        print("\n")
        print(df.round(2).to_markdown(tablefmt="github"))
        print("\n")
        # Save CSV / JSON / Excel
        # df.to_csv(f"{out_prefix}.csv", index=False)

        df.to_json(os.path.join(output_folder, "eval_stats.json"),
                   orient="index", indent=2)

        # df.to_excel(f"{out_prefix}.xlsx", index=False)
        # print(f"\n CSV, JSON, XLSX saved as '{out_prefix}.*'")

    def makeOutputPlots(self, entriesNames: list | None = None,
                        units: list | None = None,
                        unit_scalings: dict | tuple | float | None = None,
                        colours: list | None = None,
                        num_of_bins: int = 100) -> None:
        """
         _summary_

        _extended_summary_
        """

        # Plot histograms of prediction errors
        self.plotter.histPredictionErrors(self.stats,
                                          entriesNames=entriesNames,
                                          units=units,
                                          unit_scalings=unit_scalings,
                                          colours=colours,
                                          num_of_bins=num_of_bins)

        # Predictions vs targets scatter plot
        n_samples, n_outputs = self.predicted_values.shape

        # Define grid layout sizes
        n_cols = int(math.ceil(math.sqrt(n_outputs)))
        n_rows = int(math.ceil(n_outputs / n_cols))

        targets = self.target_values
        preds = self.predicted_values
        errors = preds - targets
        # TODO add check on values, must be assigned before this code can run!

        try:
            # Determine scale factors
            if unit_scalings is not None:
                # Handle unit_scalings according to type
                if isinstance(unit_scalings, dict):
                    unit_scale_values = np.asarray(
                        [unit_scalings[key] for key in unit_scalings])
                elif isinstance(unit_scalings, (tuple, list, np.ndarray)):
                    unit_scale_values = np.asarray(unit_scalings)
                else:
                    unit_scale_values = unit_scalings  # Assume scalar

            elif self.output_scale_factors is not None:
                unit_scale_values = torch_to_numpy(
                    self.output_scale_factors)

            elif self.plotter.unit_scalings is not None:
                unit_scale_values = np.asarray(
                    [self.plotter.unit_scalings[key] for key in self.plotter.unit_scalings])
            else:
                print(
                    '\033[93mNo unit scalings found. Using default scale factors of 1.0.\033[0m')
                unit_scale_values = None

        except AttributeError:
            print(
                '\033[93mNo unit scalings provided. Using default scale factors of 1.0.\033[0m')
            unit_scale_values = None

        # Scale each size of errors if unit_scalings are provided
        if unit_scale_values is not None:
            preds *= unit_scale_values
            targets *= unit_scale_values
            errors *= unit_scale_values

        if preds is None or targets is None:
            raise ValueError(
                'Predicted values or target values are None. Cannot plot results.')

        # OPTIONAL PLOTS
        # Optional predicted vs target scatter plot
        # TODO replace with better plot using seaborn!
        if self.make_plot_predict_vs_target:
            # Make plot of predicted values vs target values
            fig_vs_plot, axes_vs_plot = plt.subplots(n_rows, n_cols,
                                                     figsize=(
                                                         10*n_cols, 10*n_rows),
                                                     squeeze=False)

            # For each output dim, make scatter + identity line plot
            for id_output in range(n_outputs):
                idrow = id_output // n_cols
                idcol = id_output % n_cols

                # Select axis
                ax = axes_vs_plot[idrow][idcol]

                # Scatter of target vs predicted
                ax.scatter(targets[:, id_output], preds[:,
                           id_output], alpha=0.6, edgecolors='none')

                # Draw perfect mean prediction line
                mn = min(targets[:, id_output].min(),
                         preds[:, id_output].min())
                mx = max(targets[:, id_output].max(),
                         preds[:, id_output].max())

                ax.plot([mn, mx], [mn, mx], linestyle='--',
                        color='red', linewidth=2)

                ax.set_xlabel('Target')
                ax.set_ylabel('Predicted')
                ax.set_title(f'Output #{id_output}')

            # Remove empty subplots
            for j in range(n_outputs, n_rows*n_cols):
                idrow = j // n_cols
                idcol = j % n_cols
                axes_vs_plot[idrow][idcol].axis('off')

            # Apply layout settings
            plt.tight_layout()

            if self.plotter.save_figs or not sys.stdout.isatty():
                # Save to file
                plt.savefig(os.path.join(self.base_output_folder,
                            'predictions_vs_targets.png'), dpi=300, bbox_inches='tight')

        # Optional box plot of the errors against independent variable
        if self.indep_variable_boxplot is not None:
            # Check variable (batch) size is correct
            if self.indep_variable_boxplot.shape[0] != n_samples:
                print(
                    f"\033[93mVariable to plot against has batch size {self.indep_variable_boxplot.shape[0]}, but expected {n_samples}.\033[0m")
            else:
                # Do plot for each output
                variable_to_plot_against_label_ = self.indep_variable_boxplot_label
                num_bins = 25  # TODO allow to change this setting

                # Compute binned means and standard deviations
                bins = np.linspace(self.indep_variable_boxplot.min(
                ), self.indep_variable_boxplot.max(), num_bins + 1)
                bin_centers = 0.5 * (bins[:-1] + bins[1:])

                binned_stats_df = pd.DataFrame({
                    'x': np.repeat(self.indep_variable_boxplot, preds.shape[1]),
                    'output': np.tile(np.arange(preds.shape[1]), len(self.indep_variable_boxplot)),
                    'Prediction error': (errors).ravel()
                })

                # Cut into bins, but label by the numeric center
                binned_stats_df['bin'] = pd.cut(
                    binned_stats_df['x'], bins=bins, labels=bin_centers)

                # Make axes
                fig_against_plot, axes_against_plot = plt.subplots(
                    n_rows, n_cols, figsize=(12*n_cols, 12*n_rows), squeeze=False)

                for id_output in range(n_outputs):
                    idrow = id_output // n_cols
                    idcol = id_output % n_cols

                    # Select axis and binned values
                    ax = axes_against_plot[idrow][idcol]
                    sub = binned_stats_df[binned_stats_df['output']
                                          == id_output]

                    # Make box plot
                    sns.set_style("whitegrid")

                    # Boxplot per bin
                    sns.boxplot(x='bin', y='Prediction error',
                                data=sub,
                                ax=ax,
                                color='navy',
                                showcaps=True,
                                showfliers=True,
                                whis=(0.1, 99),
                                boxprops={
                                    'alpha': 0.5, 'color': 'navy', 'facecolor': 'lightblue'},
                                medianprops={'color': 'navy'},
                                whiskerprops={
                                    'color': 'navy', "linewidth": 2.0},
                                capprops={'color': 'navy'},
                                flierprops={
                                    'marker': 'x',
                                    'markerfacecolor': 'crimson',
                                    'markeredgecolor': 'crimson',
                                    'markersize': 6,
                                    'alpha': 0.8
                                })

                    # Raw‐error scatter, jittered horizontally so you see point cloud
                    sns.stripplot(x='bin', y='Prediction error',
                                  data=sub,
                                  ax=ax,
                                  color='orange',
                                  size=2.5,
                                  alpha=0.6,
                                  jitter=0.2)

                    # Set ticks and grid
                    # ax.yaxis.set_major_locator(MultipleLocator(0.5))
                    # ax.yaxis.set_minor_locator(MultipleLocator(0.25))
                    ax.minorticks_on()
                    ax.grid(which='major', axis='y',
                            linestyle='--', linewidth=0.8,  alpha=0.8)
                    ax.grid(which='minor', axis='y', linestyle=':',
                            linewidth=0.6,  alpha=0.5)

                    ax.set_xlabel(variable_to_plot_against_label_)

                    if units is not None:
                        ax.set_ylabel(
                            f'Prediction error [{units[id_output]}]')
                    else:
                        ax.set_ylabel('Prediction error [-]')

                    # Make x‐labels
                    try:
                        tick_labels = [np.round(c, 1) for c in bin_centers]
                    except TypeError:
                        tick_labels = [f"{c:.2f}" for c in bin_centers]

                    ax.set_xticklabels(tick_labels, rotation=60)
                    ax.set_title(f'Output #{id_output}')

                plt.tight_layout()
                if self.plotter.save_figs or not sys.stdout.isatty():
                    # Save to file
                    plt.savefig(os.path.join(self.base_output_folder,
                                'error_boxplot.png'), dpi=300, bbox_inches='tight')

        # Optional dependence plot of the errors against independent variable
        if self.indep_variable_dependenceplot is not None:
            # Build a 2xN figure
            fig_dependence_plot, axes_dependence_plot = plt.subplots(
                2, n_outputs, figsize=(16*n_outputs, 12*2), squeeze=False)

            # Check variable (batch) size is correct
            if self.indep_variable_dependenceplot.shape[0] != n_samples:
                print(
                    f"\033[93mVariable to plot against has batch size {self.indep_variable_dependenceplot.shape[0]}, but expected {n_samples}.\033[0m")
            else:
                for id_output in range(n_outputs):

                    # Top plot: Predicted vs Target
                    ax_top = axes_dependence_plot[0, id_output]
                    sns.set_theme(style="whitegrid")

                    # Prepare DataFrame for top jointplot: predictions and targets vs indep_var
                    df_top = pd.DataFrame({
                        'indep_var': self.indep_variable_dependenceplot.ravel(),
                        'Predicted': preds[:, id_output],
                        'Target': targets[:, id_output]
                    })

                    # Melt the DataFrame to facilitate grouping by type
                    # DOUBT what does melt do in practice?
                    df_top_melt = df_top.melt(id_vars='indep_var',
                                              value_vars=[
                                                  'Predicted', 'Target'],
                                              var_name='Type',
                                              value_name='Value')

                    sns.scatterplot(data=df_top_melt,
                                    x="indep_var",
                                    y="Value",
                                    hue="Type",
                                    ax=ax_top,
                                    edgecolor=(0, 0, 0, 0.85),
                                    alpha=0.9,
                                    )

                    ax_top.set_title(
                        f'Output #{id_output} Predictions vs Targets')
                    ax_top.set_xlabel(self.indep_variable_dependenceplot_label)
                    ax_top.set_ylabel('Value')
                    ax_top.legend(loc='best')
                    ax_top.grid(True, which='both',
                                linestyle='--', linewidth=0.6)
                    ax_top.minorticks_on()

                    # Bottom plot: Errors vs indep_var
                    ax_bottom = axes_dependence_plot[1, id_output]
                    sns.set_theme(style="whitegrid")

                    # Prepare DataFrame for bottom jointplot: errors vs indep_var
                    df_bottom = pd.DataFrame({
                        'indep_var': self.indep_variable_dependenceplot.ravel(),
                        'error': errors[:, id_output]
                    })

                    # Add a LOWESS smooth line using regplot on the joint axes
                    sns.regplot(x='indep_var',
                                y='error',
                                data=df_bottom,
                                scatter=True,
                                lowess=True,
                                ax=ax_bottom,
                                line_kws={'color': 'blue', 
                                          'lw': 1.2,
                                          'label': 'LOWESS fit'},
                                scatter_kws={'alpha': 0.9,
                                             'color': 'red',
                                             's': 12,
                                             'edgecolor': (0,0,0,0.4)})

                    ax_bottom.axhline(0, linestyle='--', color='black')
                    ax_bottom.set_title(
                        f'Output #{id_output} Prediction Errors')
                    ax_bottom.set_xlabel(
                        self.indep_variable_dependenceplot_label)
                    ax_bottom.set_ylabel('Error')
                    ax_bottom.legend(loc='best')
                    ax_bottom.grid(True, which='both',
                                   linestyle='--', linewidth=0.5)
                    ax_bottom.minorticks_on()

                # Setup figure layout and save
                plt.tight_layout()
                if self.plotter.save_figs or not sys.stdout.isatty():
                    fig_dependence_plot.savefig(
                        os.path.join(
                            self.base_output_folder,
                            f'dependence_plot_all_outputs.png'
                        ),
                        dpi=300,
                        bbox_inches='tight'
                    )

        # Show all figures if not tmux and interactive backend
        if sys.stdout.isatty() and plt.get_backend().lower() != 'agg':
            plt.show()

    def makeSamplePredictions(self):
        pass