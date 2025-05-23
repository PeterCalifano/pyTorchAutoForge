import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum
import numpy as np
from dataclasses import dataclass, field 
import os
from matplotlib.gridspec import GridSpec

# TODO (PC, important!) review tests and compare with those in tests/. Move or delete these here!

class backend_module(Enum):
    MATPLOTLIB = "Use matplotlib for plotting",
    SEABORN = "Use seaborn for plotting"

@dataclass
class ResultsPlotterConfig():
    save_figs: bool = False
    unit_scalings: dict | None = None
    num_of_bins: int = 100
    colours: list = field(default_factory=list)
    units: tuple[str, ...] | list[str] | None = None
    entriesNames: tuple[str, ...] | list[str] | None = None
    output_folder: str = "eval_results"

class ResultsPlotterHelper():
    """
    ResultsPlotter is a class for plotting the results of prediction errors using different backends like Matplotlib or Seaborn.
    Attributes:
        loaded_stats (dict): Dictionary containing the loaded statistics.
        backend_module (backend_module): The backend module to use for plotting (e.g., Matplotlib or Seaborn).
        stats (dict): Dictionary containing the statistics to be plotted.
        units (list): List of units for each entry.
        colours (list): List of colours for each entry.
        entriesNames (list | None): List of names for each entry.
        unit_scalings (dict): Dictionary containing scaling factors for each entry.
        save_figs (bool | None): Flag to indicate whether to save the figures.
    Methods:
        __init__(stats: dict = None, backend_module_: backend_module = backend_module.SEABORN, config: ResultsPlotterConfig = None) -> None:
            Initializes the ResultsPlotter with the given statistics, backend module, and configuration.
        histPredictionErrors(stats: dict = None, entriesNames: list | None = None, units: list | None = None, unit_scalings: dict | list | np.ndarray | float | int = None, colours: list | None = None, num_of_bins: int = 100) -> None:
            Plots a histogram of prediction errors per component without absolute value. Requires EvaluateRegressor() to be called first and matplotlib to work in Interactive Mode.
    """
    def __init__(self, stats: dict | None = None, 
                 backend_module_: backend_module = backend_module.SEABORN, 
                 config: ResultsPlotterConfig | None = None) -> None:

        self.loaded_stats = stats
        self.backend_module = backend_module_
        self.stats = None

        # Assign all config attributes dynamically TODO change by keeping config attribute!
        if config is None:
            # Define default values
            config = ResultsPlotterConfig()

        for key, value in vars(config).items():
            setattr(self, key, value)

        # Determine whether output folder exists already and if is empty
        folder_id = 0
        tmp_output_folder = self.output_folder + f"_{folder_id}"

        while os.path.isdir(tmp_output_folder):
            if len(os.listdir(tmp_output_folder)) == 0:
                self.output_folder = tmp_output_folder
                print(f"Output folder {tmp_output_folder} exists but is empty. Using it...")
                break
            else:
                folder_id += 1
                tmp_output_folder = tmp_output_folder + f"_{folder_id}"

        # Define the output folder
        self.output_folder = tmp_output_folder

        # Create the output folder if it does not exist
        print(f"Output results will be saved to folder {tmp_output_folder}")
        os.makedirs(self.output_folder, exist_ok=True) if self.output_folder is not None else None

    def histPredictionErrors(self, 
                             stats: dict = None, 
                             entriesNames: list = None, 
                             units: list = None,
                             unit_scalings: dict | list | np.ndarray | float | int = None, 
                             colours: list = None, 
                             num_of_bins: int = 100) -> int:
        """
        Method to plot histogram of prediction errors per component without absolute value. EvaluateRegressor() must be called first.
        Requires matplotlib to work in Interactive Mode.
        """

        if units == None and self.units != None:
            units = self.units

        assert (units is not None if entriesNames is not None else True)
        assert (len(entriesNames) == len(units) if entriesNames is not None else True)

        # DATA: Check if stats dictionary is empty
        if stats == None:
            self.stats == self.loaded_stats
        else:
            self.stats = stats

        if self.stats == None:
            print('Return: empty stats dictionary')
            return -1
        elif not(isinstance(self.stats, dict)):
            raise TypeError("Invalid stats input provided: must be a dictionary.")
        
        
        if 'prediction_err' in self.stats:
            prediction_errors = self.stats['prediction_err']
        else:
            print('Return: "prediction_err" key not found in stats dictionary')
            return -1
        
        if 'mean_prediction_err' in self.stats:
            mean_errors = self.stats['mean_prediction_err']
        else:
            mean_errors = None

        # Assumes that the number of entries is always smaller that the number of samples
        num_of_entries = min(prediction_errors.shape) 
        grid_cols = int(np.ceil(np.sqrt(num_of_entries)))
        grid_rows = int(np.ceil(num_of_entries / grid_cols))

        # Create combined figure and axes
        fig, axes = plt.subplots(grid_rows, 
                                 grid_cols, 
                                 figsize=(grid_cols * 10, grid_rows * 10))
        axes = axes.flatten()

        # COLOURS: Check that number of colours is equal to number of entries
        if colours is not None:
            override_condition = len(colours) < num_of_entries if colours is not None else False and len(self.colours) < num_of_entries

            if override_condition:
                Warning( "Overriding colours: number of colours specified not matching number of entries.")
                colours = None
        else:
            override_condition = False

        if colours == None and self.colours != [] and not(override_condition):
            colours = self.colours

        elif (colours == None and self.colours == []) or override_condition:
            if self.backend_module == backend_module.MATPLOTLIB:
                # Get colour palette from matplotlib
                colours = plt.cm.get_cmap('viridis', num_of_entries)

            elif self.backend_module == backend_module.SEABORN:
                # Get colour palette from seaborn
                colours = sns.color_palette("husl", num_of_entries)
            else:
                raise ValueError("Invalid backend module selected.")

        if unit_scalings is not None:
            unit_scalers_ = unit_scalings
        elif self.unit_scalings is not None:
            unit_scalers_ = self.unit_scalings
        else:
            unit_scalers_ = None

        # PLOT: Plotting loop per component
        for idEntry in np.arange(num_of_entries):

            # ENTRY NAME: Check if entry name is provided
            if entriesNames != None:
                entryName = entriesNames[idEntry]
            elif self.entriesNames != None:
                entryName = self.entriesNames[idEntry]
            else:
                entryName = "Component " + str(idEntry)

            # SCALING: Check if scaling required
            if isinstance(unit_scalers_, dict):
                if entryName in unit_scalers_:
                    unit_scaler = unit_scalers_[entryName]
                else:
                    raise ValueError("Entry name not found in unit_scalings dict.")

            elif isinstance(unit_scalers_, (int, float)):
                unit_scaler = unit_scalers_

            elif isinstance(unit_scalers_, (list, np.ndarray)):
                unit_scaler = unit_scalers_[idEntry]

            elif unit_scalers_ is None or unit_scalers_ == {}:
                # Set to one if None or empty
                unit_scaler = 1.0
            else:
                raise ValueError(f"Expected unit_scalings to be a dictionary, a list, a np.ndarray or a scalar, but found {type(unit_scalers_)}. Check input first. If the issue persists, please report it.")

            # Select axis and data
            ax = axes[idEntry]
            data = prediction_errors[:, idEntry] * unit_scaler

            if self.backend_module == backend_module.MATPLOTLIB:
                ax.hist(data,
                         bins=num_of_bins, 
                         color=colours[idEntry], alpha=0.8, 
                         edgecolor='black', 
                         label=entryName)
        
            elif self.backend_module == backend_module.SEABORN:
                sns.histplot(data,
                            bins=num_of_bins, 
                            color=colours[idEntry], 
                            kde=True, 
                            ax=ax)

            # Add average error if available
            if mean_errors is not None:
                mean_val = mean_errors[idEntry] * unit_scaler
                ax.axvline(mean_val,
                            color=colours[idEntry], 
                            linestyle='--', 
                            linewidth=1, 
                            label=f'Mean: {mean_errors[idEntry]:.2f}')

            # Labels and title
            ax.set_title(entryName)
            unit_label = units[idEntry] if (entriesNames is not None or self.entriesNames is not None) else 'N/D'
            ax.set_xlabel(f"Error [{unit_label}]")
            ax.set_ylabel("# Samples")
            ax.grid(True)
            
            # Plot legend if labels are added to data
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                ax.legend()

        # Remove unused subplots if any
        for idx in range(num_of_entries, len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()

        # Save or show combined figure
        if self.save_figs:
            output_dir = self.output_folder # Set output folder

            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, 'prediction_errors_all_components.png')
            fig.savefig(save_path, bbox_inches='tight')
        else:
            plt.show()

# %% TEST CASE
def setup_plotter():
    stats = {
        'prediction_err': np.random.randn(100, 3),
        'mean_prediction_err': np.random.randn(3)
    }
    config = ResultsPlotterConfig(
        save_figs=False,
        unit_scalings={'Component 0': 1.0,
                       'Component 1': 2.0, 'Component 2': 3.0},
        num_of_bins=50,
        colours=['red', 'green', 'blue'],
        units=['unit1', 'unit2', 'unit3'],
        entriesNames=['Component 0', 'Component 1', 'Component 2'],
        output_folder='test_output'
    )
    plotter = ResultsPlotterHelper(
        stats=stats, backend_module_=backend_module.SEABORN, config=config)
    if os.path.isdir('test_output'):
        for file in os.listdir('test_output'):
            os.remove(os.path.join('test_output', file))
        os.rmdir('test_output')
    return plotter, stats, config

def test_initialization(setup_plotter):
    plotter, stats, config = setup_plotter
    assert plotter.loaded_stats == stats
    assert plotter.backend_module == backend_module.SEABORN
    assert plotter.save_figs == config.save_figs
    assert plotter.unit_scalings == config.unit_scalings
    assert plotter.num_of_bins == config.num_of_bins
    assert plotter.colours == config.colours
    assert plotter.units == config.units
    assert plotter.entriesNames == config.entriesNames
    assert plotter.output_folder == config.output_folder

def test_histPredictionErrors(setup_plotter):
    plotter, _, _ = setup_plotter
    plotter.histPredictionErrors()

def test_histPredictionErrors_with_custom_stats(setup_plotter):
    plotter, _, _ = setup_plotter
    custom_stats = {
        'prediction_err': np.random.randn(100, 3),
        'mean_prediction_err': np.random.randn(3)
    }
    plotter.histPredictionErrors(stats=custom_stats)

def test_histPredictionErrors_with_invalid_stats(setup_plotter):
    plotter, _, _ = setup_plotter

def test_histPredictionErrors_with_missing_prediction_err(setup_plotter):
    plotter, _, _ = setup_plotter
    incomplete_stats = {'mean_prediction_err': np.random.randn(3)}
    try:
        plotter.histPredictionErrors(stats=incomplete_stats)
    except Exception as e:
        assert e == "prediction_err key not found in stats dictionary"

def test_histPredictionErrors_with_missing_mean_prediction_err(setup_plotter):
    plotter, _, _ = setup_plotter
    incomplete_stats = {'prediction_err': np.random.randn(100, 3)}
    plotter.histPredictionErrors(stats=incomplete_stats)


def test_save_figs(setup_plotter):
    plotter, _, config = setup_plotter
    plotter.save_figs = True
    plotter.output_folder = 'test_output'
    flag = plotter.histPredictionErrors()

    if flag != -1:
        for entry in config.entriesNames:
            assert os.path.isfile(os.path.join(
                'test_output', f"prediction_errors_{entry}.png"))
    else:
        print('No image saved since input is empty.')


def test_debug_resultsPlotter():

    plotter, stats, config = setup_plotter()
    assert plotter.loaded_stats == stats
    assert plotter.backend_module == backend_module.SEABORN
    assert plotter.save_figs == config.save_figs
    assert plotter.unit_scalings == config.unit_scalings
    assert plotter.num_of_bins == config.num_of_bins
    assert plotter.colours == config.colours
    assert plotter.units == config.units
    assert plotter.entriesNames == config.entriesNames
    assert plotter.output_folder == config.output_folder

    test_histPredictionErrors(setup_plotter())
    test_histPredictionErrors_with_custom_stats(setup_plotter())
    test_histPredictionErrors_with_invalid_stats(setup_plotter())
    test_histPredictionErrors_with_missing_prediction_err(setup_plotter())
    test_histPredictionErrors_with_missing_mean_prediction_err(setup_plotter())
    test_save_figs(setup_plotter())

if __name__ == '__main__':
    test_debug_resultsPlotter()

