import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum
import numpy as np
from dataclasses import dataclass, field 
import os

class backend_module(Enum):
    MATPLOTLIB = "Use matplotlib for plotting",
    SEABORN = "Use seaborn for plotting"

@dataclass
class ResultsPlotterConfig():
    save_figs: bool = False
    unit_scalings: dict = field(default_factory=dict)
    num_of_bins: int = 100
    colours: list = field(default_factory=list)
    units: list = None
    entriesNames: list = None
    output_folder: str = None

class ResultsPlotter():
    """
    ResultsPlotter is a class for plotting the results of prediction errors using different backends like Matplotlib or Seaborn.
    Attributes:
        loaded_stats (dict): Dictionary containing the loaded statistics.
        backend_module (backend_module): The backend module to use for plotting (e.g., Matplotlib or Seaborn).
        stats (dict): Dictionary containing the statistics to be plotted.
        units (list): List of units for each entry.
        colours (list): List of colours for each entry.
        entriesNames (list): List of names for each entry.
        unit_scalings (dict): Dictionary containing scaling factors for each entry.
        save_figs (bool): Flag to indicate whether to save the figures.
    Methods:
        __init__(stats: dict = None, backend_module_: backend_module = backend_module.SEABORN, config: ResultsPlotterConfig = None) -> None:
            Initializes the ResultsPlotter with the given statistics, backend module, and configuration.
        histPredictionErrors(stats: dict = None, entriesNames: list = None, units: list = None, unit_scalings: dict = None, colours: list = None, num_of_bins: int = 100) -> None:
            Plots a histogram of prediction errors per component without absolute value. Requires EvaluateRegressor() to be called first and matplotlib to work in Interactive Mode.
    """
    def __init__(self, stats: dict = None, backend_module_: backend_module = backend_module.SEABORN, 
                 config: ResultsPlotterConfig = None) -> None:

        self.loaded_stats = stats
        self.backend_module = backend_module_
        self.stats = None

        # Assign all config attributes dynamically
        if config is None:
            # Define default values
            config = ResultsPlotterConfig()

        for key, value in vars(config).items():
            setattr(self, key, value)
    
    def histPredictionErrors(self, stats: dict = None, entriesNames: list = None, units: list = None,
                             unit_scalings: dict = None, colours: list = None, num_of_bins: int = 100) -> None:
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
            return
        elif not(isinstance(self.stats, dict)):
            raise TypeError("Invalid stats input provided: must be a dictionary.")
        
        
        if 'prediction_err' in self.stats:
            prediction_errors = self.stats['prediction_err']
        else:
            print('Return: "prediction_err" key not found in stats dictionary')
            return
        
        if 'mean_prediction_err' in self.stats:
            mean_errors = self.stats['mean_prediction_err']
        else:
            mean_errors = None

        # Assumes that the number of entries is always smaller that the number of samples
        num_of_entry = min(prediction_errors.shape) 

        # COLOURS: Check that number of colours is equal to number of entries
        if colours != None:
            override_condition = len(colours) < num_of_entry if colours != None else False and len(self.colours) < num_of_entry

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
                colours = plt.cm.get_cmap('viridis', num_of_entry)

            elif self.backend_module == backend_module.SEABORN:
                # Get colour palette from seaborn
                colours = sns.color_palette("husl", num_of_entry)
            else:
                raise ValueError("Invalid backend module selected.")
            

        # PLOT: Plotting loop per component
        for idEntry in np.arange(num_of_entry):
            
            # ENTRY NAME: Check if entry name is provided
            if entriesNames != None:
                entryName = entriesNames[idEntry]
            elif self.entriesNames != None:
                entryName = self.entriesNames[idEntry]
            else:
                entryName = "Component " + str(idEntry)

            # SCALING: Check if scaling required
            if unit_scalings != None and entryName in unit_scalings:
                unit_scaler = unit_scalings[entryName]
            elif self.unit_scalings != None and entryName in self.unit_scalings:
                unit_scaler = self.unit_scalings[entryName]
            else:
                unit_scaler = 1.0

            # Define figure and title
            plt.figure(idEntry)
            plt.title("Histogram of errors: " + entryName)

            if self.backend_module == backend_module.MATPLOTLIB:
                plt.hist(prediction_errors[:, idEntry] * unit_scaler,
                         bins=num_of_bins, color=colours[idEntry], alpha=0.8, 
                         edgecolor='black', label=entryName)
        
            elif self.backend_module == backend_module.SEABORN:
                sns.displot(prediction_errors[:, idEntry] * unit_scaler, 
                            bins=num_of_bins, color=colours[idEntry], rug=True, 
                            kde=True, kind='hist')

            # Add average error if available
            if mean_errors is not None:
                plt.axvline(mean_errors[idEntry] * unit_scaler,
                            color=colours[idEntry], linestyle='--', linewidth=1, 
                            label=f'Mean: {mean_errors[idEntry]:.2f}')

            plt.xlabel("Error [{unit}]".format(unit=units[idEntry] if (
                entriesNames != None or self.entriesNames != None) else "N/D"))
            plt.ylabel("# Samples")
            plt.grid()
            plt.tight_layout()

            # SAVING: Save figure if required
            if self.save_figs:
                if self.output_folder is not None:
                    output_dir_path = self.output_folder
                    if os.path.isdir(output_dir_path):
                        os.makedirs(output_dir_path, exist_ok=False)
                else:
                    output_dir_path = "."

                plt.savefig(os.path.join(output_dir_path, "prediction_errors_" + entryName + ".png"), bbox_inches='tight')
            else:
                plt.show()

