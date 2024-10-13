import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum
import numpy as np

class backend_module(Enum):
    MATPLOTLIB = "Use matplotlib for plotting",
    SEABORN = "Use seaborn for plotting"

class ResultsPlotter():
    def __init__(self, stats: dict = {}, backend_module_: backend_module = backend_module.SEABORN, save_figs: bool = False) -> None:
        self.stats = stats # Shallow copy
        self.backend_module = backend_module_

    def HistPredictionErrors(self, entriesNames: list = None, units: list = None, 
                            unit_scalings: dict = {}, num_of_bins: int = 100, colours: list = []) -> None:
        """
        Method to plot histogram of prediction errors per component without absolute value. EvaluateRegressor() must be called first.
        Requires matplotlib to work in Interactive Mode.
        """
        assert (units is not None if entriesNames is not None else True)
        assert (len(entriesNames) == len(units) if entriesNames is not None else True)

        if self.stats == {}:
            print('Interrupt: empty statistics dictionary')
            return

        prediction_errors = self.stats['prediction_err']

        if 'average_prediction_err' in self.stats:
            avg_errors = self.stats['average_prediction_err']
        else:
            avg_errors = None

        num_of_entry = min(prediction_errors.shape) # Assumes that the number of entries is always smaller that the number of samples


        # COLOURS: Check that number of colours is equal to number of entries
        if len(colours) < num_of_entry:
            Warning("Overriding colours: number of colours specified not matching number of entries.")
            colours = []
        
        if colours == []:
            if self.backend_module == backend_module.MATPLOTLIB:
                # Get colour palette from matplotlib
                colours = plt.cm.get_cmap('viridis', num_of_entry)

            elif self.backend_module == backend_module.SEABORN:
                # Get colour palette from seaborn
                colours = sns.color_palette("husl", num_of_entry)


        # PLOT: Plotting loop per component
        for idEntry in np.arange(num_of_entry):
            
            # ENTRY NAME: Check if entry name is provided
            if entriesNames is not None:
                entryName = entriesNames[idEntry]
            else:
                entryName = "Component " + str(idEntry)

            # SCALING: Check if scaling required
            if entryName in unit_scalings:
                unit_scaler = unit_scalings[entryName]
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
            if avg_errors is not None:
                plt.axvline(avg_errors[idEntry] * unit_scaler,
                            color=colours[idEntry], linestyle='--', linewidth=1, 
                            label=f'Mean: {avg_errors[idEntry]:.2f}')

            plt.xlabel("Error [{unit}]".format(unit=units[idEntry] if entriesNames is not None else "N/D"))
            plt.ylabel("# Samples")
            plt.grid()

            # SAVING: Save figure if required
            if self.save_figs:
                plt.savefig("prediction_errors_" + entryName + ".png")

