import matplotlib.pyplot as plt
import seaborn as sns
import captum
from sympy import pretty_print
from torch import nn
from torch import Tensor
from pyTorchAutoForge.model_building import AutoForgeModule
from pyTorchAutoForge.optimization.ModelTrainingManager import TaskType
import numpy as np
from scipy import stats
import pandas as pd
from enum import Enum

from pyTorchAutoForge.utils.conversion_utils import numpy_to_torch, torch_to_numpy


class CaptumExplainMethods(Enum):
    """
    CaptumExplainMethods Enumeration class listing all explainability methods supported by ModelExplainer helper class.
    """
    IntegratedGrad = "IntegratedGradients"
    Saliency = "Saliency" 
    GradientShap = "GradientShap"

class ModelExplainer():
    def __init__(self, model: nn.Module | AutoForgeModule, 
                 task_type: TaskType, 
                 input_samples: Tensor | np.ndarray | pd.DataFrame, 
                 target_output_index: int, 
                 explain_method : CaptumExplainMethods = CaptumExplainMethods.IntegratedGrad,
                 features_names: list[str] | None = None):
        
        # Store data
        self.model = model
        self.task_type = task_type
        self.explain_method = explain_method
        self.features_names = features_names

        # Handle conversion of inputs
        if isinstance(input_samples, pd.DataFrame):
            # Convert DataFrame to numpy array
            input_samples = input_samples.to_numpy()
        
        if isinstance(input_samples, np.ndarray):
            # Convert numpy array to torch tensor
            input_samples = numpy_to_torch(input_samples)

        self.input_samples = input_samples
        self.target_output_index = target_output_index

        # Build captum method object 
        print('ModelExplainer loaded with captum explainability method object: ' + explain_method.value)
        self.captum_explainer = getattr(captum.attr, explain_method.value)
        self.captum_explainer = self.captum_explainer(self.model)

    def explain_features(self):
        """
         _summary_

        _extended_summary_
        """

        # Call the captum attribute method
        # TODO this is for integrated gradients, need to generalize
        attributions, converge_deltas = self.captum_explainer.attribute(self.input_samples, self.target_output_index, return_convergence_delta=True)

        # Convert to numpy
        attributions = torch_to_numpy(attributions)
        converge_deltas = torch_to_numpy(converge_deltas)

        # Compute importance stats
        stats = self.compute_importance_stats_(attributions)

        print("Attribution statistics: \n")
        pretty_print(stats)

        if self.features_names is None:
            self.features_names = [f"Feature {i}" for i in range(attributions.shape[1])]
    
        # Call visualization function
        self.visualize_feats_importances(self.features_names, stats["mean"], title="Feature Importances", errors_ranges=stats["std_dev"])


    def explain_layers(self):
        """
        _summary_

        _extended_summary_
        """
        # TODO implement layer-wise attribution
        raise NotImplementedError("Layer-wise attribution is not implemented yet.")

    def visualize_feats_importances(self, features_names : list[str] | tuple[str], importances : np.ndarray, title:str="Average Feature Importances", errors_ranges : np.ndarray | None = None):
        """Visualize feature importances with optional error bars.

        This function prints each feature alongside its calculated importance and then
        creates a horizontal bar plot using seaborn. Optionally, error bars are overlaid
        to represent the uncertainty or range in feature importances.

        Args:
            features_names (list[str] | tuple[str]): A list or tuple of feature names.
            importances (np.ndarray): An array containing the importance values for each feature.
            title (str, optional): The title of the plot. Defaults to "Average Feature Importances".
            errors_ranges (np.ndarray | None, optional): An array of error ranges for each feature.
            If None, error bars are not displayed.
        """

        # Print each feature and its importance
        for name, imp in zip(features_names, importances):
            print(f"{name}: {imp:.3f}")

        # Create a DataFrame for plotting and sort by importance ascending
        df = pd.DataFrame({
            'Feature': features_names,
            'Importance': importances,
            'Interval': errors_ranges if errors_ranges is not None else [0]*len(importances)
        })
        
        df.sort_values(by="Importance", ascending=True, inplace=True)

        # Set seaborn style
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))

        # Create a horizontal bar plot
        ax = sns.barplot(x="Importance", y="Feature", data=df, palette="viridis")

        # Overlay error bars if errors are provided
        if errors_ranges is not None:
            for i, (imp, err) in enumerate(zip(df['Importance'], df['Interval'])):
                ax.errorbar(imp, i, xerr=err, fmt='none', c='black', capsize=5)


        ax.set_title(title)
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")

        plt.tight_layout()
        plt.show()


    def compute_importance_stats_(self, attributions, quantiles=(0.25, 0.5, 0.75)) -> dict[str, np.ndarray]:
        """Compute mean importance and error measure from the attribution matrix.

        Args:
            attributions (np.ndarray): Attribution matrix of shape (n_samples, n_features) with feature attributions.
            quantiles (tuple): Quantiles to compute for the importance values. Default is (0.25, 0.5, 0.75).

        Returns:
            dict: Dictionary containing mean, quantiles, std deviation, and min/max values.
        """            

        # Compute mean and std dev
        means : np.ndarray = np.mean(a=attributions, axis=0)
        std_dev: np.ndarray = np.std(attributions, axis=0)

        # Compute quantiles
        quantiles_list: np.ndarray = np.empty(shape=(len(quantiles), attributions.shape[1]))

        for i, q in enumerate(quantiles):
            if q < 0 or q > 1:
                raise ValueError("Quantiles must be between 0 and 1.")
            quantiles_list[i] = np.quantile(attributions, q, axis=0)

        # Compute min, max
        lower : np.ndarray = np.min(attributions, axis=0)
        upper : np.ndarray = np.max(attributions, axis=0)

        return {"mean": means, "quantiles": quantiles_list, "std_dev": std_dev, "min_max": np.array([lower, upper])}

