from typing import Literal
import matplotlib.pyplot as plt
import seaborn as sns
from sympy import pretty_print
from torch import nn
from torch import Tensor
from pyTorchAutoForge.model_building import AutoForgeModule
from pyTorchAutoForge.optimization.ModelTrainingManager import TaskType
import numpy as np
from scipy import stats
import pandas as pd
from enum import Enum
import shap, torch
from functools import partial
from pyTorchAutoForge.utils.conversion_utils import numpy_to_torch, torch_to_numpy
import colorama

class CaptumExplainMethods(Enum):
    """
    CaptumExplainMethods Enumeration class listing all explainability methods supported by ModelExplainer helper class.
    """
    IntegratedGrad = "IntegratedGradients"
    Saliency = "Saliency" 
    GradientShap = "GradientShap"

class ShapExplainMethods(Enum):
    """
    ShapExplainMethods Enumeration class listing all explainability methods supported by ModelExplainer helper class.
    """
    SHAP = "shap"

class ModelExplainerHelper():
    def __init__(self, model: nn.Module | AutoForgeModule, 
                 task_type: TaskType, 
                 input_samples: Tensor | np.ndarray | pd.DataFrame, 
                 target_output_index: int, 
                 explain_method : CaptumExplainMethods | ShapExplainMethods = CaptumExplainMethods.IntegratedGrad,
                 features_names: tuple[str, ...] | tuple[Literal[str],...] | None = None,
                 output_names: tuple[str, ...] | tuple[Literal[str],...] | None = None,
                 device: str | torch.device = "cpu"):
        
        # Store data
        self.model = model
        self.task_type = task_type
        self.explain_method = explain_method
        self.features_names = features_names
        self.output_names = output_names
        self.device = device

        # Move data and model to device
        self.model.to(device)
        if isinstance(input_samples, Tensor):
            # Move input samples to device
            input_samples = input_samples.to(device)

        # Handle conversion of inputs
        if isinstance(input_samples, pd.DataFrame):
            # Convert DataFrame to numpy array
            input_samples = input_samples.to_numpy()
        
        if isinstance(input_samples, np.ndarray):
            # Convert numpy array to torch tensor
            input_samples = numpy_to_torch(input_samples)

        self.input_samples = input_samples
        self.target_output_index = target_output_index

        if isinstance(explain_method, CaptumExplainMethods):
            # Build captum method object 
            import captum
            self.captum_explainer = getattr(captum.attr, explain_method.value)
            self.captum_explainer = self.captum_explainer(self.model)
            print('ModelExplainer loaded with captum explainability method object: ' +
                  explain_method.value)

        elif isinstance(explain_method, ShapExplainMethods):
            print('ModelExplainer is using SHAP library...')

            # Build SHAP masker (defines how missing features are treated)
            num_of_samples = np.ceil(0.2 * self.input_samples.shape[0]).astype(int)
            background_dataset = shap.sample(torch_to_numpy(
                self.input_samples[:num_of_samples]),
                                             random_state=42)
            
            # Define model forward wrapper function for SHAP
            def forward_wrapper(X, device):
                
                # Convert numpy array to torch tensor
                X_tensor = numpy_to_torch(X, dtype=torch.float32).to(device)
                
                # Run model inference
                with torch.no_grad():
                    output = self.model(X_tensor)
                
                # Convert output to numpy array
                return torch_to_numpy(output, dtype=np.float64)

            # Wrap forward_wrapper as a partial function to specify the device
            forward_partial = partial(forward_wrapper, device=self.device)
            

            # Build SHAP explainer
            self.shap_explainer = shap.Explainer(model=forward_partial,
                                                 masker=background_dataset,
                                                 feature_names=self.features_names,
                                                 output_names=self.output_names)
            
            print('ModelExplainer loaded with SHAP explainability method object: ' + explain_method.value)

    def explain_features(self):
        """
         _summary_

        _extended_summary_
        """
        
        if isinstance(self.explain_method, CaptumExplainMethods): 

            print(f"{colorama.Style.BRIGHT}{colorama.Fore.LIGHTRED_EX}Running explainability analysis using Captum module...")
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

        elif isinstance(self.explain_method, ShapExplainMethods):
            print(f"{colorama.Style.BRIGHT}{colorama.Fore.MAGENTA}Running explainability analysis using SHAP module...")
            # Call the SHAP explainer
            shap_values = self.shap_explainer(torch_to_numpy(self.input_samples))
          
            # Run model inference to get model predictions
            model_predictions = torch_to_numpy( 
                self.model( numpy_to_torch(self.input_samples, 
                                           dtype=torch.float32).to(self.device) ) )

            # Run clustering algorithm to capture correlations
            # FIXME, this is not working unfortunately...
            #corr_clusters = shap.utils.hclust(torch_to_numpy(self.#input_samples), 
            #                          model_predictions, linkage="average")


            # Plot clustered absolute mean SHAP values
            fig1 = plt.figure(figsize=(10, 6))
            shap.plots.bar(
                shap_values,
                clustering=None,
                clustering_cutoff=1.0,
                show=False           
            )

            plt.title("Clustered Absolute Mean SHAP Values")
            fig1.savefig("shap_bar_plot.png", dpi=350, bbox_inches="tight")

            # Plot SHAP Violin layered plot
            fig2 = plt.figure(figsize=(10, 6))

            shap.plots.violin(
                shap_values, 
                features=self.input_samples, 
                feature_names=self.features_names, 
                plot_type="layered_violin",
                show=False, 
                )
            
            plt.title("SHAP Layered Violing Plot")
            fig2.savefig("shap_violin_plot.png", dpi=350, bbox_inches="tight")

            # Plot SHAP heatmap plot
            fig3 = plt.figure(figsize=(12, 6))
            shap.plots.heatmap(
                shap_values,
                show=False        # defer display so we can save
            )

            plt.title("SHAP Values Heatmap (clustered for similarity)")
            fig3.savefig("shap_heatmap.png", dpi=400, bbox_inches="tight")
            plt.close(fig3)

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

