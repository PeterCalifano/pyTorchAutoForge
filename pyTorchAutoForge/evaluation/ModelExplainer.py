import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sympy import pretty_print
from torch import nn
from torch import Tensor
from pyTorchAutoForge.model_building import AutoForgeModule
from pyTorchAutoForge.optimization.ModelTrainingManager import TaskType
import numpy as np
import pandas as pd
from enum import Enum
from functools import partial
from pyTorchAutoForge.utils.conversion_utils import numpy_to_torch, torch_to_numpy
import pathlib, os, colorama

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

try:
    import shap, captum
    class ModelExplainerHelper():
        def __init__(self, model: nn.Module | AutoForgeModule,
                    task_type: TaskType,
                    input_samples: Tensor | np.ndarray | pd.DataFrame,
                    target_output_index: int,
                    explain_method: CaptumExplainMethods | ShapExplainMethods = CaptumExplainMethods.IntegratedGrad,
                    features_names: tuple[str, ...] | None = None,
                    output_names: tuple[str, ...] | None = None,
                    device: str | torch.device = "cpu",
                    auto_plot: bool = True,
                    save_explainer_output: bool = True,
                    output_folder : str | pathlib.Path = ".",
                    cluster_features : bool = False):

            # Store data
            self.model = model
            self.task_type = task_type
            self.explain_method = explain_method
            self.features_names = features_names
            self.output_names = output_names
            self.device = device
            self.auto_plot = auto_plot
            self.output_folder = output_folder
            self.save_explainer_output = save_explainer_output
            self.cluster_features = cluster_features

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
                import shap
                print('ModelExplainer is using SHAP library...')

                # Build SHAP masker (determines how missing features are treated and computes base values for Expectation computation)
                num_of_samples = np.ceil(
                    0.2 * self.input_samples.shape[0]).astype(int)
                
                # Get random indices from input samples
                random_indices = np.random.choice(self.input_samples.shape[0], num_of_samples, replace=False)

                background_dataset = shap.sample(torch_to_numpy(
                    self.input_samples[random_indices]),
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

                print('ModelExplainer loaded with SHAP explainability method object: ' +
                    explain_method.value)

        def explain_features(self):
            """
            _summary_

            _extended_summary_
            """

            if isinstance(self.explain_method, CaptumExplainMethods):
                ## CAPTUM MODE
                print(
                    f"{colorama.Style.BRIGHT}{colorama.Fore.LIGHTRED_EX}Running explainability analysis using Captum module...")
                # Call the captum attribute method
                # TODO this is for integrated gradients, need to generalize
                attributions, converge_deltas = self.captum_explainer.attribute(
                    self.input_samples, self.target_output_index, return_convergence_delta=True)

                # Convert to numpy
                attributions = torch_to_numpy(attributions)
                converge_deltas = torch_to_numpy(converge_deltas)

                # Compute importance stats
                stats = self.captum_compute_importance_stats_(attributions)

                print("Attribution statistics: \n")
                pretty_print(stats)

                if self.features_names is None:
                    self.features_names = tuple([f"Feature_{i}" for i in range(attributions.shape[1])])

                # Call visualization function
                self.captum_visualize_feats_importances(features_names=self.features_names, 
                                                        importances=stats["mean"], 
                                                        title="Feature Importances", 
                                                        errors_ranges=stats["std_dev"])

                return {"captum_stats": stats, "captum_attributions": attributions} # TODO replace with dedicated container objects
            
            elif isinstance(self.explain_method, ShapExplainMethods):
                ## SHAP MODE
                print(f"{colorama.Style.BRIGHT}{colorama.Fore.MAGENTA}Running explainability analysis using SHAP module...")

                input_samples_ = torch_to_numpy(self.input_samples)

                # Call the SHAP explainer
                shap_values = self.shap_explainer(input_samples_)

                # Reset print style
                print(f"{colorama.Style.RESET_ALL}")

                if self.save_explainer_output:
                    self.save_shap_to_disk(shap_values, 
                                            output_h5_filename="shap_values_artifact",
                                            output_folder=self.output_folder)

                # Run model inference to get model predictions
                # model_predictions = torch_to_numpy(
                #    self.model( numpy_to_torch(self.input_samples,
                #                               dtype=torch.float32).to(self.device) ) )

                if self.auto_plot:
                    # Call the SHAP plot function
                    figs_handles = self.plot_shap_values(shap_values,
                                        input_samples_,
                                        output_folder=self.output_folder,
                                        cluster_features=self.cluster_features)

                return {"shap_explanation": shap_values, "figs_handles": figs_handles}
            
        def save_shap_to_disk(self, 
                            shap_values:shap.Explanation, 
                            output_h5_filename: str = "shap_values_artifact",
                            output_folder: str | pathlib.Path = "."):
            print("Saving SHAP to .h5 file...")
            # Create output folder if it doesn't exist
            os.makedirs(output_folder, exist_ok=True)

            # Initialize h5 file
            h5_file = os.path.join(output_folder, output_h5_filename + ".h5")

            for idx, output_name in enumerate(shap_values.output_names):
                # Convert to pd.DataFrame for convenience
                shap_val_df = pd.DataFrame(shap_values.values[..., idx], columns=shap_values.feature_names)
                shap_val_df.to_hdf(
                    h5_file, key=f"shap_values_{output_name.replace(' ', '_')}", format="table")

        def plot_shap_values(self, shap_values: shap.Explanation,
                            input_features: np.ndarray,
                            model_predictions: np.ndarray | None = None,
                            feature_names: tuple[str] | None = None,
                            output_names: tuple[str] | None = None,
                            cluster_features: bool = True,
                            output_folder: str | pathlib.Path = "."):
            """
            Visualize SHAP values for multiple outputs.

            This function generates various visualizations for SHAP values, including bar plots,
            layered violin plots, and heatmaps. It supports clustering of features and handles
            multiple outputs.

            Args:
                shap_values (shap.Explanation): SHAP values object containing feature attributions.
                input_features (np.ndarray): Input features used for generating SHAP values.
                model_predictions (np.ndarray | None, optional): Model predictions corresponding to the input features.
                Defaults to None.
                feature_names (tuple[str] | None, optional): Names of the input features. If provided, it overrides
                the feature names in the SHAP values object. Defaults to None.
                output_names (tuple[str] | None, optional): Names of the model outputs. If provided, it overrides
                the output names in the SHAP values object. Defaults to None.
                cluster_features (bool, optional): Whether to cluster features for visualization. Defaults to True.
                output_folder (str | pathlib.Path, optional): Folder where the generated plots will be saved.
                Defaults to ".".

            Returns:
                list: A list of matplotlib figures corresponding to the generated visualizations.
            """
            
            # TODO add support for pandas dataframes
            # TODO if pandas dataframe, convert to numpy array and get names automatically

            # Override shap_values.feature_names if possible
            if feature_names is not None:
                if len(feature_names) != len(shap_values.feature_names):
                    print(f"{colorama.Fore.LIGHTRED_EX}Warning: Provided feature_names length {len(feature_names)} "
                        f"does not match shap_values.feature_names length {len(shap_values.feature_names)}"
                        f"{colorama.Style.RESET_ALL}")
                else:
                    shap_values.feature_names = list(feature_names) 

            # Determine number of outputs --------------------------------------
            vals = np.asarray(shap_values.values)
            dim = np.ndim(vals)

            if dim < 3:
                num_outputs = 1
            else:
                num_outputs = vals.shape[-1]

            # Determine output names
            output_names_ = [f"output_{i}" for i in range(num_outputs)] # Default names
            if output_names is None or len(output_names) != num_outputs:
                if output_names is not None:
                    print(f"{colorama.Fore.LIGHTRED_EX}Warning: Provided output_names length {len(output_names)} "
                        f"does not match number of outputs {num_outputs}"
                        f"{colorama.Style.RESET_ALL}")
                    
                # Fetch output names from shap_values
                if hasattr(shap_values, 'output_names') and shap_values.output_names is not None:
                    output_names_ = list(shap_values.output_names)  # type: ignore
            else:
                output_names_ = list(output_names)

                # Try updating shap_values output_names
                shap_values.output_names = output_names_  

            # Determine clustering
            clusters: list | None = [None] * num_outputs

            if cluster_features:
                for i in range(num_outputs):
                    if model_predictions is not None:
                        y = model_predictions[:, i]
                        clusters[i] = shap.utils.hclust( # type: ignore
                            X=input_features,
                            y=y,
                            linkage="average"
                        )
                    else:
                        expl_i = shap_values if num_outputs == 1 else shap_values[..., i]
                        clusters[i] = shap.utils.hclust( # type: ignore
                            X=expl_i.values,
                            linkage="average"
                        )

            # --- loop through each output and save plots --------------------------
            for idx, out_name in enumerate(output_names_):

                expl_i = shap_values if num_outputs == 1 else shap_values[..., idx]
                cluster = clusters[idx] if cluster_features else None # type: ignore
                output_figs = [] 

                # Remove spaces from output name
                out_name = out_name.replace(" ", "_")

                if cluster is not None:
                    # Bar plot
                    fig = plt.figure(figsize=(12, 8))
                    shap.plots.bar(
                        expl_i,
                        clustering=cluster,
                        clustering_cutoff=1.0,
                        show=False,
                        show_data="auto"
                    )
                else:
                    # Bar plot without clustering
                    fig = plt.figure(figsize=(12, 8))
                    shap.plots.bar(
                        expl_i,
                        show=False,
                        show_data="auto"
                    )

                plt.title(f"Clustered Absolute Mean SHAP Values ({out_name})")
                fig.savefig(os.path.join(output_folder, f"shap_bar_{out_name}.png"), dpi=400, bbox_inches="tight")
                output_figs.append(fig)

                # Layered violin plot
                fig = plt.figure(figsize=(12, 8))
                shap.plots.violin(
                    expl_i,
                    features=input_features,
                    feature_names=shap_values.feature_names,  # type: ignore
                    plot_type="layered_violin",
                    show=False,
                )
                plt.title(f"SHAP Layered Violin Plot ({out_name})")
                fig.savefig(os.path.join(output_folder, f"shap_violin_{out_name}.png"),
                            dpi=400, bbox_inches="tight")
                output_figs.append(fig)

                # Heatmap plot
                # TODO Verify why this causes session to crash. Replace or add decision plot
                #fig = plt.figure(figsize=(12, 8))
                #shap.plots.heatmap(
                #    expl_i,
                #    show=False
                #)
                #plt.title(f"SHAP Values Heatmap (is_clustered: {cluster_features})")
                #fig.savefig(os.path.join(output_folder, f"shap_heatmap_{out_name}.png"),
                #            dpi=400, bbox_inches="tight")
                #output_figs.append(fig)

            return output_figs

        def captum_explain_layers(self):
            """
            _summary_

            _extended_summary_
            """
            # TODO implement layer-wise attribution
            raise NotImplementedError(
                "Layer-wise attribution is not implemented yet.")

        def captum_visualize_feats_importances(self, 
                                            features_names: tuple[str, ...], 
                                            importances: np.ndarray, 
                                            title: str = "Average Feature Importances", 
                                            errors_ranges: np.ndarray | None = None):
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
            ax = sns.barplot(x="Importance", y="Feature",
                            data=df, palette="viridis")

            # Overlay error bars if errors are provided
            if errors_ranges is not None:
                for i, (imp, err) in enumerate(zip(df['Importance'], df['Interval'])):
                    ax.errorbar(imp, i, xerr=err, fmt='none', c='black', capsize=5)

            ax.set_title(title)
            ax.set_xlabel("Importance")
            ax.set_ylabel("Feature")

            plt.tight_layout()
            plt.pause(2)
            plt.close()

        def captum_compute_importance_stats_(self, 
                                            attributions, 
                                            quantiles=(0.25, 0.5, 0.75)) -> dict[str, np.ndarray]:
            """Compute mean importance and error measure from the attribution matrix.

            Args:
                attributions (np.ndarray): Attribution matrix of shape (n_samples, n_features) with feature attributions.
                quantiles (tuple): Quantiles to compute for the importance values. Default is (0.25, 0.5, 0.75).

            Returns:
                dict: Dictionary containing mean, quantiles, std deviation, and min/max values.
            """

            # Compute mean and std dev
            means: np.ndarray = np.mean(a=attributions, axis=0)
            std_dev: np.ndarray = np.std(attributions, axis=0)

            # Compute quantiles
            quantiles_list: np.ndarray = np.empty(
                shape=(len(quantiles), attributions.shape[1]))

            for i, q in enumerate(quantiles):
                if q < 0 or q > 1:
                    raise ValueError("Quantiles must be between 0 and 1.")
                quantiles_list[i] = np.quantile(attributions, q, axis=0)

            # Compute min, max
            lower: np.ndarray = np.min(attributions, axis=0)
            upper: np.ndarray = np.max(attributions, axis=0)

            return {"mean": means, "quantiles": quantiles_list, "std_dev": std_dev, "min_max": np.array([lower, upper])}

except ImportError:
    print("\033[38;5;208mSHAP or Captum not installed in this environment. "
        "ModelExplainerHelper() class won't be available\033[0m")    
    class ModelExplainerHelper():
        def __init__(self):
            raise NotImplementedError('You tried to instantiate ModelExplainerHelper, but did not install SHAP and Captum in this environment. Please do so first.')
            