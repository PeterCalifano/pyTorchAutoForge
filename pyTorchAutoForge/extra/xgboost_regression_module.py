from typing import TypeAlias, Literal
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import mlflow
import mlflow.xgboost
from dataclasses import dataclass, asdict
import numpy as np
from pathlib import Path
from pyTorchAutoForge.utils import torch_to_numpy
from torch import Tensor
import json


# Type aliases
DataframeOrNDArray: TypeAlias = pd.DataFrame | np.ndarray
Metrics_type_alias: TypeAlias = float | np.ndarray | None


@dataclass
class GoodnessOfFitMeter:
    """
    Class to compute and store regression metrics.
    """

    mse: Metrics_type_alias = None    # Mean Squared Error
    mae: Metrics_type_alias = None    # Mean Absolute Error
    r2: Metrics_type_alias = None     # R^2 Score
    rmse: Metrics_type_alias = None   # Root Mean Squared Error
    mape: Metrics_type_alias = None   # Mean Absolute Percentage Error
    median_ae: Metrics_type_alias = None  # Median Absolute Error

    def to_dict(self) -> dict:
        """
        Convert the metrics to a dictionary.
        """
        return {
            'mse': self.mse,
            'mae': self.mae,
            'r2': self.r2,
            'rmse': self.rmse,
            'mape': self.mape,
            'median_ae': self.median_ae
        }

    # Dunder methods
    def __iter__(self):
        yield from self.to_dict().items()

    def __dataframe__(self, nan_as_null: bool = False):
        import pandas as pd
        return pd.DataFrame([self.to_dict()])

    # Methods
    def compute_fit_metrics(self,
                            y_true: DataframeOrNDArray | Tensor,
                            y_pred: DataframeOrNDArray | Tensor) -> dict[str, float]:

        # Convert to numpy if torch tensor
        if isinstance(y_true, Tensor):
            y_true = torch_to_numpy(y_true)
        if isinstance(y_pred, Tensor):
            y_pred = torch_to_numpy(y_pred)

        # Compute metrics of fit
        self.mse = float(mean_squared_error(y_true, y_pred))
        self.mae = float(mean_absolute_error(y_true, y_pred))
        self.r2 = float(r2_score(y_true, y_pred))
        self.rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        self.mape = float(np.mean(np.abs((y_true - y_pred) / y_true))
                          * 100) if np.all(y_true != 0) else np.nan
        self.median_ae = float(np.median(np.abs(y_true - y_pred)))

        self.print_metrics()

        return self.to_dict()

    def print_metrics(self) -> None:
        print("\n\033[38;5;201m------- Regression Metrics -------\033[0m")
        print(f"Mean Squared Error (MSE): {self.mse:.4g} (Lower is better)")
        print(f"Mean Absolute Error (MAE): {self.mae:.4g} (Lower is better)")
        print(f"R^2 Score: {self.r2:.4g} (Higher is better)")
        print(
            f"Root Mean Squared Error (RMSE): {self.rmse:.4g} (Lower is better)")
        print(
            f"Mean Absolute Percentage Error (MAPE): {self.mape:.4g} % (Lower is better)")
        print(
            f"Median Absolute Error (MedAE): {self.median_ae:.4g} (Lower is better)")
        print("\033[38;5;201m---------------------------------\033[0m")


@dataclass
class TabularRegressorConfig:
    """
    Configuration for XGBoost regressor parameters using the scikit-learn estimator API.
    """
    n_estimators: int = 200
    max_depth: int = 10
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    gamma: float = 0.0  # Minimum loss reduction required to make a further partition
    reg_alpha: float = 0.0  # L1 regularization term
    reg_lambda: float = 1.0  # L2 regularization term
    objective: str = 'reg:squarederror'  # Objective function
    tree_method: str = 'auto'  # Tree construction algorithm selection
    booster: str = 'gbtree'  # Booster type
    verbosity: int = 1  # 0: silent, 1: warning, 2: info, 3: debug
    n_jobs: int = -1  # Use all available cores
    random_state: int = 42
    eval_metric: str = 'rmse'  # Evaluation metric for validation set


class XGBoostTabularDataRegressor:
    """
    XGBoost regressor wrapper for tabular data with MLflow tracking.

    Automatically logs parameters, metrics, and feature importance to MLflow.
    """

    def __init__(self,
                 config: TabularRegressorConfig,
                 mlflow_logging: bool = True,
                 save_model: bool = True,
                 feature_names: tuple[str] | None = None):

        # Store config and initialize model
        self.config = config
        self.params = asdict(config)
        self.model = xgb.XGBRegressor(**self.params)
        self.history = None
        self.feature_names: tuple[str] | None = feature_names

        # Options
        self.mlflow_logging = mlflow_logging
        self.save_model = save_model

    def fit(self,
            X_train: DataframeOrNDArray,
            y_train: DataframeOrNDArray,
            X_val: DataframeOrNDArray | None = None,
            y_val: DataframeOrNDArray | None = None,
            verbose: bool = False,
            run_name: str | None = None) -> GoodnessOfFitMeter:
        """
        Fit the model with optional validation and log to MLflow.

        Parameters:
        - X_train, y_train: training data
        - X_val, y_val: optional validation data for early stopping
        - early_stopping_rounds: stop after no improvement
        - eval_metric: evaluation metric ('rmse', 'mae', etc.)
        - verbose: print training progress
        - run_name: MLflow run name
        """
        with mlflow.start_run(run_name=run_name):
            # Log all config parameters
            mlflow.log_params(self.params)

            eval_set = [(X_train, y_train)]
            if X_val is not None and y_val is not None:
                eval_set.append((X_val, y_val))

            # Train model
            self.model.fit(
                X_train,
                y_train,
                verbose=verbose,
                eval_set=eval_set,
            )

            # Assign feature names if provided
            booster_ = self.model.get_booster()
            if self.feature_names is not None and booster_.feature_names is None:
                booster_.feature_names = list(self.feature_names)

            # Get model evaluation history
            self.history = self.model.evals_result()

            if self.mlflow_logging and self.history is not None:
                # Store evaluation history and log metrics per round
                for ds, metrics in self.history.items():
                    for metric, values in metrics.items():
                        for i, v in enumerate(values):
                            mlflow.log_metric(f"{ds}_{metric}", v, step=i)

            # Log final evaluation on validation if provided
            if X_val is not None and y_val is not None:
                evals, fit_meter = self.evaluate_model_metrics(X_val, y_val)

                if self.mlflow_logging and evals is not None:
                    mlflow.log_metrics(evals)

            if self.mlflow_logging and self.save_model:
                # Log the model itself
                mlflow.xgboost.log_model(self.model,
                                         artifact_path='model',
                                         input_example=X_train,
                                         registered_model_name=run_name)

        return fit_meter

    def predict(self, X):
        """Predict target values for input features X."""
        return self.model.predict(X)

    def evaluate_model_metrics(self,
                               X: DataframeOrNDArray,
                               y_true: DataframeOrNDArray):
        """
        Compute and return regression metrics (mse, mae, r2).
        """
        # Run inference
        y_pred = self.predict(X)

        # Calculate metrics
        fit_metrics = GoodnessOfFitMeter()
        metrics_dict = fit_metrics.compute_fit_metrics(y_true, y_pred)

        return metrics_dict, fit_metrics

    def evaluate_feature_importance(self,
                                    importance_type: Literal["weight",
                                                             "gain", "cover"] = "gain",
                                    output_folder: str | Path | None = None):
        """
        Evaluate and log feature importance.
        """
        # Check if model is fitted
        if self.model is None:
            raise ValueError(
                "No model provided. Please ensure the instance is correctly configured")

        if self.history is None:
            print("\033[38;5;208mNo training history found. Please ensure the model is trained before evaluating feature importance.\033[0m")

        # Plot feature importance
        fig, ax = plt.subplots()
        feats_importance = self.model.get_booster().get_score(
            importance_type=importance_type)

        df_imp = pd.DataFrame(
            list(feats_importance.items()),
            columns=['feature', 'importance']
        ).sort_values('importance', ascending=False)

        df_imp.plot.barh(x='feature', y='importance', ax=ax)

        ax.invert_yaxis()
        ax.set_title('Feature Importance')

        fig.tight_layout()

        # Pretty print the feature importance table to the console
        print(
            "\n\033[38;5;201m------- XGBoost Features Importance -------\033[0m")
        print(df_imp.sort_values('importance',
              ascending=False).to_string(index=False))
        print("\033[38;5;201m--------------------------------------------\033[0m")

        if self.mlflow_logging:
            # Log the figure to MLflow
            mlflow.log_figure(
                fig, f'features_importance_{importance_type}.png')

        if output_folder is not None:
            # Save the figure to the specified output folder
            if not isinstance(output_folder, Path):
                output_folder = Path(output_folder)

            output_folder.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                str(output_folder / f'features_importance_{importance_type}.png'))

            # Save the DataFrame to a CSV file
            df_imp.to_csv(
                output_folder / f'features_importance_{importance_type}.csv', index=False)
            print(f"Feature importance figure saved to {output_folder}")

        plt.pause(1)
        plt.close(fig)

    def get_xgb_model_info(self) -> dict:
        """
        Given a fitted XGBRegressor or XGBClassifier, returns:
        - n_params : int      # number of learned weights
        - size_mb  : float    # serialized model size in megabytes
        """
        # Sanity check
        if not isinstance(self.model, (xgb.XGBRegressor, xgb.XGBClassifier)):
            raise TypeError("model must be XGBRegressor or XGBClassifier")

        booster = self.model.get_booster()
        booster_type = booster.attributes().get('booster', 'gbtree')

        # Count parameters
        if booster_type == 'gbtree':
            # each leaf node has one weight
            df = booster.trees_to_dataframe()
            n_params = int((df.Feature == 'Leaf').sum())

        elif booster_type == 'gblinear':
            # JSON dump has a 'weight' dict
            dump_json = booster.get_dump(dump_format='json')[0]
            j = json.loads(dump_json)
            n_params = len(j.get('weight', {}))

        else:
            # Fallback: count leaves
            df = booster.trees_to_dataframe()
            n_params = int((df.Feature == 'Leaf').sum())

        # Estimate approximate model size in MB
        raw = booster.save_raw()         # Bytes count
        size_mb = len(raw) / 1024**2  # Convert to megabytes

        return {
            'n_params': n_params,
            'size_mb': size_mb
        }
