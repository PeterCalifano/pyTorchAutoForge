'''Script implementing the quickstart tutorial of mlflow. Reference: https://mlflow.org/docs/latest/getting-started/intro-quickstart/index.html
Created by PeterC 06-07-2024, to learn how to use mlflow tracking API.'''

# ACHTUNG NOTE: While it can be valid to wrap the entire code within the start_run block, this is not recommended. 
# If there as in issue with the training of the model or any other portion of code that is unrelated to MLflow-related actions, 
# an empty or partially-logged run will be created, which will necessitate manual cleanup of the invalid run. 
# It is best to keep the training execution outside of the run context block to ensure that the loggable content 
# (parameters, metrics, artifacts, and the model) are fully materialized prior to logging.

import mlflow
#mlflow.set_tracking_uri(uri="http://<host>:<port>")

import mlflow
from mlflow.models import infer_signature

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Load the Iris dataset
X, y = datasets.load_iris(return_X_y=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the model hyperparameters
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 8888,
}

# Train the model
lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

# Predict on the test set
y_pred = lr.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)

# %% MLflow related code
# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Create a new MLflow Experiment
mlflow.set_experiment("MLflow Quickstart")

# Start an MLflow run
with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(params) # This logs the dictionary passed to the LogisticRegression scikit-learn model, which is used to build it

    # Log the loss metric
    mlflow.log_metric("accuracy", accuracy) # This logs the accuracy of the model as computed by acuracy_score

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Basic LR model for iris data") 

    # Infer the model signature
    signature = infer_signature(X_train, lr.predict(X_train)) 
    # This is specific to the given framework, in this case scikit-learn, to register the model. The input must be compatible with mlflow
    # PyTorch tensors must be converted to numpy to work.

    # Log the model with its signature --> this registers the model equivalently to a traced model in PyTorch, and enables it to be used for inference
    #                                      just by calling it through the mlflow API
    model_info = mlflow.sklearn.log_model(
        sk_model=lr,
        artifact_path="iris_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="tracking-quickstart",
    )

# %% Usage of mlflow API to load the model and make predictions

# Load the model back for predictions as a generic Python Function model
loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

predictions = loaded_model.predict(X_test)

iris_feature_names = datasets.load_iris().feature_names

result = pd.DataFrame(X_test, columns=iris_feature_names)
result["actual_class"] = y_test
result["predicted_class"] = predictions

result[:4]
