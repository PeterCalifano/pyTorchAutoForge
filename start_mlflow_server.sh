# NOTE: this script only works on peterc-desktopMSI. Server is started using the same uri as pointed by MLFLOW_TRACKING_URI environment variable, therefore works as local database automatically.

source /home/peterc/devDir/pyTorchAutoForge/.venvTorch/bin/activate
mlflow server --workers 2 --host 0.0.0.0 --port 8080 --backend-store-uri ${MLFLOW_TRACKING_URI} --default-artifact-root ${MLFLOW_ARTIFACTS_URI} 
#mlflow server --workers 2 --host 0.0.0.0 --port 8080 --backend-store-uri sqlite:///./localdb --default-artifact-root file:///localdb/artifacts
