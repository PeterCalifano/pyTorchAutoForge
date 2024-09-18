# NOTE: this script only works on peterc-desktopMSI
#mlflow server --workers 2 --host 0.0.0.0 --port 8080 --backend-store-uri ${MLFLOW_TRACKING_URI} --default-artifact-root ${MLFLOW_ARTIFACTS_URI}
mlflow server --workers 2 --host 0.0.0.0 --port 8080 --backend-store-uri sqlite:///./localdb --default-artifact-root file:///localdb/artifacts