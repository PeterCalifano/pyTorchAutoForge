#!/bin/bash
set -euo pipefail

# Script variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ENV_NAME="${CONDA_ENV:-autoforge}"
CONDA_EXE="${CONDA_EXE:-conda}"
USE_CONDA=1
HOST="${MLFLOW_HOST:-0.0.0.0}"
PORT="${MLFLOW_PORT:-8080}"
WORKERS="${MLFLOW_WORKERS:-2}"
BACKEND_STORE_URI="${MLFLOW_TRACKING_URI:-sqlite:///${REPO_ROOT}/mlruns/mlflow.db}"
DEFAULT_ARTIFACT_ROOT="${MLFLOW_ARTIFACTS_URI:-file://${REPO_ROOT}/mlruns/artifacts}"

# Usage guide
usage() {
    cat <<'EOF'
Usage: bash_scripts/start_mlflow_server.sh [options]

Options:
  -e, --env-name NAME          Conda environment name (default: CONDA_ENV or autoforge)
      --conda-exe PATH         Conda executable (default: conda)
      --no-conda               Do not activate conda environment
      --host HOST              Host bind address (default: MLFLOW_HOST or 0.0.0.0)
      --port PORT              Server port (default: MLFLOW_PORT or 8080)
      --workers COUNT          Worker count (default: MLFLOW_WORKERS or 2)
      --backend-store-uri URI  Backend store URI (default: MLFLOW_TRACKING_URI or local sqlite)
      --artifact-root URI      Artifact root URI (default: MLFLOW_ARTIFACTS_URI or local file store)
  -h, --help                   Show this help
EOF
}

# Parser loop
while [[ $# -gt 0 ]]; do
    case "$1" in
        -e|--env-name)
            ENV_NAME="$2"
            shift 2
            ;;
        --conda-exe)
            CONDA_EXE="$2"
            shift 2
            ;;
        --no-conda)
            USE_CONDA=0
            shift
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --backend-store-uri)
            BACKEND_STORE_URI="$2"
            shift 2
            ;;
        --artifact-root)
            DEFAULT_ARTIFACT_ROOT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

# Source conda environment
if [[ "${USE_CONDA}" -eq 1 ]]; then
    CONDA_BASE="$("${CONDA_EXE}" info --base)"
    # shellcheck source=/dev/null
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${ENV_NAME}"
fi

# Make sure artifact directory exists for local file store
mkdir -p "${REPO_ROOT}/mlruns/artifacts"

# Start MLflow server with python command
exec mlflow server \
    --workers "${WORKERS}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --backend-store-uri "${BACKEND_STORE_URI}" \
    --default-artifact-root "${DEFAULT_ARTIFACT_ROOT}"
