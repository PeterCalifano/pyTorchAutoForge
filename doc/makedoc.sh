#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-autoforge}"

SERVE=0
HOST="127.0.0.1"
PORT="8000"

while getopts "ah:p:" opt; do
    case $opt in
        a) SERVE=1 ;;
        h) HOST="$OPTARG" ;;
        p) PORT="$OPTARG" ;;
        *) echo "Invalid option"; exit 1 ;;
    esac
done

if [ "$SERVE" -eq 1 ]; then
    mkdocs serve --dev-addr "${HOST}:${PORT}"
else
    mkdocs build --strict
fi
