#!/bin/bash
set -euo pipefail

# Script variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

CONDA_EXE="${CONDA_EXE:-conda}"
ENV_NAME="${CONDA_ENV:-autoforge}"
REPOSITORY="pypi"
DIST_DIR="dist"
CLEAN=1
UPLOAD=1

# Usage guide
usage() {
    cat <<'EOF'
Usage: ./release_to_pypi.sh [options]

Builds package artifacts, runs twine check, then uploads to PyPI by default.

Options:
  -e, --env-name NAME    Conda environment name (default: CONDA_ENV or autoforge)
      --conda-exe PATH   Conda executable (default: conda)
  -r, --repository NAME  Twine repository name (default: pypi)
      --dist-dir PATH    Distribution output directory (default: dist)
      --no-clean         Keep existing dist and egg-info folders
      --skip-upload      Build and check only
  -h, --help             Show this help
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
        -r|--repository)
            REPOSITORY="$2"
            shift 2
            ;;
        --dist-dir)
            DIST_DIR="$2"
            shift 2
            ;;
        --no-clean)
            CLEAN=0
            shift
            ;;
        --skip-upload)
            UPLOAD=0
            shift
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

if ! command -v "${CONDA_EXE}" >/dev/null 2>&1; then
    echo "Conda executable not found: ${CONDA_EXE}" >&2
    exit 1
fi

# Source conda environment
CONDA_BASE="$("${CONDA_EXE}" info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

cd "${REPO_ROOT}"

# Clean dist and egg-info if requested (default: yes)
if [[ "${CLEAN}" -eq 1 ]]; then
    rm -rf "${DIST_DIR}"
    find . -maxdepth 1 -type d -name "*.egg-info" -exec rm -rf {} +
fi

# Build and check artifacts
python -m pip install --upgrade build twine
python -m build --outdir "${DIST_DIR}"
python -m twine check "${DIST_DIR}"/*

# Upload to PyPI if requested (default: yes)
if [[ "${UPLOAD}" -eq 1 ]]; then
    python -m twine upload --repository "${REPOSITORY}" "${DIST_DIR}"/*
else
    echo "Upload skipped. Artifacts are in ${DIST_DIR}."
fi
