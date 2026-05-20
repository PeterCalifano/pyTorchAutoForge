#!/bin/bash
set -euo pipefail

# Script variables
ENV_NAME="${CONDA_ENV:-autoforge}"
CONDA_EXE="${CONDA_EXE:-conda}"
USE_CONDA=1
REQUIRE_CUDA=0

# Usage guide
usage() {
    cat <<'EOF'
Usage: bash_scripts/check_torch_availability.sh [options]

Options:
  -e, --env-name NAME   Conda environment name (default: CONDA_ENV or autoforge)
      --conda-exe PATH  Conda executable (default: conda)
      --no-conda        Do not activate conda environment
      --require-cuda    Return non-zero when CUDA is unavailable
  -h, --help            Show this help
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
        --require-cuda)
            REQUIRE_CUDA=1
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

# Source conda environment
if [[ "${USE_CONDA}" -eq 1 ]]; then
    CONDA_BASE="$("${CONDA_EXE}" info --base)"
    # shellcheck source=/dev/null
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${ENV_NAME}"
fi

# Set environment variable to make CUDA errors easier to debug
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
export PTAF_REQUIRE_CUDA="${REQUIRE_CUDA}"

# Check PyTorch and CUDA availability with a python command
python -c '
import os
import sys

import torch

cuda_available = torch.cuda.is_available()
print("Torch version:", torch.__version__)
print("CUDA availability:", cuda_available)

if cuda_available:
    device_index = torch.cuda.current_device()
    print("Torch device props:", torch.cuda.get_device_properties(device_index))
    tensor = torch.tensor(2.0, device="cuda").fill_(3.14)
    print("Tensor created on CUDA:", tensor)
elif os.environ.get("PTAF_REQUIRE_CUDA") == "1":
    print("CUDA is not available", file=sys.stderr)
    sys.exit(1)
'
