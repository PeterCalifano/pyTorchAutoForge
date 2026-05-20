#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

# Script variables
CONDA_EXE="${CONDA_EXE:-conda}"
ENV_NAME="${CONDA_ENV:-autoforge}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
CREATE_ENV=0
EDITABLE=0
BUILD_DOCS=0
RUN_CHECK=0
ARM_MODE=0
USE_MAMBA=0
INSTALL_NVIDIA_DEPLOY=0
PYTORCH_URL=""
TORCHVISION_URL=""
TORCHVISION_SOURCE_TAG=""
EXTRAS=()

usage() {
    cat <<'EOF'
Usage: ./conda_install.sh [options]

Default behavior:
  - use an existing conda environment named "autoforge"
  - install core pyTorchAutoForge package only, without optional extras
  - do not create environments unless --create-env is passed

Environment:
  -n, --env-name NAME           Conda environment name (default: CONDA_ENV or autoforge)
  -v, --venv_name NAME          Compatibility alias for --env-name
  -c, --create-env             Create environment if missing
      --python-version VERSION Python version for new envs (default: PYTHON_VERSION or 3.12)
      --conda-exe PATH         Conda executable (default: conda)
      --mamba                  Use mamba for env creation when available

Install:
      --core                   Core package only (default)
  -e, --editable               Install editable from this checkout
      --wheel                  Install non-editable package from this checkout (default)
      --extras LIST            Comma-separated pyproject extras, for example test,docs,explain
      --with-test              Add test extra
      --with-docs              Add docs extra
      --with-explain           Add explain extra
      --with-shap              Add explain-shap extra
      --with-captum            Add explain-captum extra
      --with-cuda              Add cuda_all extra
      --with-classical-ml      Add classical-ml extra
      --with-xgboost           Add xgboost extra
      --with-pysr              Add pysr extra
      --build-docs             Build MkDocs after install
      --check                  Run tests/.configuration/test_env.py after install

ARM / Jetson:
  -j, --jetson                 Enable ARM/Jetson-friendly mode
      --jetson_target          Compatibility alias for --jetson
      --arm                    Alias for --jetson
      --pytorch-url URL        Install board-specific PyTorch wheel/url before PTAF
      --torchvision-url URL    Install board-specific torchvision wheel/url before PTAF
      --torchvision-source-tag TAG
                               Build torchvision from source tag, for example v0.20.0
      --nvidia-deploy          Install NVIDIA deploy packages from pypi.nvidia.com

Compatibility:
  -s, --sudo_mode              Accepted for old callers; no system packages are installed here
  -h, --help                   Show this help
EOF
}

add_extra() {
    local raw_extra_="$1"
    local extra_=""

    IFS=',' read -ra split_extras_ <<< "${raw_extra_}"
    for extra_ in "${split_extras_[@]}"; do
        extra_="${extra_//[[:space:]]/}"
        if [[ -n "${extra_}" ]]; then
            EXTRAS+=("${extra_}")
        fi
    done
}

join_extras() {
    local joined_=""
    local seen_=","
    local extra_=""

    for extra_ in "${EXTRAS[@]}"; do
        if [[ "${seen_}" == *",${extra_},"* ]]; then
            continue
        fi
        seen_="${seen_}${extra_},"
        if [[ -z "${joined_}" ]]; then
            joined_="${extra_}"
        else
            joined_="${joined_},${extra_}"
        fi
    done

    printf "%s" "${joined_}"
}

source_conda() {
    local base_dir_=""
    base_dir_="$("${CONDA_EXE}" info --base)"
    # shellcheck source=/dev/null
    source "${base_dir_}/etc/profile.d/conda.sh"
}

conda_env_exists() {
    "${CONDA_EXE}" env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"
}

create_conda_env() {
    local create_cmd_=("${CONDA_EXE}" "create" "-y" "-n" "${ENV_NAME}" "python=${PYTHON_VERSION}" "pip" "setuptools" "wheel")

    if [[ "${USE_MAMBA}" -eq 1 ]] && command -v mamba >/dev/null 2>&1; then
        create_cmd_[0]="mamba"
    fi

    echo "Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}"
    "${create_cmd_[@]}"
}

install_url_if_requested() {
    local label_="$1"
    local url_="$2"

    if [[ -n "${url_}" ]]; then
        echo "Installing ${label_} from explicit URL"
        python -m pip install "${url_}"
    fi
}

install_torchvision_from_source() {
    local tag_="$1"
    local temp_dir_=""

    if [[ -z "${tag_}" ]]; then
        return
    fi

    temp_dir_="$(mktemp -d)"
    trap 'rm -rf "${temp_dir_}"' RETURN

    echo "Building torchvision from source tag ${tag_}"
    git clone --depth 1 --branch "${tag_}" https://github.com/pytorch/vision.git "${temp_dir_}/vision"
    python -m pip install "${temp_dir_}/vision"
}

detect_arm_mode() {
    local machine_=""
    machine_="$(uname -m)"

    case "${machine_}" in
        aarch64|arm64|armv7l|armv8l)
            ARM_MODE=1
            ;;
    esac
}

# Parser loop
while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--env-name|-v|--venv_name)
            ENV_NAME="$2"
            shift 2
            ;;
        -c|--create-env|--create_conda_env)
            CREATE_ENV=1
            shift
            ;;
        --python-version)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --conda-exe)
            CONDA_EXE="$2"
            shift 2
            ;;
        --mamba)
            USE_MAMBA=1
            shift
            ;;
        --core)
            EXTRAS=()
            shift
            ;;
        -e|--editable|--editable_mode)
            EDITABLE=1
            shift
            ;;
        --wheel)
            EDITABLE=0
            shift
            ;;
        --extras)
            add_extra "$2"
            shift 2
            ;;
        --with-test)
            add_extra "test"
            shift
            ;;
        --with-docs)
            add_extra "docs"
            shift
            ;;
        --with-explain)
            add_extra "explain"
            shift
            ;;
        --with-shap)
            add_extra "explain-shap"
            shift
            ;;
        --with-captum)
            add_extra "explain-captum"
            shift
            ;;
        --with-cuda)
            add_extra "cuda_all"
            shift
            ;;
        --with-classical-ml)
            add_extra "classical-ml"
            shift
            ;;
        --with-xgboost)
            add_extra "xgboost"
            shift
            ;;
        --with-pysr)
            add_extra "pysr"
            shift
            ;;
        --build-docs)
            BUILD_DOCS=1
            shift
            ;;
        --check)
            RUN_CHECK=1
            shift
            ;;
        -j|--jetson|--jetson_target|--arm)
            ARM_MODE=1
            shift
            ;;
        --pytorch-url)
            PYTORCH_URL="$2"
            shift 2
            ;;
        --torchvision-url)
            TORCHVISION_URL="$2"
            shift 2
            ;;
        --torchvision-source-tag)
            TORCHVISION_SOURCE_TAG="$2"
            shift 2
            ;;
        --nvidia-deploy)
            INSTALL_NVIDIA_DEPLOY=1
            shift
            ;;
        -s|--sudo_mode)
            echo "--sudo_mode accepted for compatibility; system package installation is not handled by this script."
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

# Check architecture before sourcing conda
detect_arm_mode

if ! command -v "${CONDA_EXE}" >/dev/null 2>&1; then
    echo "Conda executable not found: ${CONDA_EXE}" >&2
    exit 1
fi

# Source conda environment
source_conda

if conda_env_exists; then
    echo "Using existing conda environment '${ENV_NAME}'"
elif [[ "${CREATE_ENV}" -eq 1 ]]; then
    # Create conda if target environment not found
    create_conda_env
else
    echo "Conda environment '${ENV_NAME}' does not exist. Re-run with --create-env to create it." >&2
    exit 1
fi

# Activate environment and basic tools setup
conda activate "${ENV_NAME}"

python -m pip install --upgrade pip setuptools wheel build

if [[ "${ARM_MODE}" -eq 1 ]]; then
    echo "ARM/Jetson mode active. x86-only pyproject dependencies are skipped by platform markers."
    if [[ -z "${PYTORCH_URL}" ]]; then
        echo "No --pytorch-url provided. Use board-specific PyTorch install before GPU workflows if torch is not already available."
    fi
fi

# Install PyTorch and torchvision from URLs if provided
install_url_if_requested "PyTorch" "${PYTORCH_URL}"
install_url_if_requested "torchvision" "${TORCHVISION_URL}"
install_torchvision_from_source "${TORCHVISION_SOURCE_TAG}"

cd "${REPO_ROOT}"

# Append extras from command line and join into CSV for pip install
extras_csv="$(join_extras)"

# Install package with extras if provided
install_spec="."
if [[ -n "${extras_csv}" ]]; then
    install_spec=".[${extras_csv}]"
fi

if [[ "${EDITABLE}" -eq 1 ]]; then
    # Install as editable
    echo "Installing editable package: ${install_spec}"
    python -m pip install -e "${install_spec}"
else
    # Normal install    
    echo "Installing package: ${install_spec}"
    python -m pip install "${install_spec}"
fi

# Install nvidia tools from pypi.nvidia.com
if [[ "${INSTALL_NVIDIA_DEPLOY}" -eq 1 ]]; then
    echo "Installing optional NVIDIA deploy packages"
    python -m pip install -U --extra-index-url https://pypi.nvidia.com \
        pycuda torch-tensorrt tensorrt "nvidia-modelopt[all]"
fi

# Build documentation
if [[ "${BUILD_DOCS}" -eq 1 ]]; then
    CONDA_ENV="${ENV_NAME}" "${REPO_ROOT}/doc/makedoc.sh"
fi

# Test environment setup
if [[ "${RUN_CHECK}" -eq 1 ]]; then
    python "${REPO_ROOT}/tests/.configuration/test_env.py"
fi
