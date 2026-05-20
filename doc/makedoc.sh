#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONDA_EXE="${CONDA_EXE:-conda}"
CONDA_BASE="$("${CONDA_EXE}" info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if [ -n "${CONDA_ENV:-}" ]; then
    conda activate "${CONDA_ENV}"
elif [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
    echo "Using active conda env: ${CONDA_DEFAULT_ENV}"
else
    conda activate autoforge
fi

ensure_docs_dependencies_() {
    python - <<'PY'
import importlib.util
import sys

required_modules_ = (
    "sphinx",
    "pydata_sphinx_theme",
    "myst_parser",
    "autoapi",
    "sphinx_copybutton",
)
missing_modules_ = [
    module_name_
    for module_name_ in required_modules_
    if importlib.util.find_spec(module_name_) is None
]
if missing_modules_:
    print("Missing docs modules: " + ", ".join(missing_modules_))
    sys.exit(1)
PY
}

if ! ensure_docs_dependencies_; then
    echo "Installing Sphinx documentation dependencies into conda env: ${CONDA_DEFAULT_ENV}"
    python -m pip install -r doc/requirements.txt
    ensure_docs_dependencies_
fi

SERVE=0
STRICT=0
HOST="127.0.0.1"
PORT="8000"

while getopts "ash:p:" opt; do
    case $opt in
        a) SERVE=1 ;;
        s) STRICT=1 ;;
        h) HOST="$OPTARG" ;;
        p) PORT="$OPTARG" ;;
        *) echo "Invalid option"; exit 1 ;;
    esac
done

SPHINX_ARGS=(-b html doc site)
if [ "$STRICT" -eq 1 ]; then
    SPHINX_ARGS=(-W --keep-going "${SPHINX_ARGS[@]}")
fi

if [ "$SERVE" -eq 1 ]; then
    python -m sphinx "${SPHINX_ARGS[@]}"
    cd site
    python -m http.server "${PORT}" --bind "${HOST}"
else
    python -m sphinx "${SPHINX_ARGS[@]}"
fi
