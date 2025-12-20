#!/bin/bash

set -euo pipefail

usage() {
  echo "Usage: $0 [-c conda_home] [-e env_name] [-- pytest args]" >&2
}

conda_home="${CONDA_HOME:-/home/peterc/miniconda3}"
env_name="${CONDA_ENV:-autoforge}"

while getopts ":c:e:h" opt; do
  case "$opt" in
    c) conda_home="$OPTARG" ;;
    e) env_name="$OPTARG" ;;
    h) usage; exit 0 ;;
    \?) echo "Invalid option: -$OPTARG" >&2; usage; exit 1 ;;
  esac
done

shift $((OPTIND - 1))

python_bin="${conda_home}/envs/${env_name}/bin/python"

if [[ ! -x "$python_bin" ]]; then
  echo "Python executable not found at $python_bin" >&2
  exit 1
fi

exec "$python_bin" -m pytest "$@"
