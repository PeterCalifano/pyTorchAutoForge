#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate autoforge

while getopts "a,i:,o:,p:" opt; do
    case $opt in
        a) AUTOBUILD=1 ;;
        i) INPUT="${OPTARG:-doc}" ;;
        o) OUTPUT="${OPTARG:-test_autodoc}" ;;
        p) PORT="${OPTARG:-8000}" ;;
        *) echo "Invalid option"; exit 1 ;;
    esac
done

# Set default values if not provided
AUTOBUILD=${AUTOBUILD:-0}
INPUT=${INPUT:-doc}
OUTPUT=${OUTPUT:-test_autodoc}
PORT=${PORT:-8000}

if [ -n "$AUTOBUILD" ]; then
    sphinx-autobuild "$INPUT" "$OUTPUT" --port "$PORT" --open-browser
else
    make html
fi
