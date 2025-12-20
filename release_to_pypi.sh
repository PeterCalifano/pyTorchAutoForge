#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate autoforge

# Clear dist folder
if [ -d "dist" ]; then
    rm -r dist
fi

# Remove egg-info folder
if [ -d "*.egg-info" ]; then
    rm -r *.egg-info
fi

# Build latest wheel
python -m build

if [ -d "dist" ]; then
    twine check dist/*
    twine upload --repository pypi dist/* #-u __token__ -p $PYPI_TOKEN --verbose
fi