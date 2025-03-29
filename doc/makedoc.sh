#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate autoforge
sphinx-autobuild doc test_autodoc --port 8000 --open-browser
