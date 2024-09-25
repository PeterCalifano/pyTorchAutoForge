#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

echo "Hello! Installation script begins. This is going to install pytorch-autoforge with all the dependencies in a virtual env and build its documentation."
echo -e "Starting in 1 second...\n\n"
sleep 1
sudo apt install python3.11 python3.11-venv # Install python3-venv

cd ../.. # Go to the root directory of the project
python3.11 -m venv .venvTorch # Create virtual environment
source .venvTorch/bin/activate # Activate virtual environment
pip install -r requirements.txt --require-virtualenv # Install dependencies
pip install $SCRIPT_DIR --require-virtualenv # Install the package
cd $SCRIPT_DIR

# Install sphinx and theme, and build the documentation
pip install sphinx sphinx_rtd_theme --require-virtualenv
make -C docs html