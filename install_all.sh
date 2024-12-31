echo "Hello! Installation script begins. This is going to install pytorch-autoforge with all the dependencies in a virtual env and build its documentation."
echo -e "Starting in 1 second...\n\n"
sleep 1
sudo apt install python3.11 python3.11-venv # Install python3-venv
python3.11 -m venv .venvTorch # Create virtual environment
source .venvTorch/bin/activate # Activate virtual environment
pip install -e . --require-virtualenv# Install the package in editable mode
pip install -r requirements.txt --require-virtualenv # Install dependencies

# Install sphinx and theme, and build the documentation
pip install sphinx sphinx_rtd_theme --require-virtualenv
make -C docs html