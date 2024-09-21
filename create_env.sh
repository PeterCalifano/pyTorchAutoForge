sudo apt install python3.11 python3.11-venv # Install python3-venv
python3.11 -m venv .venvTorch # Create virtual environment
source .venvTorch/bin/activate # Activate virtual environment
pip install -r requirements.txt --require-virtualenv # Install dependencies
pip install . --require-virtualenv # Install the package
