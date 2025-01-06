'''Module setup file created by PeterC - 06/30/2024.
Major update for release version: 09/17/2024 (v0.3.0)
Update for jetson: 01-05-2025 (v0.3.1)'''

from setuptools import setup, find_packages

import getpass
import sys
import platform

# Get the current user's username
username = getpass.getuser()

# Check if the username contains "ele" or "pilo"
if "ele" in username.lower() or "pilo" in username.lower():
    sys.stderr.write(
        "Error: not allowed to install this library.\n")
    sys.exit(1)

# Check architecture

# Install torch for Jetson
#pip install torch https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl --require-virtualenv

install_requires = [
    "torch-tb-profiler<=0.4.3",
    "scikit-learn<=1.6.1",
    "scipy<=1.16.1",
    "numpy<=2.2.1",
    "onnx<=1.15.1",
    "onnxscript<=0.1.0.dev20240609",
    "optuna<=4.1.1",
    "mlflow<=2.19.1",
]

def is_jetson():
    try:
        with open('/proc/device-tree/model') as f:
            model = f.read().lower()
            return 'nvidia' in model and 'jetson' in model
    except FileNotFoundError:
        return False

if not is_jetson():
    # Install torch for Jetson
    # pip install torch https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl --require-virtualenv

    torch_dep = "torch<=2.6.1"
    torchvision_dep = "torchvision<=0.25.1"

    # Add dependencies
    install_requires.append(torch_dep)
    install_requires.append(torchvision_dep)
else:
    # pyTorchAutoForge
    print('pyTorchAutoForge: installing on Jetson. torch and torchvision are assumed to be installed separately.')

setup(
    name='pyTorchAutoForge',
    version='0.3.1',
    packages=find_packages(),
    install_requires=install_requires,
    entry_points={
        'console_scripts': [
            # Define command-line executables here if needed
        ],
    },
    author='Pietro Califano (PeterC)',
    author_email='petercalifano.gs@gmail.com, pietro.califano@polimi.it',
    description='Custom library based on raw PyTorch to automate DNN development, tracking and deployment, tightly integrated with MLflow and Optuna.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='',
    classifiers=[
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)
