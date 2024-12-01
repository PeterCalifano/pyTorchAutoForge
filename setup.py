'''Module setup file created by PeterC - 06/30/2024.
Major update for release version: 09/17/2024 (v0.1)'''
from setuptools import setup, find_packages

import getpass
import sys

# Get the current user's username
username = getpass.getuser()

# Check if the username contains "ele" or "pilo"
if "ele" in username or "pilo" in username:
    sys.stderr.write(
        "Error: not allowed to install this library.\n")
    sys.exit(1)


setup(
    name='pyTorchAutoForge',
    version='0.3.0',
    packages=find_packages(),
    install_requires=[
        "torch==2.2.1",
        "torch-tb-profiler==0.4.3",
        "torchaudio==2.2.1",
        "torchvision==0.17.1",
        "scikit-learn==1.4.1.post1",
        "scipy==1.12.0",
        "numpy==1.26.4",
        "onnx==1.16.1",
        "onnxscript==0.1.0.dev20240609",
        "tensorboard==2.16.2",
        "tensorboard-data-server==0.7.2",
        "optuna==3.6.1",
        "mlflow==2.14.1",
    ],
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
    python_requires='>=3.10',
)
