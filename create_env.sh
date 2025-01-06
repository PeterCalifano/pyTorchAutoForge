sudo apt install python3.11 python3.11-venv # Install python3-venv

# Default values
jetson_target=false

# Parse options using getopt
# NOTE: no ":" after option means no argument, ":" means required argument, "::" means optional argument
OPTIONS=j
LONGOPTIONS=jetson_target

# Parsed arguments list with getopt
PARSED=$(getopt --options ${OPTIONS} --longoptions ${LONGOPTIONS} --name "$0" -- "$@") 
# TODO check if this is where I need to modify something to allow things like -B build, instead of -Bbuild

# Check validity of input arguments 
if [[ $? -ne 0 ]]; then
  # e.g. $? == 1
  #  then getopt has complained about wrong arguments to stdout
  exit 2
fi

# Parse arguments
eval set -- "$PARSED"

# Process options (change default values if needed)
while true; do
  case "$1" in
    -j|--jetson_target)
      jetson_target=true
      shift
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Not a valid option: $1" >&2
      exit 3
      ;;
  esac
done


if [ $jetson_target ] & [ ! -f /usr/local/cuda/lib64/libcusparseLt.so ]; then
    echo "libcusparseLt.so not found. Downloading and installing..."
    # if not exist, download and copy to the directory
    wget https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-sbsa/libcusparse_lt-linux-sbsa-0.5.2.1-archive.tar.xz
    tar xf libcusparse_lt-linux-sbsa-0.5.2.1-archive.tar.xz
    sudo cp -a libcusparse_lt-linux-sbsa-0.5.2.1-archive/include/* /usr/local/cuda/include/
    sudo cp -a libcusparse_lt-linux-sbsa-0.5.2.1-archive/lib/* /usr/local/cuda/lib64/
    rm libcusparse_lt-linux-sbsa-0.5.2.1-archive.tar.xz
    rm -r libcusparse_lt-linux-sbsa-0.5.2.1-archive
fi

if [ "$jetson_target" = true ]; then

    # Create virtualenv for Jetson
    python3 -m venv .venvTorch --system-site-packages # Create virtual environment
    source .venvTorch/bin/activate # Activate virtual environment
    pip install -r requirements.txt --require-virtualenv # Install dependencies
    pip install -e . --require-virtualenv # Install the package in editable mode

    # Remove torch and torchvision
    pip uninstall -y torch torchvision torchaudio

    # Install torch for Jetson
    pip install torch https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl --require-virtualenv

    # Build and install torchvision from source
    # From guide: https://github.com/azimjaan21/jetpack-6.1-pytorch-torchvision-/blob/main/README.md
    git clone https://github.com/pytorch/vision.git
    cd vision
    git checkout tags/v0.20.0
    python3 setup.py install 

    # Clean up
    cd ..
    sudo rm -r vision
    
    # Try to build torch-tensorrt
    source .venvTorch/bin/activate # Activate virtual environment

    pip install norse==1.0.0 --ignore-requires-python --require-virtualenv && pip install tonic expelliarmus --require-virtualenv 
    #pip install aestream --ignore-requires-python --require-virtualenv # FIXME: build fails due to "CUDA20" entry
    pip install --upgrade setuptools --require-virtualenv # Ensure setuptools is up to date
    pip install nvidia-pyindex --require-virtualenv
    pip install pycuda --require-virtualenv # Install pycuda

    # ACHTUNG: this must run correctly before torch_tensorrt
    pip install "nvidia-modelopt[all]" -U --extra-index-url https://pypi.nvidia.com

    #  Install torch-tensorrt from source (requires new version of setuptools)
    mkdir lib
    cd lib

    # Check if submodule exists 
    if [ -d "TensorRT" ]; then
        echo "TensorRT submodule exists"
    else
        git submodule add --branch release/2.5 https://github.com/pytorch/TensorRT.git # Try to use release/2.6 (latest)
    fi
    
    cd TensorRT
    git checkout release/2.5
    git pull

    cuda_version=$(nvcc --version | grep Cuda | grep release | cut -d ',' -f 2 | sed -e 's/ release //g')
    export TORCH_INSTALL_PATH=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))")
    export SITE_PACKAGE_PATH=${TORCH_INSTALL_PATH::-6}
    export CUDA_HOME=/usr/local/cuda-${cuda_version}/

    # Replace the MODULE.bazel with the jetpack one
    cat toolchains/jp_workspaces/MODULE.bazel.tmpl | envsubst > MODULE.bazel

    # build and install torch_tensorrt wheel file
    python setup.py install --use-cxx11-abi
    cd ../..
    
else

    # Create virtualenv for other targets
    python3 -m venv .venvTorch # Create virtual environment
    source .venvTorch/bin/activate # Activate virtual environment
    pip install -r requirements.txt --require-virtualenv # Install dependencies that do not cause issues...

    # Here just try to use pip
    pip install norse==1.0.0 --ignore-requires-python --require-virtualenv && pip install tonic aestream expelliarmus --require-virtualenv --ignore-requires-python 
    pip install setuptools # Ensure setuptools is up to date
    pip install nvidia-pyindex
    pip install pycuda # Install pycuda
    pip install torch torchvision torchaudio # Install torch and torchvision
    
    pip install -e . --require-virtualenv # Install pyTorchAutoForge

    pip install nvidia-tensorrt
    pip install nvidia-modelopt[all] # Install modelopt
    pip install torch-tensorrt # Install torch-tensorrt
fi


  # Test installation by printing version in python
  echo -e "\e[32m-------------------------------------------------- Testing installations --------------------------------------------------\n\e[0m"
  echo -e "\n \e[32m \tTesting torch, torchvision, tensorrt, torch-tensorrt...\n"
  python -c "import torch; import torchvision; print('Torch Version:', torch.__version__); print('TorchVision Version:', torchvision.__version__); print('CUDA Available:', torch.cuda.is_available());"
  python -c "import tensorrt; print('TensorRT Version:', tensorrt.__version__)"
  python -c "import torch_tensorrt; print('Torch-TensorRT Version:', torch_tensorrt.__version__)"
  
  echo -e "\n \e[32m \tTesting spiking_networks modules...\n"
  python -c "import norse; print('Norse Version:', norse.__version__)"
  python -c "import tonic; print('Tonic Version:', tonic.__version__)"

  echo -e "\n \e[32m \tTesting pyTorchAutoForge...\n"
  python -c "import pyTorchAutoForge; print('pyTorchAutoForge Version:', pyTorchAutoForge.__version__)"

  echo -e "\n \e[32m \tTesting nvidia-modelopt...\n"
  python -c "import modelopt.torch.quantization.extensions as ext; ext.precompile()" # Attempt to load modelopt and compile extensions



