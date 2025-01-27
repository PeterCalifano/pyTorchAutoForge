sudo apt install python3.11 python3.11-venv # Install python3-venv

# Default values
jetson_target=false
editable_mode=true
venv_name=".venvTorch"

# Parse options using getopt
# NOTE: no ":" after option means no argument, ":" means required argument, "::" means optional argument
OPTIONS=j,v:
LONGOPTIONS=jetson_target,venv_name:

# Parsed arguments list with getopt
PARSED=$(getopt --options ${OPTIONS} --longoptions ${LONGOPTIONS} --name "$0" -- "$@") 
# TODO check if this is where I need to modify something to allow things like -B build, instead of -Bbuild

# Check validity of input arguments 
if [[ $? -ne 0 ]]; then

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
    -v|--venv_name)
      venv_name=$2
      shift 2
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

    echo "TODO Update script for Jetson target"
    exit 0 # FIXME: Jetson target NOT updated

    # Create virtualenv for Jetson
    python3 -m venv $venv_name --system-site-packages # Create virtual environment
    source $venv_name/bin/activate # Activate virtual environment
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
    source $venv_name/bin/activate # Activate virtual environment

    pip install norse==1.0.0 --ignore-requires-python --require-virtualenv && pip install tonic expelliarmus --require-virtualenv 
    #pip install aestream --ignore-requires-python --require-virtualenv # FIXME: build fails due to "CUDA20" entry
    pip install nvidia-pyindex --require-virtualenv
    pip install pycuda --require-virtualenv # Install pycuda

    # ACHTUNG: this must run correctly before torch_tensorrt
    pip install "nvidia-modelopt[all]" -U --extra-index-url https://pypi.nvidia.com

    source $venv_name/bin/activate # Activate virtual environment

    #  Install torch-tensorrt from source 
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

    # Install required python packages of torch-tensorrt
    python -m pip install -r toolchains/jp_workspaces/requirements.txt # NOTE: Installs correct version of setuptools. Do not touch it.

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
    python3 -m venv $venv_name # Create virtual environment
    source $venv_name/bin/activate # Activate virtual environment
    
    #pip install -r requirements.txt --require-virtualenv # Install dependencies that do not cause issues...
    #python -m pip install -r toolchains/jp_workspaces/test_requirements.txt # Required for test cases

    # Tools for building and installing wheels
    echo "Installing setuptools, twine, and build..."
    pip install setuptools twine build --require-virtualenv
    python -m ensurepip --upgrade --require-virtualenv
    python -m pip install --upgrade pip --require-virtualenv

    # Install key modules not managed by dependencies installation for versioning reasons
    echo "Installing additional key modules..."
    pip install norse tonic aestream expelliarmus --require-virtualenv 
    
    # Build pyTorchAutoForge wheel
    if [ "$editable_mode" = true ]; then
        echo "Building and installing pyTorchAutoForge in editable mode..."
        pip install -e . --require-virtualenv # Install the package in editable mode
    else
      echo "Building and installing pyTorchAutoForge wheel..."
      python -m build 
      pip install -e dist/*.whl --require-virtualenv # Install pyTorchAutoForge wheel
    fi

    # Install tools for model optimization and deployment
    echo "Installing tools for model optimization and deployment by Nvidia..."
    python -m pip install pycuda torch torchvision torch-tensorrt tensorrt "nvidia-modelopt[all]" -U --extra-index-url https://pypi.nvidia.com
  fi

  deactivate # Deactivate virtual environment if any
  source $venv_name/bin/activate # Activate virtual environment
  # Check installation by printing versions in python
  python -m test_env



