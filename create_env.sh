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

# Create virtualenv for Jetson
python3 -m venv .venvTorch # Create virtual environment
source .venvTorch/bin/activate # Activate virtual environment
# For other targets
pip install -r requirements.txt --require-virtualenv # Install dependencies
pip install . --require-virtualenv # Install the package

if [ "$jetson_target" = true ]; then
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

    # Test installation using script
    python -c "import torch; import torchvision; print('Torch Version:', torch.__version__); print('TorchVision Version:', torchvision.__version__); print('CUDA Available:', torch.cuda.is_available()); if torch.cuda.is_available(): print('CUDA Device:', torch.cuda.get_device_name(0))"
    
    # Try to build torch-tensorrt
    mkdir lib
    cd lib

    # Check if submodule exists 
    if [ -d "TensorRT" ]; then
        echo "TensorRT submodule exists"
    else
        git submodule add --branch release/2.6 https://github.com/pytorch/TensorRT.git # Try to use release/2.6 (latest)
    fi
    
    pip install --upgrade setuptools # Ensure setuptools is up to date

    cd TensorRT
    git checkout release/2.6
    git pull

    cuda_version=$(nvcc --version | grep Cuda | grep release | cut -d ',' -f 2 | sed -e 's/ release //g')
    export TORCH_INSTALL_PATH=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))")
    export SITE_PACKAGE_PATH=${TORCH_INSTALL_PATH::-6}
    export CUDA_HOME=/usr/local/cuda-${cuda_version}/

    # Replace the MODULE.bazel with the jetpack one
    cat toolchains/jp_workspaces/MODULE.bazel.tmpl | envsubst > MODULE.bazel

    # build and install torch_tensorrt wheel file
    python setup.py --use-cxx11-abi install --user
    cd ..

fi



