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
    pip uninstall -y torch torchvision

    # Install torch for Jetson
    pip install torch https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl

    # Build and install torchvision from source
    # From guide: https://github.com/azimjaan21/jetpack-6.1-pytorch-torchvision-/blob/main/README.md
    git clone https://github.com/pytorch/vision.git
    cd vision
    git checkout tags/v0.20.0
    python3 setup.py install --require-virtualenv

    # Clean up
    cd ..
    sudo rm -r vision

    # Test installation using script
    python -c "import torch; import torchvision; print('Torch Version:', torch.__version__); print('TorchVision Version:', torchvision.__version__); print('CUDA Available:', torch.cuda.is_available()); if torch.cuda.is_available(): print('CUDA Device:', torch.cuda.get_device_name(0))"

    python3 -m pip install tensorrt tensorrt-lean tensorrt-dispatch

fi



