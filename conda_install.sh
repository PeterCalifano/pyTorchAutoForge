# Default values
jetson_target=false
editable_mode=false
sudo_mode=false
venv_name="autoforge"
create_conda_env=false

# Parse options using getopt
# NOTE: no ":" after option means no argument, ":" means required argument, "::" means optional argument
OPTIONS=j,v:,s,c,e
LONGOPTIONS=jetson_target,venv_name:,sudo_mode,create_conda_env,editable_mode

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
      jetson_target=1
      echo "Jetson target selected..."
      shift
      ;;
    -v|--venv_name)
      venv_name=$2
      echo "Conda environment name: $venv_name"
      shift 2
      ;;
    -s|--sudo_mode)
      sudo_mode=true
      echo "Sudo mode requested..."
      shift
      ;;
    
    -c|--create_conda_env)
      create_conda_env=true
      echo "Creating and initializing conda environment..."
      shift
      ;;
    -e|--editable_mode)
      editable_mode=true
      echo "Editable mode selected..."
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

if [ $jetson_target = true ] && [ ! $sudo_mode = true ]; then
  echo "Jetson target requires sudo mode. Please use -s option."
  exit 1
fi

if [ $create_conda_env = true ]; then
  # Create and activate conda environment
  conda create -n $venv_name python=3.12
  source $(conda info --base)/etc/profile.d/conda.sh
  conda activate $venv_name
else
  echo "Attempt to activate existing conda environment..."
  
  # Check if conda environment exists else stop
  if conda info --envs | grep -q "$venv_name"; then
    echo "Conda environment $venv_name found. Activating it..."
    conda init bash
    # Activate conda environment
    conda activate $venv_name
  else
    echo "Conda environment $venv_name does not exist. Please create it first or run this script with -c flag."
    exit 1
  fi
fi

sleep 1

if [ $jetson_target = false ] && [ ! -f /usr/local/cuda/lib64/libcusparseLt.so ]; then
    echo "libcusparseLt.so not found. Downloading and installing..."
    # if not exist, download and copy to the directory
    wget https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-sbsa/libcusparse_lt-linux-sbsa-0.5.2.1-archive.tar.xz
    tar xf libcusparse_lt-linux-sbsa-0.5.2.1-archive.tar.xz
    sudo cp -a libcusparse_lt-linux-sbsa-0.5.2.1-archive/include/* /usr/local/cuda/include/
    sudo cp -a libcusparse_lt-linux-sbsa-0.5.2.1-archive/lib/* /usr/local/cuda/lib64/
    rm libcusparse_lt-linux-sbsa-0.5.2.1-archive.tar.xz
    rm -r libcusparse_lt-linux-sbsa-0.5.2.1-archive
fi

if [ $jetson_target = true ]; then

  #pip install -r requirements.txt  # Install dependencies
  #pip install -e .  # Install the package in editable mode

  # Tools for building and installing wheels
  echo "Installing setuptools, twine, and build..."
  pip install setuptools twine build 
  python3 -m ensurepip --upgrade 
  python3 -m pip install --upgrade pip 

  # Install key modules not managed by dependencies installation for versioning reasons
  echo "Installing additional key modules..."
  
  # Remove torch and torchvision 
  pip uninstall -y torch torchvision torchaudio

  # Install torch for Jetson
  pip install torch https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl 

  # Build and install torchvision from source
  # From guide: https://github.com/azimjaan21/jetpack-6.1-pytorch-torchvision-/blob/main/README.md
  git clone https://github.com/pytorch/vision.git
  cd vision
  git checkout tags/v0.20.0
  python3 setup.py install 

  # Clean up
  cd ..
  sudo rm -r vision

  #pip install norse==1.0.0 aestream tonic expelliarmus --ignore-requires-python3   # FIXME: build fails due to "CUDA20" entry

  pip install nvidia-pyindex pycuda 

  # ACHTUNG: this must run correctly before torch_tensorrt
  pip install "nvidia-modelopt[all]" -U --extra-index-url https://pypi.nvidia.com

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

  # Install required python3 packages of torch-tensorrt
  python3 -m pip install -r toolchains/jp_workspaces/requirements.txt # NOTE: Installs the correct version of setuptools. Do not touch it.

  cuda_version=$(nvcc --version | grep Cuda | grep release | cut -d ',' -f 2 | sed -e 's/ release //g')
  export TORCH_INSTALL_PATH=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))")
  export SITE_PACKAGE_PATH=${TORCH_INSTALL_PATH::-6}
  export CUDA_HOME=/usr/local/cuda-${cuda_version}/

  # Replace the MODULE.bazel with the jetpack one # DOUBT: why needed?
  cat toolchains/jp_workspaces/MODULE.bazel.tmpl | envsubst > MODULE.bazel

  # Build and install torch_tensorrt wheel file with CXX11 ABI
  python3 setup.py install --use-cxx11-abi
  cd ../..

  # Finally, build pyTorchAutoForge wheel
  if [ $editable_mode -eq 1 ]; then
      echo "Building and installing pyTorchAutoForge in editable mode..."
      pip install -e .  # Install the package in editable mode
  else
    echo "Building and installing pyTorchAutoForge wheel..."
    # Remove previous build 
    rm -rf dist
    rm -rf build
    rm -rf pyTorchAutoForge.egg-info
    # Build and install
    python3 -m build 
    pip install dist/*.whl  # Install pyTorchAutoForge wheel # FIXME editable mode does not work for this
  fi
    
else
  #pip install -r requirements.txt  # Install dependencies that do not cause issues...
  #python3 -m pip install -r toolchains/jp_workspaces/test_requirements.txt # Required for test cases

  # Tools for building and installing wheels
  echo "Installing setuptools, twine, and build..."
  pip install setuptools twine build 
  python3 -m ensurepip --upgrade 
  python3 -m pip install --upgrade pip 

  # Install key modules not managed by dependencies installation for versioning reasons
  echo "Installing additional key modules..."

  # Build pyTorchAutoForge wheel
  if [ $editable_mode -eq 1 ]; then
      echo "Building and installing pyTorchAutoForge in editable mode..."
      pip install -e .  # Install the package in editable mode
  else
    echo "Building and installing pyTorchAutoForge wheel..."
    # Remove previous build 
    rm -rf dist
    rm -rf build
    rm -rf pyTorchAutoForge.egg-info
    # Build and install
    python3 -m build 
    pip install dist/*.whl  # Install pyTorchAutoForge wheel # FIXME editable mode does not work for this
  fi

  # Install tools for model optimization and deployment
  echo "Installing tools for model optimization and deployment by Nvidia..."
  python3 -m pip install pycuda torch torchvision torch-tensorrt tensorrt "nvidia-modelopt[all]" -U --extra-index-url https://pypi.nvidia.com
fi

# Check installation by printing versions in python3
cd ./tests/.configuration/
python3 -m test_env
cd ../..


