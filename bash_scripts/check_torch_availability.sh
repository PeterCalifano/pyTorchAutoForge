env_name=$1
if [ -z "$env_name" ]; then
  echo "Usage: $0 <env_name>"
  exit 1
fi

source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$env_name"
export CUDA_LAUNCH_BLOCKING=1

# Test python imports
python -c "import torch; print('CUDA availability:', torch.cuda.is_available()); print('Torch device props:', torch.cuda.get_device_properties(0)); print('Torch version:', torch.__version__)"

# Try to make a tensor on CUDA
python -c "import torch; tensor = torch.tensor(2).fill_(3.14).to('cuda'); print('Tensor created on CUDA:', tensor)"