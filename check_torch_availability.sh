env_name=$1
if [ -z "$env_name" ]; then
  echo "Usage: $0 <env_name>"
  exit 1
fi

source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$env_name"
python -c "import torch; print('CUDA availability:', torch.cuda.is_available()); print('Torch device props:', torch.cuda.get_device_properties(0)); print('Torch version:', torch.__version__)"