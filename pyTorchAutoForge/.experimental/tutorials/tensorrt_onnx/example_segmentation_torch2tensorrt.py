# Fetch data from kaggle
import kagglehub

# Download latest version
path = kagglehub.dataset_download("mateuszbuda/lgg-mri-segmentation")

print("Path to dataset files:", path)

## Key steps
# 1. Convert pretrained model from pytorch to onnx
# 2. Import onnx model to tensorrt
# 3. Apply optimization and generate an engine
# 4. Run inference using tensorrt engine

## Key components
# 1. ONNx parser, 2. Builder, 3. Engine, 4. Logger