import tensorrt as trt
import torch 
import numpy as np
import sys 
import pycuda as cuda

# Build the RRT Engine from ONNX File

def build_engine_from_onnx(onnx_path, engine_path, precision="fp32"):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network( 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) )

    # Parse ONNX
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(f"ONNX parse error: {parser.get_error(error)}")
            sys.exit(1)

    config = builder.create_builder_config()
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)

    # Set memory pool limit (new API)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    # Add optimization profile for dynamic shapes
    profile = builder.create_optimization_profile()

    # Get input tensor name - assuming first input
    input_tensor = network.get_input(0)
    input_name = input_tensor.name

    # Set min, opt, max shapes
    # Modify these shapes according to your model's requirements
    min_shape = (1, NUM_CLASS, H, W)
    opt_shape = (1, NUM_CLASS, H, W)
    max_shape = (1, NUM_CLASS, H, W)

    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    # Use new API for building engine
    engine = builder.build_serialized_network(network, config)

    # Save engine
    with open(engine_path, 'wb') as f:
        f.write(engine)

    return engine

# Set the RT Engine for Inference


class TensorRTInference:
    def __init__(self, engine_path):
        # Load TRT engine
        logger = trt.Logger(trt.Logger.INFO)
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()

        # Allocate buffers
        self.inputs = []
        self.outputs = []
        self.bindings = []

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to bindings
            self.bindings.append(int(device_mem))
            # Append to the appropriate list
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

    def infer(self, input_data):
        # Transfer input data to device
        cuda.memcpy_htod(self.inputs[0]['device'], input_data)
        # Run inference
        self.context.execute_v2(self.bindings)
        # Transfer predictions back
        for out in self.outputs:
            cuda.memcpy_dtoh(out['host'], out['device'])
        return [out['host'] for out in self.outputs]


# Usage example
engine_path = "model.trt"
infer = TensorRTInference(engine_path)

# Prepare your input data (example)
input_data = np.random.random((1, 3, H, W)).astype(np.float32)

# Run inference
output = infer.infer(input_data)

def convert_to_onnx(model, input_shape, onnx_path):
    """
    Convert PyTorch model to ONNX format.
   
    Args:
        model: The loaded PyTorch model
        input_shape: Tuple of input shape (batch_size, channels, height, width)
        onnx_path: Path to save the ONNX model
    """
    model.eval()  # Set to evaluation mode

    # Create dummy input
    dummy_input = torch.randn(
        input_shape, device='cuda' if torch.cuda.is_available() else 'cpu')

    # Export to ONNX
    torch.onnx.export(
        model,                     # model being run
        dummy_input,              # model input
        onnx_path,                # where to save the model
        export_params=True,       # store the trained parameter weights inside the model file
        opset_version=11,         # the ONNX version to export the model to
        do_constant_folding=True,  # optimize constant folding for better performance
        input_names=['input'],    # the model's input names
        output_names=['output'],  # the model's output names
        dynamic_axes={
            'input': {0: 'batch_size'},    # variable length axes
            'output': {0: 'batch_size'}
        }
    )
    print(f"Model exported to {onnx_path}")
