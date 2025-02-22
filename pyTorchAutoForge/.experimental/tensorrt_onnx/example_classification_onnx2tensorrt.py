# %% Script implementing steps from reference tutorial: https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide.html#the-tensorrt-ecosystem. It shows the full workflow from a pretrained model in pytorch, converted through ONNx, to a tensorrt engine to be used in Python.

# NOTE: to get and untar the model, run:
# wget https://download.onnxruntime.ai/onnx/models/resnet50.tar.gz
# tar xzf resnet50.tar.gz

# Move to the directory containing this script if not already there
import os
if os.getcwd().split("/")[-1] != "tensorrt_onnx":
    os.chdir("pyTorchAutoForge/.experimental/tensorrt_onnx")

# %% Prepare model for inference as standalone runtime in Python
import numpy as np
PRECISION = np.float32

# Call trtexec to convert the model to a TensorRT engine
import subprocess, os
path_to_onnx_model = os.path.join("resnet50v2", "resnet50v2.onnx")
path_to_tensorrt_engine = os.path.join("resnet50v2", "resnet50v2.engine")

if os.path.isfile(path_to_onnx_model):
    # Check if the engine file already exists
    if not os.path.isfile(path_to_tensorrt_engine):
        # Convert the ONNX model to a TensorRT engine
        subprocess.run(args=["bash", "convert_onnx_model_trtexec.sh", "-p", path_to_onnx_model, "-o", path_to_tensorrt_engine])
else:
    print(f"Error: ONNX model file does not exist at {path_to_onnx_model}")

#%% Load it using TensorRT onnx_helper (specific for this example)
# Copied from onnx_helper.py in TensorRT/quickstart/IntroNotebooks
import tensorrt as trt

# NOTE: this import requires installation: conda install -c conda-forge libstdcxx-ng
# This is because of mismatch of libstdc++ versions between the system and the one used to compile TensorRT, that would cause:
# ImportError: /home/peterc/miniconda3/envs/autoforge/lib/python3.11/site-packages/zmq/backend/cython/../../../../.././libstdc++.so.6: version GLIBCXX_3.4.32' not found (required by /home/peterc/miniconda3/envs/autoforge/lib/python3.11/site-packages/pycuda/_driver.cpython-311-x86_64-linux-gnu.so)
import pycuda._driver as cuda
import pycuda.autoinit # This is needed for initializing CUDA context automatically

# NOTE: alternatively, instead of usinc pycuda.autoinit, you can use the following:
# cuda.init() # Initialize CUDA context
# cuda_ctx = cuda.Device(0).make_context() # Create a context on the first GPU
# After all codes have run, call cuda_ctx.pop() to release the context

class ONNXClassifierWrapper():
    def __init__(self, file, target_dtype = np.float32):
        
        self.target_dtype = target_dtype
        self.num_classes = 1000
        self.load(file)
        
        self.stream = None
      
    def load(self, file):
        """Load tensorrt engine from .engine file (output of trtexec)"""
        with open(file, "rb") as f:
            print('Deserializing engine file...')
            # Create runtime session
            self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 
            # Deserialize engine (this actually reads the engine file bytes stream)
            self.engine = self.runtime.deserialize_cuda_engine(f.read()) 
            print ('Engine deserialized. Starting CUDA execution context...')
            # Create execution context for CUDA
            self.context = self.engine.create_execution_context()
            print ('CUDA execution context created.')
        
    def allocate_memory(self, batch):
        """Allocate memory for input and output for the specific datatype"""
        self.output = np.empty(self.num_classes, dtype = self.target_dtype) # Need to set both input and output precisions to FP16 to fully enable FP16

        # Allocate device memory (CUDA buffers on device)
        self.d_input = cuda.mem_alloc(1 * batch.nbytes)
        self.d_output = cuda.mem_alloc(1 * self.output.nbytes)
        
        tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        assert(len(tensor_names) == 2)

        # Bind tensor names to memory addresses of device buffers
        self.context.set_tensor_address(tensor_names[0], int(self.d_input))
        self.context.set_tensor_address(tensor_names[1], int(self.d_output))

        self.stream = cuda.Stream()

        
    def predict(self, batch): # result gets copied into output
        """Run inference on the model"""
        # Initialize memory and stream if not previously done
        if self.stream is None:
            self.allocate_memory(batch)
            
        # Transfer input data to device
        cuda.memcpy_htod_async(self.d_input, batch, self.stream)
        # Execute model
        self.context.execute_async_v3(self.stream.handle)
        # Transfer predictions back
        cuda.memcpy_dtoh_async(self.output, self.d_output, self.stream)

        # Synchronize CUDA stream threads
        self.stream.synchronize()
        return self.output
        
trt_model = ONNXClassifierWrapper(path_to_tensorrt_engine, target_dtype=PRECISION)

# Define dummy input data
input_shape = (1, 3, 224, 224)
dummy_input_batch = np.zeros(input_shape , dtype = PRECISION)

# Run inference
print('Running inference...')
output = trt_model.predict(dummy_input_batch)
print('Output predictions:', output)

# Get argmax of the output
output_predictions = output.argmax()
print('Argmax of output:', output_predictions)
print('Example completed.')