import torch
import pynvml
import warnings

# GetDevice:
def GetDeviceMulti():
    '''Function to get device to run models on. Used by most modules of pyTorchAutoForge'''

    MIN_FREE_MEM_RATIO = 0.3
    MIN_FREE_MEM_SIZE = 3  # Minimum free memory in GB

    if torch.cuda.is_available():
        # Iterate through all available GPUs to check memory availability
        pynvml.nvmlInit()  # Initialize NVML for accessing GPU memory info. 
        # DEVNOTE: Small overhead at each call using init-shutdown this way. Can be improved by init globally and shutting down at python program exit (atexit callback)

        max_free_memory = 0
        selected_gpu = None

        for gpu_idx in range(torch.cuda.device_count()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx) 
            total_memory = pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024 ** 3)     # Memory in GB
            free_memory = pynvml.nvmlDeviceGetMemoryInfo(handle).free / (1024 ** 3)  # Memory in GB
            
            # Ratio of free memory with respect to total memory
            free_memory_ratio = free_memory / total_memory

            # Select the GPU with most free memory that meets the minimum requirements)
            if free_memory_ratio >= MIN_FREE_MEM_RATIO and free_memory > MIN_FREE_MEM_SIZE and free_memory > max_free_memory:
                max_free_memory = free_memory
                selected_gpu = gpu_idx
        
        pynvml.nvmlShutdown()  # Shutdown NVML

        if selected_gpu is not None:
            return f"cuda:{selected_gpu}"

    # Check for MPS (for Mac with Apple Silicon)
    if torch.backends.mps.is_available():
        return "mps"

    # If no GPU is available, return CPU
    if torch.cuda.is_available():
        warnings.warn("CUDA is available, but no GPU meets the minimum requirements. Using CPU instead.") 
        
    return "cpu"


# Temporary placeholder class (extension wil be needed for future implementations, e.g. multi GPUs)
class DeviceManager():
    def __init__(self):
        pass

    @staticmethod
    def GetDevice(self):
        return GetDevice()


def test_GetDevice_():
    # Test the GetDevice function
    assert GetDevice() == "cuda:0" or GetDevice() == "cpu" or GetDevice() == "mps"
    print("GetDevice() test passed. Selected device: ", GetDevice())


if __name__ == "__main__":
    test_GetDevice_()
