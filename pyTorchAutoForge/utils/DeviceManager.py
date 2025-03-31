"""
    DeviceManager module for managing and selecting the optimal computation device.

    This module provides functionality to determine the best device for computation
    based on the system's hardware capabilities and available resources. It includes
    support for CUDA-enabled GPUs, Jetson devices, Apple Silicon (MPS), and CPU as a fallback.

    Functions:
        GetDeviceMulti:
            Determines the optimal device for computation based on available memory
            and compatibility. It prioritizes GPUs with sufficient free memory and
            falls back to MPS or CPU if no suitable GPU is available.

    Classes:
        DeviceManager:
            A placeholder class for managing devices. Currently, it provides a static
            method to retrieve the optimal computation device.

    Constants:
        on_rtd:
            A boolean indicating whether the code is running in the ReadTheDocs environment.
        is_jetson:
            A boolean indicating whether the code is running on a Jetson device.

    Notes:
        - The GetDeviceMulti function uses NVML to query GPU memory information.
        - For Jetson devices, the device selection is simplified to either CUDA or CPU.
        - In the ReadTheDocs environment, a dummy version of GetDeviceMulti is provided
          that always returns "cpu".

    Todo:
        - Improve the Jetson device detection logic for better clarity and accuracy.
        - Optimize NVML initialization and shutdown to reduce overhead.
        - Extend the DeviceManager class for multi-GPU support and additional features.
"""
import torch
import warnings
import platform
from typing import Literal
import os 

# Environment variable defined in ReadTheDocs environment
on_rtd = os.environ.get('READTHEDOCS') == 'True'

# Detect if running on a Jetson device
# TODO (PC) improve this piece of code, it's not expressive enough. is_jetson appears to possibly be true in both if-else branches which is confusing. Also the first GetDeviceMulti() is GetDevice() for a jetson, therefore, re-use that function. Clarify that the if conditional
if torch.cuda.is_available(): # DEVNOTE posed as if because cuda may not be available
    device_name = torch.cuda.get_device_name(0).lower()
    is_jetson = any(keyword in device_name for keyword in [
                    "xavier", "orin", "jetson"])
else:
    is_jetson = "tegra" in platform.uname().machine.lower()  # Tegra-based ARM devices


if not on_rtd:
    if is_jetson:
        # GetDevice for Jetson devices
        def GetDeviceMulti() -> Literal['cuda:0'] | Literal['cpu'] | Literal['mps']:
            if torch.cuda.is_available():
                return "cuda:0"
            return "cpu"

    else:
        # GetDevice for Non-Tegra devices
        import pynvml
        def GetDeviceMulti() -> Literal['cuda:0'] | Literal['cpu'] | Literal['mps']:
            """
            GetDeviceMulti Determines the optimal device for computation based on available memory and compatibility.

            The heuristic used for device selection prioritizes GPUs with sufficient free memory, ensuring efficient computation. 
            It checks all available GPUs and selects the one with the highest free memory that meets the following criteria:
            - At least 30% of the total memory is free (MIN_FREE_MEM_RATIO).
            - At least 3 GB of free memory is available (MIN_FREE_MEM_SIZE).
            If no GPU meets these requirements, it falls back to MPS (for Apple Silicon) or CPU as a last resort.

            Returns:
                Literal['cuda:0'] | Literal['cpu'] | Literal['mps']: 
                    The selected device: a CUDA GPU (e.g., 'cuda:0'), MPS (for Apple Silicon), or CPU.
            """

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
                    total_memory = pynvml.nvmlDeviceGetMemoryInfo(
                        handle).total / (1024 ** 3)     # Memory in GB
                    free_memory = pynvml.nvmlDeviceGetMemoryInfo(
                        handle).free / (1024 ** 3)  # Memory in GB

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
                warnings.warn(
                    "CUDA is available, but no GPU meets the minimum requirements. Using CPU instead.")

            return "cpu"
else:
    # Define dummy version of GetDeviceMulti for ReadTheDocs
    def GetDeviceMulti() -> Literal['cuda:0'] | Literal['cpu'] | Literal['mps']:
        return "cpu"    

# Temporary placeholder class (extension wil be needed for future implementations, e.g. multi GPUs)
class DeviceManager():
    def __init__(self):
        pass

    @staticmethod
    def GetDevice():
        return GetDeviceMulti()


# TODO move to tests folder
def test_GetDevice_():
    # Test the GetDevice function
    assert GetDeviceMulti() == "cuda:0" or GetDeviceMulti(
    ) == "cpu" or GetDeviceMulti() == "mps" 
    print("GetDevice() test passed. Selected device: ", GetDeviceMulti())


if __name__ == "__main__":
    test_GetDevice_()
