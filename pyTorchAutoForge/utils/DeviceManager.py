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
import platform
from typing import Literal
import os 
import functools

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


def _handle_selection_override(selection_override: torch.device | str | None) -> torch.device | Literal['cuda'] | Literal['cpu'] | Literal['mps'] | Literal['xpu'] | None:
    """
    _handle_selection_override _summary_

    _extended_summary_

    :param selection_override: _description_
    :type selection_override: torch.device | str | None
    :raises ValueError: _description_
    :return: _description_
    :rtype: torch.device | str | None
    """
    if selection_override is not None and isinstance(selection_override, str):
        # If a specific device is requested, return it directly
        if "cuda" in selection_override:
            return torch.device(selection_override)
        elif selection_override.lower() in ['cpu']:
            return 'cpu'
        elif selection_override.lower() in ['mps']:
            return 'mps'
        elif selection_override.lower() in ['xpu']:
            return 'xpu'
        else:
            raise ValueError(
                f"Invalid device selection override: {selection_override}")

if not on_rtd:
    if is_jetson:
        # GetDevice for Jetson devices
        @functools.lru_cache(maxsize=1)
        def GetDeviceMulti(selection_override: torch.device | str | None = None,
                           expected_max_vram_gb: float | None = None) -> torch.device | Literal['cuda'] | Literal['cpu'] | Literal['mps'] | Literal['xpu']:
            selection_override = _handle_selection_override(selection_override)
            if selection_override is not None:
                return selection_override

            if torch.cuda.is_available():
                return "cuda"
            return "cpu"

        def Wait_for_gpu_memory(min_free_mb: int, 
                                gpu_index: int | str | torch.device = 0,
                                check_interval_in_seconds: int = 30,
                                wait_for_seconds_after_ok: int = 0) -> None:
            pass # Dummy function for rtd-docs

        def GetCudaAvailability(expected_max_vram: float | None = None,
                                min_mem_free_ratio: float | None = None) -> tuple[Literal[False], torch.device] | tuple[Literal[True], torch.device]:
            return True, torch.device("cuda:0")  # Always return cuda for Jetson devices

    else:
        # GetDevice for Non-Tegra devices
        try:
            import pynvml

            def Wait_for_gpu_memory(min_free_mb: int, 
                                    gpu_index: int | str | torch.device = 0, 
                                    check_interval_in_seconds: int = 30,
                                    wait_for_seconds_after_ok: int = 0) -> None:
                """
                Wait_for_gpu_memory waits until at least `min_free_mb` of GPU memory is available on the specified GPU.

                Args:
                    min_free_mb (int): Minimum free memory in MB required to proceed.
                    gpu_index (int): Index of the GPU to monitor.
                    check_interval (int): Time in seconds to wait between memory checks.

                This function pauses execution until the specified amount of free memory is available.
                """
                import time
                if isinstance(gpu_index, torch.device):
                    try:
                        gpu_index = int(str(gpu_index).split(":")[-1])
                    except:
                        # If specified device does not contain index (e.g. "cuda"), query function to get availability
                        is_available, gpu_index = GetCudaAvailability()
                        gpu_index = gpu_index.index

                elif isinstance(gpu_index, str):
                    if "cuda" in gpu_index:
                        gpu_index = gpu_index.split(":")[-1]
                        if not gpu_index.isdigit():
                            # If specified device does not contain index (e.g. "cuda"), query function to get availability
                            is_available, gpu_index = GetCudaAvailability()
                            gpu_index = gpu_index.index

                    else:
                        raise ValueError(
                            f"Invalid gpu_index string format: {gpu_index}")

                # Get handle to the specified CUDA device
                pynvml.nvmlInit()
                gpu_index = int(gpu_index)
                device_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

                while True:
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(device_handle)
                    free_mb = int(mem_info.free) / 1024 / 1024

                    if free_mb >= min_free_mb:
                        print(f"GPU {gpu_index}: {free_mb:.1f} MB free, proceeding.")
                        break

                    else:
                        print(f"Waiting for GPU {gpu_index} memory to free up... ({free_mb:.1f} MB free, need {min_free_mb} MB)")
                        time.sleep(check_interval_in_seconds)

                pynvml.nvmlShutdown()  # Shutdown NVML after use

                # Wait for additional wait_for_seconds_after_ok seconds after a successful check
                if wait_for_seconds_after_ok > 0:
                    print(
                        f"Waiting for additional {wait_for_seconds_after_ok} seconds after successful memory check...")
                    time.sleep(wait_for_seconds_after_ok)

                # Recheck mem again without wait
                #Wait_for_gpu_memory(min_free_mb=min_free_mb, 
                #                    gpu_index=gpu_index, check_interval_in_seconds=check_interval_in_seconds, wait_for_seconds_after_ok=0)


            def GetCudaAvailability(expected_max_vram_gb : float | None = None,
                                    min_mem_free_ratio: float | None = None) -> tuple[Literal[False], torch.device] | tuple[Literal[True], torch.device]:
                """
                GetCudaAvailability prints GPU info, selects the GPU with the most free memory,
                and returns its device string ("cuda:<idx>") or "cpu" if no CUDA device is available.
                """
                if not torch.cuda.is_available():
                    print("CUDA is not available. Suggesting CPU...")
                    return False, torch.device("cpu")

                has_pynvml = False
                try:
                    pynvml.nvmlInit()
                    has_pynvml = True

                except ImportError:
                    # Fallback: list device count and names
                    count = torch.cuda.device_count()
                    print(f"CUDA available but pynvml is not. Found {count} GPU(s).")
                    for i in range(count):
                        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                    return True, torch.device("cuda:0")

                device_count = torch.cuda.device_count()
                print(f"Number of GPUs: {device_count}")

                best_gpu = None
                max_free = 0.0

                for idx in range(device_count):

                    handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

                    free_gb = mem.free / (1024 ** 3)
                    total_gb = mem.total / (1024 ** 3)
                    name = torch.cuda.get_device_name(idx)

                    print(f"GPU {idx}: {name}, Free: {free_gb:.3f} GB / Total: {total_gb:.3f} GB")

                    # Ratio of free memory with respect to total memory
                    free_memory_ratio = free_gb / total_gb

                    # Select the GPU with most free memory that meets the minimum requirements
                    min_mem_condition = True
                    if min_mem_free_ratio is not None:
                        min_mem_condition = free_memory_ratio >= min_mem_free_ratio 

                    if expected_max_vram_gb is not None:
                        min_mem_condition = min_mem_condition and free_gb > expected_max_vram_gb

                    if (min_mem_condition) and free_gb > max_free:
                        max_free = free_gb
                        best_gpu = idx

                if has_pynvml:
                    pynvml.nvmlShutdown()

                if best_gpu is not None:
                    print(f"Selecting GPU {best_gpu} with {max_free:.3f} GB free memory")
                    return True, torch.device(f"cuda:{best_gpu}")

                print("No GPU matching all conditions found. Suggesting CPU...")
                return False, torch.device("cpu")
            
            @functools.lru_cache(maxsize=1)
            def GetDeviceMulti(selection_override : torch.device | str | None = None, 
                               expected_max_vram_gb: float | None = None) -> torch.device | Literal['cuda'] | Literal['cpu'] | Literal['mps'] | Literal['xpu']:
                """
                GetDeviceMulti Determines the optimal device for computation based on available memory and compatibility.

                The heuristic used for device selection prioritizes GPUs with sufficient free memory, ensuring efficient computation.
                It checks all available GPUs and selects the one with the highest free memory that meets the following criteria:
                - At least 30% of the total memory is free (MIN_FREE_MEM_RATIO).
                - At least 3 GB (or selected amount) of free memory is available (MIN_FREE_MEM_SIZE).
                If no GPU meets these requirements, it falls back to MPS (for Apple Silicon) or CPU as a last resort.

                Returns:
                    Literal['cuda'] | Literal['cpu'] | Literal['mps']:
                        The selected device: a CUDA GPU (e.g., 'cuda'), MPS (for Apple Silicon), or CPU.
                """
                # DEVNOTE: Small overhead at each call using init-shutdown this way. Can be improved by init globally and shutting down at python program exit (atexit callback)

                selection_override = _handle_selection_override(selection_override)
                if selection_override is not None:
                    return selection_override

                if expected_max_vram_gb is not None and not (isinstance(expected_max_vram_gb, (float, int))):
                    raise TypeError(
                        f"Expected expected_max_vram to be float or int, got {type(expected_max_vram_gb)} instead.")

                MIN_FREE_MEM_RATIO = 0.3
                # Minimum free memory in GB
                MIN_FREE_MEM_SIZE = 3 if expected_max_vram_gb is None else expected_max_vram_gb

                if torch.cuda.is_available():
                    # Query function to evaluate cuda availability
                    is_available, selected_gpu = GetCudaAvailability(expected_max_vram_gb=MIN_FREE_MEM_SIZE, 
                                                       min_mem_free_ratio=MIN_FREE_MEM_RATIO)

                    if selected_gpu is not None and is_available:
                        return selected_gpu  # type:ignore

                # Check for MPS (for Mac with Apple Silicon)
                if torch.backends.mps.is_available():
                    return "mps"

                # If no GPU is available, return CPU
                if torch.cuda.is_available():
                    print("\033[38;5;208mCUDA is available, but no GPU meets the minimum requirements.\033[0m")

                    invalid_input = True
                    while invalid_input:
                        # Ask to user if he wants to use the CPU
                        usr_input = input(
                            "Run program in CPU? (Y/n): ").strip().lower()

                        if usr_input == 'n' or usr_input == 'no':
                            import sys
                            print("Chosen not to continue. Exiting program...")
                            sys.exit(0)
                        elif usr_input == 'y' or usr_input == 'yes':
                            invalid_input = False
                            print("Chosen to continue. Running with CPU...")
                            return "cpu"
                        else:
                            print(
                                "Invalid input. Please enter 'Y/yes' or 'n/no'.")
                            continue

                return "cpu"


        except ImportError:
            print("\033[38;5;208mpynvml import error. Package is required to use more advanced GetDeviceMulti functionalities memory management. Please install it using 'pip install pynvml'. PTAF will use simplified logic instead.\033[0m")

            # Fall back to simplified logic/dummy functions
            def Wait_for_gpu_memory(min_free_mb: int,
                                    gpu_index: int | str | torch.device = 0,
                                    check_interval_in_seconds: int = 30,
                                    wait_for_seconds_after_ok: int = 0) -> None:
                """
                Wait_for_gpu_memory is a dummy function that does nothing.
                It is used as a placeholder when pynvml is not available.
                """
                usr_input = input(f"pynvml not available. Cannot check for GPU memory availability. Would you like to continue anyway?")

                while True:
                    if usr_input.lower() not in ['y', 'yes', 'n', 'no']:

                        usr_input = input(
                            "Invalid input. Please enter 'Y/yes' or 'n/no': ").strip().lower()
                        continue

                    elif usr_input.lower() in ['n', 'no']:
                        print("Exiting program...")
                        import sys
                        sys.exit(0)

                    else:
                        print(f"Continuing program execution without checking GPU memory availability, using 'cuda' as default device.")
                        break

            @functools.lru_cache(maxsize=1)
            def GetDeviceMulti(selection_override: torch.device | str | None = None,
                               expected_max_vram: float | None = None) -> torch.device | Literal['cuda'] | Literal['cpu'] | Literal['mps'] | Literal['xpu']:
                
                selection_override = _handle_selection_override(selection_override)
                if selection_override is not None:
                    return selection_override
                
                if torch.cuda.is_available():
                    return "cuda"
                return "cpu"
            
            def GetCudaAvailability(expected_max_vram_gb: float | None = None,
                                    min_mem_free_ratio: float | None = None) -> tuple[Literal[False], torch.device] | tuple[Literal[True], torch.device]:
                return False, torch.device("cuda:0")

else:
    # Define dummy version of GetDeviceMulti for ReadTheDocs
    @functools.lru_cache(maxsize=1)
    def GetDeviceMulti(selection_override: torch.device | str | None = None,
                       expected_max_vram_gb: float | None = None) -> torch.device | Literal['cuda'] | Literal['cpu'] | Literal['mps'] | Literal['xpu']:
        return "cpu"    

    def GetCudaAvailability(expected_max_vram_gb: float | None = None,
                            min_mem_free_ratio: float | None = None) -> tuple[Literal[False], torch.device] | tuple[Literal[True], torch.device]:
        return False, torch.device("cpu")

    
# Temporary placeholder class (extension wil be needed for future implementations, e.g. multi GPUs)
class DeviceManager():
    def __init__(self):
        pass

    @staticmethod
    def GetDevice():
        return GetDeviceMulti()


# TODO move to tests folder

if __name__ == "__main__":
    GetCudaAvailability()
