import threading
from click import pause
from numpy import ndarray
from numpy.typing import NDArray
import numpy as np
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import os
import colorama 

class ModelProfilerHelper():
    """
    A class to profile PyTorch models.

    Attributes:
    -----------
    model : torch.nn.Module
        The PyTorch model to be profiled.
    device : str
        The device to run the model on ('cpu' or 'cuda').
    last_prof : torch.profiler.profile
        The last profiling result.
    output_prof_filename : str or None
        The filename to save the profiling result.
    with_stack : bool
        Whether to record stack information.
    input_sample : torch.Tensor or None
        The input sample for the model.
    activities : list
        The activities to profile (default is [ProfilerActivity.CPU]).
    record_shapes : bool
        Whether to record tensor shapes.

    Methods:
    --------
    __init__(model, input_shape_or_sample, device='cpu', activities=None, record_shapes=False, output_prof_filename=None, with_stack=False):
        Initializes the ModelProfiler with the given model and input sample or shape.

    run_prof(activities=None, record_shapes=False, input_sample=None):
        Runs the profiler on the model with the given input sample.
    """

    def __init__(self, model: torch.nn.Module, 
                 input_shape_or_sample: tuple[int, ...] | NDArray[np.floating | np.integer] | torch.Tensor,
                 device: str = 'cpu', 
                 activities: tuple[ProfilerActivity, ...] = (ProfilerActivity.CPU,),
                 record_shapes: bool = False, 
                 output_prof_filename: str | None = None, 
                 with_stack: bool = False):
        # Store data
        self.model = model
        self.device = device
        self.last_prof = None 
        self.output_prof_filename = output_prof_filename
        self.with_stack = with_stack
        self.input_sample = None 

        if isinstance(activities, ProfilerActivity):
            activities = (activities,) # Convert to tuple

        self.activities = list(activities)
        self.record_shapes = record_shapes

        if isinstance(input_shape_or_sample, tuple):
            # If input is a list or tuple indicating shape, generate random
            self.input_sample = torch.randn((1, *input_shape_or_sample[1:]))
        else:
            if isinstance(input_shape_or_sample, np.ndarray):
                self.input_sample = torch.from_numpy(input_shape_or_sample)
            elif isinstance(input_shape_or_sample, torch.Tensor):
                # Input is a sample of torch tensor, store it
                self.input_sample = input_shape_or_sample
            else:
                raise TypeError("Input must be a list, tuple specifying the input sizes or a sample as ndarray or torch.Tensor.")

        # Move model and data to device
        self.model.to(self.device)

        if self.input_sample is not None:
            self.input_sample = self.input_sample.to(self.device)

    def run_prof(self, activities: tuple[ProfilerActivity, ...] | None = None,
                 record_shapes: bool = False, 
                 input_sample: torch.Tensor | None = None):

        if input_sample is not None:
            # Store input sample
            self.input_sample = input_sample.to(self.device)

        if self.input_sample is None:
            raise ValueError("Input sample is None. Please provide a sample to run profiling!")

        # Get default values from init, if not provided
        if activities is not None:
            self.activities = list(activities)

        if record_shapes is not None:
            self.record_shapes = record_shapes

        # Set model to eval()
        self.model.eval()

        # Run profiling in inference mode
        with profile(activities=self.activities, record_shapes=self.record_shapes, with_stack=self.with_stack) as prof:
            with record_function("model_inference"):
                self.model(self.input_sample)

        # Print a summary of the profiling
        # TODO: add custom "sort_by"
        print(prof.key_averages().table(sort_by=f"{self.device}_time_total", row_limit=20))

        # Store profile object
        self.last_prof = prof

        # Save profile to file if filename is provided
        if self.output_prof_filename is not None:
            prof.export_chrome_trace(self.output_prof_filename)

        return prof
        
    def make_summary(self):
    # TODO extend method, this is only the first basic version  
        import torchinfo # Conditional import
        if self.input_sample is not None:
            input_size = self.input_sample.shape
            input_type = self.input_sample.dtype
        else:
            raise ValueError("Input sample is None. Cannot generate summary.")

        model_summary = torchinfo.summary(model=self.model, 
                                          input_size=input_size, 
                                          device=self.device, 
                                          col_names=("input_size", "output_size", "num_params", "mult_adds"))

        return model_summary
    
    @staticmethod
    def make_netron_diagram(model_path : str) -> None:
        import netron # Conditional import
        # Check extension of model path
        (model_path_root, model_ext) = os.path.splitext(model_path)

        if model_ext == '.pth':
            raise ValueError(f"{colorama.Fore.RED}PyTorch model checkpoint (.pth) cannot be used with Netron. Please convert the model to .onnx or a traced/scripted PyTorch model (.pt) format. This version does not support automatic conversion.{colorama.Style.RESET_ALL}")

        if model_ext not in ['.onnx', '.pt']:
            raise ValueError(f"{colorama.Fore.RED}Model path must have extension '.onnx' or '.pt'.{colorama.Style.RESET_ALL}")
        
        # Start netron server on a new thread in daemon mode (kill when main thread exits)
        sys_thread = threading.Thread(
            target=netron.start, 
            args=(model_path, ('localhost', 65511), True),
            daemon=True 
        )

        try:
            sys_thread.start()
            # Print info to open netron server 
            print(f"Netron server started on localhost:65511. Open in browser: http://localhost:65511")
        except Exception as e:
            print(f"An error occurred while starting the Netron local server: {e}")


        input_value = ""
        while input_value.lower() not in ['y', 'yes']:
            input_value = input("Script execution paused as Netron server is opened on a thread in daemon mode. Thread will terminate at script termination. \nInput Y to continue script execution... \n\n")

            if input_value not in ['y', 'yes']:
                print("Invalid input. Please enter 'Y' or 'yes'.")

        
if __name__ == "__main__":
    pass
