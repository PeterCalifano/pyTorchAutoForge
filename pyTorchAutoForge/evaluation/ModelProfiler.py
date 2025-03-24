from numpy import ndarray
import torch
from torch.profiler import profile, record_function, ProfilerActivity

class ModelProfiler():
    """
    ModelProfiler _summary_

    _extended_summary_
    """
    def __init__(self, model, input_shape_or_sample : list | tuple | ndarray | torch.Tensor, device : str = 'cpu', activities : list | None = None, record_shapes : bool = False, output_prof_filename : str | None = None, with_stack : bool = False):
        # Store data
        self.model = model
        self.device = device
        self.last_prof = None 
        self.output_prof_filename = output_prof_filename
        self.with_stack = False
        self.input_sample = None 

        # Default values
        self.activities = activities if activities is not None else [ProfilerActivity.CPU]
        self.record_shapes = record_shapes

        if isinstance(input_shape_or_sample, (list, tuple)):
            # If input is a list or tuple indicating shape, generate random
            self.input_sample = torch.randn(*input_shape_or_sample)
        else:
            if isinstance(input_shape_or_sample, ndarray):
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

    def run_prof(self, activities: list | None = None, record_shapes : bool = False, input_sample : torch.Tensor | None = None):

        if input_sample is not None:
            # Store input sample
            self.input_sample = input_sample.to(self.device)

        if self.input_sample is None:
            raise ValueError("Input sample is None. Please provide a sample to run profiling!")

        # Get default values from init, if not provided
        if activities is not None:
            self.activities = activities

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
        


if __name__ == "__main__":
    pass
