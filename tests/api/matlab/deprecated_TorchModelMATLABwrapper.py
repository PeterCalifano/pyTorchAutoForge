from pyTorchAutoForge.api.matlab import TorchModelMATLABwrapper, MatlabWrapperConfig
import numpy as np


# TODO both test and class require major updates. Do it in a test-driven development way.
def test_TorchModelMATLABwrapper():

    # Get script path
    import os
    import torch
    import atexit
    file_dir = os.path.dirname(os.path.realpath(__file__))
    import torch.nn as nn

    class SampleCNN(nn.Module):
        def __init__(self):
            super(SampleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=5)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.conv1(x))

    # Create a sample model and trace it
    model = SampleCNN()
    input_sample = torch.rand(1, 3, 256, 256)
    traced_model = torch.jit.trace(model, input_sample)

    # Save the traced model to a temporary path
    module_path = os.path.join(file_dir, 'sample_cnn_traced')
    traced_model.save(module_path + '.pt')

    # Clean up: delete the temporary file after the test
    atexit.register(lambda: os.remove(module_path + '.pt'))

    # Check if model exists
    if not os.path.isfile(module_path + '.pt'):
        raise FileNotFoundError('Model specified by: ',
                                module_path, ': NOT FOUND. Run create_sample_tracedModel.py to create model first.')

    # Define wrapper configuration and wrapper object
    wrapper_config = MatlabWrapperConfig()
    model_wrapper = TorchModelMATLABwrapper(module_path, wrapper_config)

    print('Input shape:', input_sample.shape)
    output = model_wrapper.forward(input_sample)
    assert isinstance(output, np.ndarray)
    assert output.shape == (1, 16, 252, 252)
    print('Output shape:', output.shape)


if __name__ == '__main__':
    test_TorchModelMATLABwrapper()
