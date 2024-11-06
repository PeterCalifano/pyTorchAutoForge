from pyTorchAutoForge.api.matlab import TorchModelMATLABwrapper, MatlabWrapperConfig
import numpy as np

def test_TorchModelMATLABwrapper():
    # Get script path
    import os
    file_dir = os.path.dirname(os.path.realpath(__file__))

    module_path = os.path.join(file_dir, '../..', 'data/sample_cnn_traced')

    # Check if model exists
    if not os.path.isfile(module_path + '.pt'):
        raise FileNotFoundError('Model specified by: ',
                                module_path, ': NOT FOUND. Run create_sample_tracedModel.py to create model first.')

    # Define wrapper configuration and wrapper object
    wrapper_config = MatlabWrapperConfig()

    model_wrapper = TorchModelMATLABwrapper(module_path, wrapper_config)

    # Test forward method
    input_sample = np.random.rand(1, 3, 256, 256)

    print('Input shape:', input_sample.shape)
    output = model_wrapper.forward(input_sample)
    assert isinstance(output, np.ndarray)
    assert output.shape == (1, 16, 252, 252)
    print('Output shape:', output.shape)


if __name__ == '__main__':
    test_TorchModelMATLABwrapper()
