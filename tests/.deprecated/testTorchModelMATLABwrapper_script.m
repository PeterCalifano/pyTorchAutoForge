close all
clear
clc

% terminate(pyenv)
if pyenv().Status == matlab.pyclient.Status.Terminated
    pythonEnvPath = fullfile('..', '..', '..', '.venvTorch', 'bin', 'python');
    pyenv("Version", pythonEnvPath, 'ExecutionMode', 'OutOfProcess')
end


%% TEST MATLAB torch wrapper
% pythonEnvPath = fullfile('..', '..', '..', '.venvTorch', 'bin', 'python');
% pyenv("Version", pythonEnvPath, 'ExecutionMode', 'OutOfProcess')
            
% Try importing required modules
np = py.importlib.import_module('numpy');
autoforge = py.importlib.import_module('pyTorchAutoForge');
autoforge_matlab_wrapper = py.importlib.import_module('pyTorchAutoForge.api.matlab'); % Load the specific module
py.importlib.reload(autoforge);
py.importlib.reload(autoforge_matlab_wrapper);

% Define model path
model_path = '/home/peterc/devDir/pyTorchAutoForge/tests/data/sample_cnn_traced_cpu'; % Set this to the path of your PyTorch model checkpoint

% ACHTUNG: model is assumed to be traced using torch.jit.trace

% Instantiate the PyTorch model wrapper
% This class, loaded from pyTorchAutoForge.api.matlab, should have a constructor requiring the model path
% wrapper_config = autoforge_matlab_wrapper.MatlabWrapperConfig(DEBUG_MODE = true);

% torch_model_wrapper = autoforge_matlab_wrapper.TorchModelMATLABwrapper(model_path, wrapper_config);
torch_model_wrapper = autoforge_matlab_wrapper.test_TorchModelMATLABwrapper();

% Example input tensor - set to an example shape that matches the model's expected input
input_tensor = np.random.rand(1, 3, 224, 224); % Shape might vary depending on your model, e.g., [batch_size, channels, height, width]

% Convert the input tensor to a torch tensor using MATLAB to Python conversion
torch_input_tensor = py.torch.tensor(input_tensor);

% Perform a forward pass through the model using the forward method of the wrapper
% Assuming the forward method works similar to PyTorch's nn.Module, which returns the output tensor
output_tensor = torch_model_wrapper.forward(torch_input_tensor);

% Convert the Python tensor back to MATLAB format for visualization or further processing
output_array = double(py.array.array('d', py.numpy.nditer(output_tensor)));
disp(output_array);


%% TEST: TCP branch (todo)
% All scripts path: /home/peterc/devDir/MachineLearning_PeterCdev/matlab/LimbBasedNavigationAtMoon

% Initialize communication with server
serverAddress = '127.0.0.1';
portNumber = uint16(50000);


% Create communication handler and initialize directly
commHandler = CommManager(serverAddress, portNumber, 20, "bInitInPlace", true);




