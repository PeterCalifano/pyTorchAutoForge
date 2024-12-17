close all
clear
clc

%% TEST: CTorchModelWrapper
charDevice    = 'cpu';
charModelPath = '.';

currentDir = pwd;
cd(fullfile('..','..','..'));
addpath(genpath(fullfile('.', 'pyTorchAutoForge', 'api','matlab')));
cd(currentDir)

try
    objModel = CTorchModelWrapper(charModelPath, charDevice);
catch
    disp('This error should be printed if no remote server is available (correct behaviour).')
    clear objModel
end

% Define wrapper object
% charAddress = "https://dkd7j3xr-50000.euw.devtunnels.ms/";
charAddress = "127.0.0.1";
i32PortNumber = 50000;

objModel = CTorchModelWrapper(charModelPath, charDevice, "charServerAddress", charAddress, ...
    'i32PortNumber', i32PortNumber, 'enumTorchWrapperMode', 'TCP'); %#ok<*NASGU>

clear objModel

% Define image array to send over TCP
ui8Image = uint8(randn(1024, 1024));

% TEST: TensorCommManager (TENSOR MODE)
tensorCommManager = TensorCommManager(charAddress, i32PortNumber, 15, "bInitInPlace", true, ...
                                      "bMULTI_TENSOR", false);
% Test send to server
writtenBytes = tensorCommManager.WriteBuffer(ui8Image); % PASSED, 17-12-2024

% Test read back from server
[dTensorArray, self] = tensorCommManager.ReadBuffer(); % PASSED, 17-12-2024

% Check difference wrt to sent data
transmissionErr = sum(single(ui8Image) - dTensorArray, 'all'); % PASSED, 17-12-2024
assert(transmissionErr == 0, 'Transmission error occurred. Received image does not match sent image.')


% TEST: TensorCommManager (TENSOR MODE)
tensorCommManager_multi = TensorCommManager(charAddress, i32PortNumber, 15, "bInitInPlace", true, ...
                                      "bMULTI_TENSOR", true);





