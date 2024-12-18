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
i32PortNumber_multi = 50001;

objModel = CTorchModelWrapper(charModelPath, charDevice, "charServerAddress", charAddress, ...
    'i32PortNumber', i32PortNumber, 'enumTorchWrapperMode', 'TCP'); %#ok<*NASGU>

clear objModel

% Define image array to send over TCP
ui8Image = uint8(randn(1024, 1024));

%% TEST: TensorCommManager (TENSOR MODE)
tensorCommManager = TensorCommManager(charAddress, i32PortNumber, 15, "bInitInPlace", true, ...
                                      "bMULTI_TENSOR", false);
% Test send to server
writtenBytes = tensorCommManager.WriteBuffer(ui8Image); % PASSED, 17-12-2024

% Test read back from server
[dTensorArray, tensorCommManager] = tensorCommManager.ReadBuffer(); % PASSED, 17-12-2024

% Check difference wrt to sent data
transmissionErr = sum(single(ui8Image) - dTensorArray, 'all'); % PASSED, 17-12-2024
assert(transmissionErr == 0, 'Transmission error occurred. Received image does not match sent image.')


%% TEST: TensorCommManager (MULTI-TENSOR MODE)
tensorCommManager_multi = TensorCommManager(charAddress, i32PortNumber_multi, 15, "bInitInPlace", true, ...
                                      "bMULTI_TENSOR", true);
% Test send to server of array (auto-wrapping)
writtenBytes = tensorCommManager_multi.WriteBuffer(ui8Image); % PASSED, 18-12-2024

% Test read back from server
[cellTensorArray, tensorCommManager_multi] = tensorCommManager_multi.ReadBuffer(); % PASSED, 18-12-2024

% Check difference wrt to sent data
transmissionErr = sum(single(ui8Image) - cellTensorArray{1,1}, 'all'); % PASSED, 18-12-2024
assert(transmissionErr == 0, 'Transmission error occurred. Received image does not match sent image.')

%% Test send/receive of multi-dim array (TENSOR-MODE)
dMultiTensor = randn(10, 10, 5);

writtenBytes = tensorCommManager.WriteBuffer(dMultiTensor); % PASSED, 18-12-2024

% Test read back from server
[dTensorArray, tensorCommManager] = tensorCommManager.ReadBuffer(); % PASSED, 18-12-2024

% Check difference wrt to sent data
transmissionErr = sum(single(dMultiTensor) - dTensorArray, 'all'); % PASSED, 18-12-2024
assert(transmissionErr == 0, 'Transmission error occurred. Received tensor does not match sent tensor.')


%% Test send/receive of multi-dim array cell (MULTI-TENSOR-MODE)

writtenBytes = tensorCommManager_multi.WriteBuffer({ui8Image, dMultiTensor}); % PASSED, 18-12-2024

% Test read back from server
[cellTensorArray, tensorCommManager_multi] = tensorCommManager_multi.ReadBuffer(); % PASSED, 18-12-2024

% Check difference wrt to sent data
transmissionErr_1 = sum(single(ui8Image) - cellTensorArray{1,1}, 'all'); % PASSED, 18-12-2024
transmissionErr_2 = sum(single(dMultiTensor) - cellTensorArray{2}, 'all'); % PASSED, 18-12-2024

assert(transmissionErr_1 == 0, 'Transmission error occurred. Received tensor does not match sent tensor.')
assert(transmissionErr_2 == 0, 'Transmission error occurred. Received tensor does not match sent tensor.')


fprintf("\nAll tests passed.\n")









