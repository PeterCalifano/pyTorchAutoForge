close all
clear
clc

%% TEST: CTorchModelWrapper
charDevice    = 'cpu';
charModelPath = '.';

currentDir = pwd;
cd(fullfile('..','..','..'));
addpath(genpath(fullfile('.', 'pyTorchAutoForge', 'api','matlab')));
addpath(genpath(fullfile('.', 'lib', 'CommManager4MATLAB','src')));
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
i32PortNumber_multi = 50001;


% Define image array to send over TCP
strDataPath = fullfile("..", "..", "data");
ui8Image = imread(fullfile(strDataPath, "moon_image_testing.png"));
[strImageLabels] = JSONdecoder(fullfile(strDataPath, "moon_labels_testing.json"));

%% TEST: evaluating model directly through TensorCommManager
% Define TensorCommManager instance
tensorCommManager_multi = TensorCommManager(charAddress, i32PortNumber_multi, 15, "bInitInPlace", true, ...
                                      "bMULTI_TENSOR", true);

% Test send to server of array (auto-wrapping)
ui8TensorShapedImage = zeros(1,1,size(ui8Image, 1), size(ui8Image, 2), 'uint8');
ui8TensorShapedImage(1,1,:,:) = ui8Image;

writtenBytes = tensorCommManager_multi.WriteBuffer(ui8TensorShapedImage);

% Get data back from server
[cellTensorArray, tensorCommManager_multi] = tensorCommManager_multi.ReadBuffer();

disp(cellTensorArray);

return
%% TEST: using Torch wrapper object
% Define torch wrapper
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














