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
i32PortNumber_multi = 50003; % 50001

%% TEST: evaluating model directly through TensorCommManager
% Define TensorCommManager instance
tensorCommManager_multi = TensorCommManager(charAddress, i32PortNumber_multi, 100, "bInitInPlace", true, ...
    "bMULTI_TENSOR", true);

ui8TestID = 1;

if ui8TestID == 0
    % Define image array to send over TCP
    strDataPath = fullfile("..", "..", "data");
    ui8Image = imread(fullfile(strDataPath, "moon_image_testing.png"));
    [strImageLabels] = JSONdecoder(fullfile(strDataPath, "moon_labels_testing.json"));


    % Test send to server of array (auto-wrapping)
    ui8TensorShapedImage = zeros(1,1,size(ui8Image, 1), size(ui8Image, 2), 'uint8');
    ui8TensorShapedImage(1,1,:,:) = ui8Image;

    writtenBytes = tensorCommManager_multi.WriteBuffer(ui8TensorShapedImage);

    % Get data back from server
    [cellTensorArray, tensorCommManager_multi] = tensorCommManager_multi.ReadBuffer();

    disp(cellTensorArray);
    strCentroidRangePredictions = cell2struct(cellTensorArray, {'Predictions'});
    ShowCentroidRangePredictions(ui8Image, strCentroidRangePredictions, strImageLabels);

elseif ui8TestID == 1

    % Define image array to send over TCP
    % strDataPath = fullfile("/home/peterc/devDir/ML-repos/pyTorchAutoForge/tests/data/test_images", "Bennu");
    % ui8Frame_1 = imread(fullfile(strDataPath, "000001.png"));
    % ui8Frame_2 = imread(fullfile(strDataPath, "000002.png"));

    strDataPath = fullfile("/home/peterc/devDir/nav-backend/simulationCodes/data/datasets/TestCase_ItokawaRCS1_RTO_3t1_J11p0_45dt/");
    ui8Frame_1 = rgb2gray(imread(fullfile(strDataPath, "000001.png"), 'png'));
    ui8Frame_2 = rgb2gray(imread(fullfile(strDataPath, "000050.png")));
        
    % Test send to server of array (auto-wrapping)
    ui8TensorShapedFrame_1 = zeros(1,1,size(ui8Frame_1, 1), size(ui8Frame_1, 2), 'uint8');
    ui8TensorShapedFrame_2 = zeros(1,1,size(ui8Frame_2, 1), size(ui8Frame_2, 2), 'uint8');

    ui8TensorShapedFrame_1(1,1,:,:) = ui8Frame_1;
    ui8TensorShapedFrame_2(1,1,:,:) = ui8Frame_2;

    cellTensorImages = {ui8TensorShapedFrame_1, ui8TensorShapedFrame_2};
    writtenBytes = tensorCommManager_multi.WriteBuffer(cellTensorImages);

    % Return output dictionary ['keypoints0', 'scores0', 'descriptors0', 'keypoints1', 'scores1',
    % 'descriptors1', 'matches0', 'matches1', 'matching_scores0', 'matching_scores1']

    % Get data back from server
    [cellTensorArray, tensorCommManager_multi] = tensorCommManager_multi.ReadBuffer();

    dKeypoints0     = cellTensorArray{1};
    ui32Matches0    = cellTensorArray{7};
    dMatchingScore0 = cellTensorArray{9};

    dKeypoints1     = cellTensorArray{4};
    ui32Matches1    = cellTensorArray{8};

    % Get matched keypoints
    bValidMatch = ui32Matches0 > -1 & dMatchingScore0 > 0.95;
    dMatchedKps0 = dKeypoints0(bValidMatch, :);
    dMatchedKps1 = dKeypoints1(ui32Matches0(bValidMatch), :);

    % Show matchings
    ShowFeatureMatchingsPredictions(ui8Frame_1, ui8Frame_2, ...
        dMatchedKps0, dMatchedKps1);
end
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








