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
strCentroidRangePredictions = cell2struct(cellTensorArray, {'Predictions'});
ShowCentroidRangePredictions(ui8Image, strCentroidRangePredictions, strImageLabels);

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

%% LOCAL FUNCTION
function ShowCentroidRangePredictions(ui8Image, strCentroidRangePredictions, strImageLabels)

dCentroidPrediction         = double(strCentroidRangePredictions.Predictions(1:2));
dRangePrediction            = double(strCentroidRangePredictions.Predictions(3));
dApparentRadiusPrediction   = double(strCentroidRangePredictions.Predictions(4));

% Show output
figure();
imshow(ui8Image);
hold on;
plot(dCentroidPrediction(2), dCentroidPrediction(1), 'rx', 'DisplayName', 'Centre of figure Prediction')

% Draw circle with apparent radius (gently offered by GPT4.0)
dTheta = linspace(0, 2*pi, 100); % Angles for the circle
dxCircle = dApparentRadiusPrediction * cos(dTheta) + dCentroidPrediction(2);
dyCircle = dApparentRadiusPrediction * sin(dTheta) + dCentroidPrediction(1);
plot(dxCircle, dyCircle, 'g-', 'DisplayName', 'Predicted conic');

dxCircle_GT = strImageLabels.dRadiusInPix * cos(dTheta) + strImageLabels.dCentroid(2);
dyCircle_GT = strImageLabels.dRadiusInPix * sin(dTheta) + strImageLabels.dCentroid(1);
plot(strImageLabels.dCentroid(2), strImageLabels.dCentroid(1), 'bo', 'DisplayName', 'Centre of figure GroundTruth')
plot(dxCircle_GT, dyCircle_GT, 'b-', 'DisplayName', 'GroundTruth conic');

% Add text to display range prediction
textPosition1 = [10, 10]; % Position for text on the image
textString1 = sprintf(['Predictions:\n' ...
                       '  Centroid: [%.2f, %.2f]\n' ...
                       '  Apparent Radius: %.2f\n' ...
                       '  Range: %.2f'], ...
                      dCentroidPrediction(1), dCentroidPrediction(2), ...
                      dApparentRadiusPrediction, dRangePrediction);
text(textPosition1(1), textPosition1(2), textString1, 'Color', 'yellow', 'FontSize', 10, 'FontWeight', 'bold', 'VerticalAlignment', 'top');

% Add text to display labels from strImageLabels
textPosition2 = [10, 95]; % Position for the labels text below the range prediction
textString2 = sprintf(['Labels:\n' ...
    '  Centroid: [%.2f, %.2f]\n' ...
    '  Radius in Pixels: %.2f\n' ...
    '  Range in Radii: %.2f\n' ...
    '  Reference Radius: %.2f'], ...
    strImageLabels.dCentroid(1), strImageLabels.dCentroid(2), ...
    strImageLabels.dRadiusInPix, strImageLabels.dRangeInRadii, ...
    strImageLabels.dRefRadius);
text(textPosition2(1), textPosition2(2), textString2, 'Color', 'cyan', 'FontSize', 10, 'FontWeight', 'bold', 'VerticalAlignment', 'top');


legend('show'); % Show legend
hold off;

end









