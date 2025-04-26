function [importedModel] = ImportModelFromONNx(path2model, taskType)
arguments
    path2model (1,1) string
    taskType   (1,1) string
end

%% PROTOTYPE
% Function loading Neural networks models stored as ONNx format
% -------------------------------------------------------------------------------------------------------------
%% DESCRIPTION
% What the function does
% -------------------------------------------------------------------------------------------------------------
%% INPUT
% path2model (1,1) string
% taskType   (1,1) string
% -------------------------------------------------------------------------------------------------------------
%% OUTPUT
% importedModel
% -------------------------------------------------------------------------------------------------------------
%% CHANGELOG
% 09-06-2024        Pietro Califano         First prototype of ONNx model importer
% -------------------------------------------------------------------------------------------------------------
%% DEPENDENCIES
% [-]
% -------------------------------------------------------------------------------------------------------------
%% Future upgrades
% [-]
% -------------------------------------------------------------------------------------------------------------
%% Function code

% Perform checks on dependencies and inputs
if not(isfile(path2model))
    error('Specified .onnx file has not been found. Check input path.')
end

taskType = lower(taskType);

if strcmpi(taskType, 'classification') && strcmpi(taskType, 'regression')
    error('Input taskType is not one of the valid entries: classification, regression')
end

% Import network
disp('Attempting to import model from ONNx file...')
importedModel = importONNXNetwork(path2model, 'OutputLayerType', taskType);
disp('Model imported successfully.')

