classdef testTorchModelMATLABwrapper < matlab.unittest.TestCase
    % TBD: how to use this class?
    
    % Shared setup for the test environment of the class --> executed once BEFORE test cases
    methods (TestClassSetup)
        function self = SetupTestEnv(self)
            pythonObj = pyenv(Version = fullfile('..', '..', '..', '.venvTorch', 'bin', 'python3'));
            print(pythonObj);
            
            % Try importing required modules
            np = py.importlib.import_module('numpy');
            pyTorchAutoForge = py.importlib.import_module('pyTorchAutoForge');
            py.importlib.reload(pyTorchAutoForge);

        end 
    end

    % Shared cleanup for the test environment of the class --> executed once AFTER test cases
    methods (TestClassTeardown)
        % TODO
    end


    %% UNIT TEST SETUP
    methods (TestMethodSetup)
        % Setup for each test

    end
    

    %% UNIT TEST CODE
    methods (Test)
        function TestEnvironment(testCase)
            % Assert library correct loading
        end

        % Test methods
        function TestInstantiation(testCase)  
            % Test instantiation of MATLAB wrapper objects (default)

            % Select model filename to load
            % modelPath = "";
            % modelFilename = "model.pt";

            % Instantiate MATLAB torch model wrapper
            % model = py.pyTorchAutoForge.api.matlab.TorchModelMATLABwrapper(modelPath, modelFilename, self.DEBUG_MODE);
            
            
            % testCase.verifyFail("");
        end
    end
    
end
