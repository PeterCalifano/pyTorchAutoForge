classdef testTorchModelMATLABwrapper < matlab.unittest.TestCase
    % TBD: how to use this class?
    properties
        pythonEnv
    end
    % Shared setup for the test environment of the class --> executed once BEFORE test cases
    methods (TestClassSetup)
        function SetupTestEnv(testCase)
            testCase.pythonEnv = pyenv(Version = fullfile('..', '..', '..', '.venvTorch', 'bin', 'python3.11'));
            disp(testCase.pythonEnv);
            
            % Not clear how to make pythonObj available to the class?
        end 
    end

    % Shared cleanup for the test environment of the class --> executed once AFTER test cases
    methods (TestClassTeardown)
        % TODO
        function TeardownTestEnv(testCase)
            terminate(testCase.pythonEnv);
        end
    end


    %% UNIT TEST SETUP
    methods (TestMethodSetup)
        % Setup for each test
        % function Setup_TestInstantiation()
        % 
        %     % Try importing required modules
        %     np = py.importlib.import_module('numpy');
        %     pyTorchAutoForge = py.importlib.import_module('pyTorchAutoForge');
        %     py.importlib.reload(pyTorchAutoForge);
        % 
        % end
    end
    

    %% UNIT TEST CODE
    methods (Test)
        function TestEnvironment(testCase)
            % Assert pyenv is correctly loaded
            version = testCase.pythonEnv.Version;
            testCase.fatalAssertFalse(isempty(version));
        end

        % Test methods
        function TestInstantiation(testCase)  

            % testCase.fatalAssertNotEmpty(pyTorchAutoForge);
            % Test instantiation of MATLAB wrapper objects (default)

            % Select model filename to load
            modelPath = "../testData/";
            modelFilename = "testModel";
            modelToLoad = fullfile(modelPath, modelFilename);

            % testCase.assertEqual(isfile(modelToLoad + '.pt'), true);

            % Instantiate MATLAB torch model wrapper
            % model = py.pyTorchAutoForge.api.matlab.TorchModelMATLABwrapper(modelPath, modelFilename, self.DEBUG_MODE);
            
            % Add assert on model instance
        end
    end
    
end
