classdef test_CTorchModelWrapper < matlab.unittest.TestCase
    properties
        pythonEnv
    end
    % Shared setup for the test environment of the class --> executed once BEFORE test cases
    methods (TestClassSetup)
        

    end

    % Shared cleanup for the test environment of the class --> executed once AFTER test cases
    methods (TestClassTeardown)


    end


    %% UNIT TEST SETUP
    methods (TestMethodSetup)
        % Setup for each test

        % function SetupTest_pyenv(testCase)
        % 
        %     if pyenv().Status == matlab.pyclient.Status.Terminated
        %         testCase.pythonEnv = pyenv(Version = fullfile('..', '..', '..', '.venvTorch', 'bin', 'python3.11', ...
        %             'ExecutionMode', 'OutOfProcess'));
        %         disp(testCase.pythonEnv);
        %     end
        % 
        %     % Load modules
        %     testCase.np = py.importlib.import_module('numpy');
        %     testCase.pyTorchAutoForge = py.importlib.import_module('pyTorchAutoForge');
        %     py.importlib.reload(pyTorchAutoForge);
        % 
        % end
        % 
        % function SetupTest_TCP(testCase)
        % % TODO
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
