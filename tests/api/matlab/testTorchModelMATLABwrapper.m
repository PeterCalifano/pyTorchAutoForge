classdef testTorchModelMATLABwrapper < matlab.unittest.TestCase
    % TBD: how to use this class?
    
    methods (TestClassSetup)
        function self = SetupTestSet(self)
            % Shared setup for the entire test class
            pythonObj = pyenv(Version='/home/peterc/devDir/pyTorchAutoForge/.venvTorch/bin/python3.11');
            print(pythonObj);
            np = py.importlib.import_module('numpy');
            pyTorchAutoForge = py.importlib.import_module('pyTorchAutoForge');
            py.importlib.reload(pyTorchAutoForge);

            self.DEBUG_MODE = true;
        end
        
    end
    
    methods (TestMethodSetup)
        % Setup for each test
    end
    
    methods (Test)
        % Test methods
        function TestInstantiation(testCase)
            
            % Select model filename to load
            modelPath = "";
            modelFilename = "model.pt";

            % Instantiate MATLAB torch model wrapper
            model = py.pyTorchAutoForge.api.matlab.TorchModelMATLABwrapper(modelPath, modelFilename, self.DEBUG_MODE);
            
            
            testCase.verifyFail("");
        end
    end
    
end
