classdef CTorchModelWrapper < handle
    %% DESCRIPTION
    % Class to enable evaluation of torch models in MATLAB directly using numpy as exchange library. This
    % class relies either on pyenv (interpreter) or on TCP connection to dedicated python tcp (tcp server in
    % pyTorchAutoForge.api module).
    % -------------------------------------------------------------------------------------------------------------
    %% CHANGELOG
    % 06-11-2024        Pietro Califano         First prototype implementation for pyenv 
    % -------------------------------------------------------------------------------------------------------------
    %% DEPENDENCIES
    % [-]
    % -------------------------------------------------------------------------------------------------------------
    %% Future upgrades
    % [-]
    % -------------------------------------------------------------------------------------------------------------

    properties (SetAccess = protected, GetAccess = public)
        pyenv_modules = dictionary;
        python_env;
        charModelPath;
        enumAPI_MODE; % 'TCP', 'PYENV' (define enum)
        charDevice = 'cpu';
        objCommHandler
    end

    methods (Access = public)
        %% CONSTRUCTOR
        function self = CTorchModelWrapper(charModelPath, charDevice, enumAPI_MODE, kwargs)
            arguments
                charModelPath (1,1) string 
                charDevice    (1,1) string = 'cpu'
                enumAPI_MODE  (1,1) EnumTorchWrapperMode {isa(enumAPI_MODE, 'EnumTorchWrapperMode')}= EnumTorchWrapperMode.TCP
            end
            arguments
                kwargs.charPythonEnvPath     (1,1) = ''
                kwargs.charServerAddress     (1,1) = '127.0.0.1' % Assumes localhost server
                kwargs.int32PortNumber       (1,1) = 50005       % Assumes free port number
                kwargs.charInterfaceFcnsPath (1,1) string = '/home/peterc/devDir/MachineLearning_PeterCdev/matlab/LimbBasedNavigationAtMoon'
            end
            
            % Assign properties
            self.charDevice = charDevice;
            self.enumAPI_MODE = enumAPI_MODE;
            self.charModelPath = charModelPath;

            if enumAPI_MODE == EnumTorchWrapperMode.PYENV

                % assert(kwargs.charPythonEnvPath ~= '', 'Selected PYENV API mode: kwargs.charPythonEnvPath cannot be empty!')
                self = init_pyenv(kwargs.charPythonEnvPath);

            elseif enumAPI_MODE == EnumTorchWrapperMode.TCP
                
                assert(isfolder(kwargs.charInterfaceFcnsPath), 'Non-existent kwargs.charInterfaceFcnsPath. You need to provide a valid location of functions to manage communication with AutoForge TCP server.')
                self = init_tcpInterface(kwargs.charServerAddress, kwargs.int16PortNumber, kwargs.charInterfaceFcnsPath);
    
            else
                error('Invalid API mode.')
            end

        end
        
        %% PUBLIC METHODS
        % Method to perform inference
        function Y = forward(self, X)
            arguments
                self
                X
            end

            % Call forward method of model depending on mode
        end
    end

    %% PROTECTED METHODS
    methods (Access = protected)

        function [self] = init_pyenv(self, charPythonEnvPath)
            arguments
                self
                charPythonEnvPath (1,1) string = fullfile('..', '..', '..', '.venvTorch', 'bin', 'python3.11');
            end

            % pyenv initialization to use interpreter
            % DEVNOTE: THIS REQUIRES INPUT PATH FROM USER!
            assert(isfile(charPythonEnvPath)) % Assert path existence

            if pyenv().Status == matlab.pyclient.Status.Terminated || pyenv().Status == matlab.pyclient.Status.NotLoaded
                self.python_env = pyenv(Version = charPythonEnvPath);
                pause(1);
                pyenv;

            elseif pyenv().Status == matlab.pyclient.Status.Loaded
                warning('Running python environment detected (Loaded state). Wrapper will use it.')
                self.python_env = pyenv(); % Assign existent
            end
        
            fprintf('\nUsing python environment:\n');
            disp(self.python_env);

            % Create modules objects
            self.pyenv_modules('np') = py.importlib.import_module('numpy');
            self.pyenv_modules('torch') = py.importlib.import_module('torch');
            self.pyenv_modules('autoforge_api_matlab') = py.importlib.import_module('pyTorchAutoForge.api.matlab');

            py.importlib.reload(self.pyenv_modules('autoforge_api_matlab'));

            % Loaded modules:
            keys = self.pyenv_modules.keys();
            fprintf('Loaded modules (aliases): \n')
            for key = keys
                fprintf("\t%s;", key)
            end
            fprintf("\n");
        end

        
        function [self] = init_tcpInterface(self, charServerAddress, int32PortNumber, charInterfaceFcnsPath, dCommTimeout)
            arguments
                self
                charServerAddress     (1,1) string 
                int32PortNumber       (1,1) int32 
                charInterfaceFcnsPath (1,1) string
                dCommTimeout          (1,1) double = 20
            end
            
            % Add path to interface functions
            addpath(genpath(charInterfaceFcnsPath));
            
            % Create communication handler and initialize directly
            self.objCommHandler = CommManager(charServerAddress, int32PortNumber, dCommTimeout, "bInitInPlace", true);

        end
    end

end
