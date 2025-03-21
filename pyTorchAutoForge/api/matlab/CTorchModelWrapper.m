classdef CTorchModelWrapper < handle
    %% DESCRIPTION
    % Class to enable evaluation of torch models in MATLAB directly using numpy as exchange library. This
    % class relies either on pyenv (interpreter) or on TCP connection to dedicated python tcp (tcp server in
    % pyTorchAutoForge.api module).
    % -------------------------------------------------------------------------------------------------------------
    %% CHANGELOG
    % 06-11-2024    Pietro Califano    First prototype implementation for pyenv 
    % 14-11-2024    Pietro Califano    Completed prototype of tcp interface initialization and class constructor
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
        enumTorchWrapperMode; % 'TCP', 'PYENV' (define enum)
        charDevice = 'cpu';
        objTensorCommHandler
        charPTAF_HOME = '/home/peterc/devDir/pyTorchAutoForge'
        bMultiTensorMode = true
    end

    methods (Access = public)
        %% CONSTRUCTOR
        function self = CTorchModelWrapper(charModelPath, charDevice, kwargs)
            arguments
                charModelPath (1,1) string 
                charDevice    (1,1) string = 'cpu'
            end
            arguments
                kwargs.enumTorchWrapperMode  (1,1) EnumTorchWrapperMode {isa(kwargs.enumTorchWrapperMode, 'EnumTorchWrapperMode')} = EnumTorchWrapperMode.TCP
                kwargs.charPythonEnvPath     (1,1) string = ''
                kwargs.charServerAddress     (1,1) string = '127.0.0.1' % Assumes localhost server
                kwargs.i32PortNumber         (1,1) int32 = 50005       % Assumes free port number
                kwargs.charInterfaceFcnsPath (1,1) string = '/home/peterc/devDir/MachineLearning_PeterCdev/matlab/LimbBasedNavigationAtMoon'
                kwargs.bMultiTensorMode (1,1) logical {islogical, isscalar} = true
            end
            
            % Assign properties
            self.charDevice = charDevice;
            self.enumTorchWrapperMode = kwargs.enumTorchWrapperMode;
            self.charModelPath = charModelPath;
            self.bMultiTensorMode = kwargs.bMultiTensorMode;

            if self.enumTorchWrapperMode == EnumTorchWrapperMode.PYENV

                % assert(kwargs.charPythonEnvPath ~= '', 'Selected PYENV API mode: kwargs.charPythonEnvPath cannot be empty!')
                self = self.init_pyenv(kwargs.charPythonEnvPath);

            elseif self.enumTorchWrapperMode == EnumTorchWrapperMode.TCP
                
                assert(isfolder(kwargs.charInterfaceFcnsPath), 'Non-existent kwargs.charInterfaceFcnsPath. You need to provide a valid location of functions to manage communication with AutoForge TCP server.')
                self = self.init_tcpInterface(kwargs.i32PortNumber, kwargs.charServerAddress);
    
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
            switch self.enumTorchWrapperMode
                case EnumTorchWrapperMode.TCP
                    
                    if isfloat(X)
                        X = single(X);
                    elseif iscell(X)
                        for idC = 1:length(X)
                            if isfloat(X{idC})
                                X{idC} = single(X{idC});
                            end
                        end
                    end

                    % Call TensorCommManager to forward data
                    dWrittenBytes = self.objTensorCommHandler.WriteBuffer(X);

                    % Read buffer from server with output
                    [Y, self.objTensorCommHandler] = self.objTensorCommHandler.ReadBuffer();


                case EnumTorchWrapperMode.PYENV
                    error('NOT IMPLEMENTED/WORKING YET')

                otherwise
                    error('Invalid Torch Wrapper Mode.')
            end

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

        
        function [self] = init_tcpInterface(self, i32PortNumber, charServerAddress, dCommTimeout)
            arguments
                self
                i32PortNumber      (1,1) int32 
                charServerAddress  (1,1) string = '127.0.0.1' % Localhost is default
                dCommTimeout       (1,1) double = 60
            end
            
            % Add path to interface functions
            charInterfacePath = fullfile(self.charPTAF_HOME, 'lib', 'CommManager4MATLAB', 'src');
            addpath(genpath(charInterfacePath));
            bIsModuleAvailable = not(isempty(which('CommManager.m')));

            assert(bIsModuleAvailable, 'ERROR: CommManager not found. Have you initialized the submodules?')

            % Create communication handler and initialize directly
            self.objTensorCommHandler = TensorCommManager(charServerAddress, ...
                                                        i32PortNumber, ...
                                                        dCommTimeout, ...
                                                        "bInitInPlace", true, ...
                                                        "bMULTI_TENSOR", self.bMultiTensorMode);

        end
    end

end
