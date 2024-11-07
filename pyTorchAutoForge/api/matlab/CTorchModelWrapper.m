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
        api_mode; % 'tcp', 'pyenv' (define enum)
        device = 'cpu';
    end

    methods (Access = public)
        % CONSTRUCTOR
        function self = CTorchModelWrapper(model_path, device, kwargs)
            arguments
                model_path (1,1) string 
                device (1,1) string = 'cpu'
            end

            arguments
                kwargs.charPythonEnvPath (1,1) = ''
            end

            [self] = init_pyenv(self);

        end


        function Y = forward(self, X)
            arguments
                self
                X
            end

            % Call forward method of model depending on mode
        end
    end


    methods (Access=protected)

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

            elseif pyenv().Status == matlab.pyclient.Status.Loaded
                warning('Running python environment detected (Loaded state). Wrapper will use it.')
                self.python_env = pyenv(); % Assign existent
            end
        
            fprintf('\nUsing python environment:\n');
            disp(self.python_env)

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


        function [self, flags] = init_tcpInterface(self)
            arguments
                self
            end
            error('Not implemented yet')

        end
    end

end
