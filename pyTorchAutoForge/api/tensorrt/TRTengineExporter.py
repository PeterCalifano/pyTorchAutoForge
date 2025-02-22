import tensorrt as trt
import torch 
import numpy as np
import sys, os, shutil
import pycuda as cuda
import subprocess
from enum import Enum
from pyTorchAutoForge.api.onnx import ModelHandlerONNx

class TRTprecision(Enum):
    FP32 = 0
    FP16 = 1
    INT16 = 2
    INT8 = 3

class TRTengineExporterMode(Enum):
    TRTEXEC = 0
    PYTHON = 1    

class TRTengineExporter:
    """
    TRTengineExporter is a helper class to export TensorRT engines from ONNX models.
    
    It supports two modes: TRTEXEC (using the trtexec command-line tool) and PYTHON (not yet implemented).
    The class also allows specifying the precision mode for the engine (FP32, FP16, INT16, INT8).
    """
    def __init__(self, exporter_mode: TRTengineExporterMode = TRTengineExporterMode.TRTEXEC, precision: TRTprecision = TRTprecision.FP32, input_onnx_model: str | None = None, output_engine_path: str | None = None, torch_model: torch.nn.Module | None = None, input_sample : torch.Tensor | None = None) -> None:

        # Save mode and precision values
        self.exporter_mode = exporter_mode
        self.precision = precision

        # Save input and output paths 
        self.input_onnx_model = input_onnx_model
        self.output_engine_path = output_engine_path      

        self.torch_model = torch_model

        if torch_model is not None:
            # Full export chain from torch model
            print('Chain of exports from torch model to TensorRT engine requested. Model will intermediately be converted to ONNX format using ModelHandlerONNx.')
            if input_onnx_model is None:
                input_onnx_model = "./tmp_onnx_model_artifact.onnx"

            if input_sample is None:
                # TODO Try to create a dummy input sample reading size from model
                raise ValueError("Please provide an input sample to export a model directly from torch.nn.Module.")

            self.build_intermediate_onnx_(torch_model, input_sample, input_onnx_model)
         
        if self.exporter_mode == TRTengineExporterMode.PYTHON:
            raise NotImplementedError("Engine building is not implemented for Python mode yet.") # TODO Implement, see InferenceTensorRT_example.py by Umberto
            # Define logger and builder for TensorRT
            self.logger = trt.Logger(trt.Logger.INFO)
            self.builder = trt.Builder(self.logger)

            self.network # TODO builder.create_network( 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) ) 

            # For ONNX export
            self.onnx_parser #= trt.OnnxParser(self.network, self.logger)

        elif self.exporter_mode == TRTengineExporterMode.TRTEXEC: 
            # Check if TRTEXEC executable is available
            if shutil.which("trtexec") is None:
                raise FileNotFoundError("trtexec executable not found in system PATH. Please install TensorRT library and add it to PATH.")
        else:
            raise ValueError("Invalid TRTengineExporterMode value. Please select TRTEXEC or PYTHON.")

    def build_intermediate_onnx_(self, torch_model: torch.nn.Module, input_sample: torch.Tensor, input_onnx_model: str) -> None:
        """
        Converts input PyTorch model to intermediate ONNX model before building TensorRT engine.

        :param torch_model: The PyTorch model to convert.
        :type torch_model: torch.nn.Module
        :param input_sample: A sample input tensor for the model.
        :type input_sample: torch.Tensor
        :param input_onnx_model: Path to save the ONNX model.
        :type input_onnx_model: str
        :raises FileNotFoundError: If the ONNX model was not exported successfully.
        """
                    
        ModelHandlerONNx(torch_model, dummy_input_sample=input_sample, onnx_export_path=input_onnx_model, run_export_validation=True).torch_export()
        
        # Check if model was exported successfully (file exists)
        if not os.path.exists(input_onnx_model):
            raise FileNotFoundError("ONNX model was not exported successfully or incorrect path was checked. Please check any error message.")
        else:
            print("Intermediate ONNX model exported successfully to:", input_onnx_model)

    def check_input_output_paths_(self, input_onnx_model: str | None, output_engine_path: str | None)  -> tuple[str, str]:
        """
        Internal helper method to check validity of input and output paths for engine building.

        :param input_onnx_model: Path to the input ONNX model
        :type input_onnx_model: str | None
        :param output_engine_path: Path to save the output engine
        :type output_engine_path: str | None
        :raises ValueError: If input or output paths are not provided
        :raises NotImplementedError: If an invalid code branch is reached
        """

        # Save input and output paths if not provided at helper class instantiation
        if input_onnx_model is None and self.input_onnx_model is None:
            raise ValueError("Please provide an input ONNX model path as input or at helper class instantiation.")
        elif input_onnx_model is not None and self.input_onnx_model is None:
            self.input_onnx_model = input_onnx_model

        elif input_onnx_model is None and self.input_onnx_model is not None:
            input_onnx_model = self.input_onnx_model

        else: 
            raise NotImplementedError("Code branch is invalid or not implemented. Please report this issue to petercalifano.gs@gmail or create an issue on the GitHub repository.")

        if output_engine_path is None and self.output_engine_path is None:
            raise ValueError("Please provide an output engine path as input or at helper class instantiation.")
        elif output_engine_path is not None and self.output_engine_path is None:
            self.output_engine_path = output_engine_path

        elif output_engine_path is None and self.output_engine_path is not None:
            output_engine_path = self.output_engine_path
        else: 
            raise NotImplementedError("Code branch is invalid or not implemented. Please report this issue to petercalifano.gs@gmail or create an issue on the GitHub repository.")
        
        return input_onnx_model, output_engine_path

    def build_engine_from_onnx(self, input_onnx_model: str | None = None, output_engine_path: str | None = None):
        """
        Builds a TensorRT engine from an ONNX model.

        :param input_onnx_model: Path to the input ONNX model, defaults to None.
        :type input_onnx_model: str | None, optional
        :param output_engine_path: Path to save the output engine, defaults to None.
        :type output_engine_path: str | None, optional
        :raises NotImplementedError: If the Python mode is selected.
        :raises RuntimeError: If the engine building process fails.
        :raises ValueError: If invalid paths or modes are provided.
        """

        input_onnx_model, output_engine_path = self.check_input_output_paths_(input_onnx_model, output_engine_path)

        if self.exporter_mode == TRTengineExporterMode.PYTHON:
            raise NotImplementedError("Engine building is not implemented for Python mode yet.")
        elif self.exporter_mode == TRTengineExporterMode.TRTEXEC: 

            # Call trtexec to build engine in a subprocess
            command = f"trtexec --onnx={input_onnx_model} --saveEngine={output_engine_path}"  # TODO add precision flags
            
            if self.precision == TRTprecision.FP16:
                command += ' --fp16'
            elif self.precision == TRTprecision.INT8:
                command += ' --int8'
            elif self.precision == TRTprecision.INT16:
                command += ' --int16'
            elif self.precision == TRTprecision.FP32:
                command += '' # No flag is FP32 (default)

            # Run command and get output to stdout through subprocess pipe
            result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            if result.returncode != 0:
                print("trtexec output:", result.stdout.decode())
                print("trtexec error:", result.stderr.decode())
                raise RuntimeError("Failed to build engine, trtexec returned a non-zero exit code.")
            else:
                print("trtexec output:", result.stdout.decode())
            
            print("Engine built successfully and saved to:", output_engine_path)

        else:
            raise ValueError("Invalid TRTengineExporterMode value. Please select TRTEXEC or PYTHON.")