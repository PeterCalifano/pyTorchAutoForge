import tensorrt as trt
import torch 
import numpy as np
import sys 
import pycuda as cuda


# TODO
class TRTengineExporter:
    def __init__(self) -> None:
        # Define logger and builder for TensorRT
        self.logger = trt.Logger(trt.Logger.INFO)
        self.builder = trt.Builder(self.logger)

        self.network # TODO builder.create_network( 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) ) 

        # For ONNX export
        self.onnx_parser #= trt.OnnxParser(self.network, self.logger)

    def build_engine(self, model, file_path: str):

        pass