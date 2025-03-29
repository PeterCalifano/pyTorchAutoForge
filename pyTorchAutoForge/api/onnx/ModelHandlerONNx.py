import numpy
import torch, onnx, os
from pyTorchAutoForge.model_building.modelClasses import torchModel
from pyTorchAutoForge.utils import AddZerosPadding, torch_to_numpy, timeit_averaged_
from onnxruntime import InferenceSession
from numpy.testing import assert_allclose

class ModelHandlerONNx:
    """
     _summary_
     TODO

    _extended_summary_
    """
    # CONSTRUCTOR

    def __init__(self, model: torch.nn.Module | torchModel | onnx.ModelProto, dummy_input_sample: torch.Tensor | numpy.ndarray, onnx_export_path: str = '.', opset_version: int = 13, run_export_validation: bool = True, generate_report: bool = False) -> None:
        
        # Store shallow copy of model
        if isinstance(model, torch.nn.Module):
            self.torch_model: torch.nn.Module = model
            #self.onnx_model 

        elif isinstance(model, onnx.ModelProto):
            #self.torch_model
            self.onnx_model = model
        else:
            raise ValueError("Model must be of base type torch.nn.Module or onnx.ModelProto") 

        # Store export details
        self.run_export_validation = run_export_validation
        self.onnx_filepath = ""
        self.dummy_input_sample = dummy_input_sample
        self.onnx_export_path = onnx_export_path
        self.opset_version = opset_version
        self.IO_names = {'input': ['input'], 'output': ['output']}
        self.dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        self.generate_report = generate_report

        # Get version of modules installed in working environment
        self.torch_version = torch.__version__

    # METHODS
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def torch_export(self, input_tensor: torch.Tensor | None = None, onnx_model_name: str | None = None, dynamic_axes: dict = None, IO_names: dict = None, verbose: bool = True) -> None:
        """Export the model to ONNx format using TorchScript backend."""

        # TODO (PC) move all this preliminary code to a dedicated method
        if onnx_model_name is None and self.onnx_export_path is not None:
            onnx_model_name = os.path.basename(self.onnx_export_path)

            if onnx_model_name == "":
                onnx_model_name = 'onnx_export'

        elif onnx_model_name is None and self.onnx_export_path is None:
            print('No name provided for the ONNx model. Assign default value.')
            onnx_model_name = 'onnx_export'
        
        if not os.path.exists(self.onnx_export_path):
            os.makedirs(self.onnx_export_path)

        # Check if any model is already exported in the export path and append ID to the filename if any
        nameID = 0
        onnx_model_name_tmp = onnx_model_name
        while os.path.isfile(os.path.join(self.onnx_export_path, onnx_model_name_tmp + ".onnx")):
            onnx_model_name_tmp = onnx_model_name + str(nameID)
            nameID += 1
        onnx_model_name = onnx_model_name_tmp

        # Assign input tensor from init if not provided
        if input_tensor is None and self.dummy_input_sample is not None:
            input_tensor = self.dummy_input_sample
        else:
            raise ValueError("Input tensor must be provided or dummy input sample must be provided when constructing this class.")

        if dynamic_axes is None:
            # Assume first dimension (batch size) is dynamic
            dynamic_axes = self.dynamic_axes

        if IO_names is None:
            IO_names = self.IO_names

        # Inputs description: 
        # 1) model being run
        # 2) model input (or a tuple for multiple inputs)
        # 3) where to save the model (can be a file or file-like object)
        # 4) Store the trained parameter weights inside the model file
        # 5) ONNX version to export the model to
        # 6) Whether to execute constant folding for optimization
        # 7) Model input name
        # 8) Model output name

        self.onnx_filepath = os.path.join(
            self.onnx_export_path, onnx_model_name + ".onnx")

        torch.onnx.export(self.torch_model,               
                        input_tensor,                      
                        self.onnx_filepath,
                        export_params=True,                         
                        opset_version=self.opset_version,           
                        do_constant_folding=True,                   
                        input_names=IO_names['input'],              
                        output_names=IO_names['output'],            
                        dynamic_axes=dynamic_axes,
                        verbose=verbose, report=self.generate_report)

        print(f"Model exported to ONNx format: {os.path.join(self.onnx_export_path, onnx_model_name + '.onnx')}")

        if self.run_export_validation:
            # Reload the model from disk
            self.onnx_model = self.onnx_load(os.path.join(
                self.onnx_export_path, onnx_model_name + ".onnx"))
            self.onnx_validate(self.onnx_model)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def torch_dynamo_export(self, input_tensor: torch.Tensor | None = None, onnx_model_name: str = 'onnx_dynamo_export', dynamic_axes: dict = None, IO_names: dict = None, verbose: bool = True) -> None:
        """Export the model to ONNx format using TorchDynamo."""

        # Check if any model is already exported in the export path and append ID to the filename if any
        nameID = 0
        onnx_model_name_tmp = onnx_model_name
        while os.path.isfile(os.path.join(self.onnx_export_path, onnx_model_name_tmp + ".onnx")):
            onnx_model_name_tmp = onnx_model_name + str(nameID)
            nameID += 1
        onnx_model_name = onnx_model_name_tmp

        # Assign input tensor from init if not provided
        if input_tensor is None and self.dummy_input_sample is not None:
            input_tensor = self.dummy_input_sample
        else:
            raise ValueError("Input tensor must be provided or dummy input sample must be provided when constructing this class.")

        if dynamic_axes is None:
            # Assume first dimension (batch size) is dynamic
            dynamic_axes = self.dynamic_axes

        if IO_names is None:
            IO_names = self.IO_names

        # Inputs description:
        # 1) model being run
        # 2) model input (or a tuple for multiple inputs)
        # 3) where to save the model (can be a file or file-like object)
        # 4) Store the trained parameter weights inside the model file
        # 5) ONNX version to export the model to
        # 6) Whether to execute constant folding for optimization
        # 7) Model input name
        # 8) Model output name
        
        onnx_program = torch.onnx.export(self.torch_model, input_tensor,
                          export_params=True,
                          opset_version=self.opset_version,
                          do_constant_folding=True,
                          input_names=['input'],
                          output_names=['output'],
                          dynamic_axes=self.dynamic_axes, 
                          dynamo=True, report=self.generate_report)

        # Call model optimization
        onnx_program.optimize()

        # Save optimized model (serialized ONNx model)
        onnx_file_path = os.path.join(self.onnx_export_path, onnx_model_name + ".onnx")
        onnx_program.save(onnx_file_path)

        print(f"Model exported to ONNx format using TorchDynamo: {os.path.join(self.onnx_export_path, onnx_model_name + '.onnx')}")

        if self.run_export_validation:
            # Reload the model from disk
            self.onnx_model = self.onnx_load(os.path.join(
                self.onnx_export_path, onnx_model_name + ".onnx"))
            self.onnx_validate(self.onnx_model)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def convert_to_onnx_opset(self, onnx_model : onnx.ModelProto = None, onnx_opset_version : int = None) -> onnx.ModelProto:
        """Convert the model to a different ONNx operation set version."""
        
        # Handle default values
        if onnx_opset_version is None:
            onnx_opset_version = self.opset_version

        if onnx_model is None and self.onnx_model is None:
            raise ValueError(
                "No ONNx model provided for conversion and no model stored in onnx_model attribute.")
        elif onnx_model is None:    
            onnx_model = self.onnx_model
    
        try: 
            model_proto = onnx.version_converter.convert_version(model=onnx_model, target_version=onnx_opset_version)
            return model_proto
            
        except Exception as e:
            print(f"Error converting model to opset version {self.onnx_opset_version}: {e}")
            return None

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def onnx_validate(self, onnx_model: onnx.ModelProto, test_sample : torch.Tensor | numpy.ndarray = None, output_sample : torch.Tensor | numpy.ndarray = None) -> None:
        """Validate the ONNx model using onnx.checker.check_model."""

        print('Validating model using checker.check_model...')
        onnx.checker.check_model(onnx_model, full_check=True)

        if test_sample is not None:
            print('Validating model using onnxruntime...')
            ort_session = InferenceSession(
                onnx_model, providers=["CPUExecutionProvider"])

            # Compute ONNX Runtime output prediction
            ort_inputs = {ort_session.get_inputs()[0].name: torch_to_numpy(tensor=test_sample)} # Assumes input is only one tensor
            ort_outs = ort_session.run(None, ort_inputs)

            if output_sample is not None:
                # Compare ONNX Runtime and PyTorch results
                assert_allclose(torch_to_numpy(output_sample), ort_outs[0], rtol=1e-03, atol=1e-06)

                print('Output equivalence test passed successfully with tolerances rtol=1e-03 and atol=1e-06.')


    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def onnx_compare_timing(self, torch_model : torch.nn.Module, onnx_model: onnx.ModelProto, test_sample : torch.Tensor | numpy.ndarray, num_iterations : int = 100) -> dict:
        
        # Move model to cpu for comparison
        torch_model.to('cpu')

        # Prepare onnxruntime session
        ort_session = InferenceSession(onnx_model, providers=["CPUExecutionProvider"])

        # Construct input dictionary for onnxruntime
        ort_inputs = {ort_session.get_inputs()[0].name: torch_to_numpy(tensor=test_sample)}
        # Get function pointer
        ort_session_run = ort_session.run        

        # Get averaged runtimes and return
        return { 'avg_time_torch': timeit_averaged_(torch_model, num_iterations, test_sample),
            'avg_time_onnx': timeit_averaged_(ort_session_run, num_iterations, None, ort_inputs)
        }

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # def save_onnx_proto(self, modelProto: onnx.ModelProto) -> None:
    #    """Method to save ONNx model proto to disk."""
    #    modelFilePath = os.path.join(self.onnx_export_path, self.model_filename + '.onnx')
    #    onnx.save_model(modelProto, modelFilePath.replace('.onnx', '_ver' + str(self#.onnx_target_version) + '.onnx'))

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def onnx_load(self, onnx_filepath: str = "") -> onnx.ModelProto:
        """Method to load ONNx model from disk."""

        if onnx_filepath == "": 
            onnx_filepath = self.onnx_filepath

        self.onnx_model = onnx.load(onnx_filepath)

        return self.onnx_model

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def onnx_load_to_torch(self) -> torch.nn.Module:
        """Method to load ONNx model from disk."""

        pass


################################## LEGACY CODE ##################################
# %% Torch to/from ONNx format exporter/loader based on TorchDynamo (PyTorch >2.0) - 09-06-2024
def ExportTorchModelToONNx(model: torch.nn.Module, dummyInputSample: torch.tensor, onnxExportPath: str = '.', 
                           onnxSaveName: str = 'trainedModelONNx', modelID: int = 0, onnx_version=None):

    # Define filename of the exported model
    if modelID > 999:
        stringLength = modelID
    else:
        stringLength = 3

    modelSaveName = os.path.join(
        onnxExportPath, onnxSaveName + AddZerosPadding(modelID, stringLength))

    # Export model to ONNx object
    # NOTE: ONNx model is stored as a binary protobuf file!
    modelONNx = torch.onnx.dynamo_export(model, dummyInputSample)
    # modelONNx = torch.onnx.export(model, dummyInputSample) # NOTE: ONNx model is stored as a binary protobuf file!

    # Save ONNx model
    pathToModel = modelSaveName + '.onnx'
    modelONNx.save(destination=pathToModel)  # NOTE: this is a torch utility, not onnx!

    # Try to convert model to required version
    if (onnx_version is not None) and type(onnx_version) is int:
        convertedModel = None
        print('Attempting conversion of ONNx model to version:', onnx_version)
        try:
            print(f"Model before conversion:\n{modelONNx}")
            # Reload onnx object using onnx module
            tmpModel = onnx.load(pathToModel)
            # Convert model to get new model proto
            convertedModelProto = onnx.version_converter.convert_version(
                tmpModel, onnx_version)

            # TEST
            # convertedModelProto.ir_version = 7

            # Save model proto to .onnbx
            onnx.save_model(convertedModelProto, modelSaveName +
                            '_ver' + str(onnx_version) + '.onnx')

        except Exception as errorMsg:
            print('Conversion failed due to error:', errorMsg)
    else:
        convertedModel = None

    return modelONNx, pathToModel, convertedModel

def LoadTorchModelFromONNx(dummyInputSample: torch.tensor, onnxExportPath: str = '.', onnxSaveName: str = 'trainedModelONNx', modelID: int = 0):
    
    # Define filename of the exported model
    if modelID > 999:
        stringLength = modelID
    else:
        stringLength = 3

    modelSaveName = os.path.join(
        onnxExportPath, onnxSaveName + '_', AddZerosPadding(modelID, stringLength))

    if os.path.isfile():
        modelONNx = onnx.load(modelSaveName)
        torchModel = None
        return torchModel, modelONNx
    else:
        raise ImportError('Specified input path to .onnx model not found.')
