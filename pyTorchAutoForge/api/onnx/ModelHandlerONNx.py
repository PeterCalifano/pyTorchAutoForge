import numpy
import torch, onnx, os
from pyTorchAutoForge.model_building.modelBuildingBlocks import AutoForgeModule
from pyTorchAutoForge.utils import AddZerosPadding, torch_to_numpy, timeit_averaged_
from numpy.testing import assert_allclose
from pyTorchAutoForge.utils import numpy_to_torch

# TODO: add support to use onnx simplify 
#simplified_model, check_status = onnxsim.simplify(onnx_model)

class ModelHandlerONNx:
    """
     _summary_
     TODO

    _extended_summary_
    """
    # CONSTRUCTOR
    def __init__(self, 
                 model: torch.nn.Module | AutoForgeModule | onnx.ModelProto, 
                 dummy_input_sample: torch.Tensor | numpy.ndarray, 
                 onnx_export_path: str = '.', 
                 opset_version: int = 13, 
                 run_export_validation: bool = True,
                 generate_report: bool = False,
                 run_onnx_simplify: bool = False) -> None:

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
        self.run_onnx_simplify = run_onnx_simplify

        # Get version of modules installed in working environment
        self.torch_version = torch.__version__

    # METHODS
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def _make_model_filename_and_folder(self, onnx_model_name) -> str:
        """Method to generate a filename for the model based on its class name and version, and to prepare the export folder."""
        if onnx_model_name is None and self.onnx_export_path is not None:
            onnx_model_name = os.path.basename(self.onnx_export_path)

            if onnx_model_name == "":
                onnx_model_name = 'onnx_export'

        elif onnx_model_name is None and self.onnx_export_path is None:
            print('No name provided for the ONNx model. Assign default value.')
            onnx_model_name = 'onnx_export'
        
        os.makedirs(os.path.dirname(self.onnx_export_path), exist_ok=True)

        # Check if any model is already exported in the export path and append ID to the filename if any
        nameID = 0
        onnx_model_name_tmp = onnx_model_name + "_" + str(nameID)
        while os.path.isfile(os.path.join(os.path.dirname(self.onnx_export_path), onnx_model_name_tmp + ".onnx")):
            onnx_model_name_tmp = onnx_model_name + "_" + str(nameID)
            nameID += 1

        onnx_model_name = onnx_model_name_tmp
        self.onnx_filepath = os.path.join(
            self.onnx_export_path, onnx_model_name + ".onnx")
        
        return onnx_model_name

    def _run_onnx_simplify(self, onnx_model: onnx.ModelProto) -> onnx.ModelProto:
        
        from onnxsim import simplify
        model_simplified, check = simplify(onnx_model)

        # Reload the model from disk
        self.onnx_model = self.onnx_load(onnx_filepath=self.onnx_filepath)

        # Simplify the ONNX model
        model_simplified, check = simplify(onnx_model)

        # Save simplified model
        onnx.save(model_simplified, self.onnx_filepath)
        print(f"ONNX model simplified and saved: {self.onnx_filepath}")

        if not check:
            print('\033[38;5;208mWarning: ONNX model simplifier internal validation failed.\033[0m')
        
        return self.onnx_filepath, model_simplified

    def torch_export(self, 
                     input_tensor: torch.Tensor | None = None, 
                     onnx_model_name: str | None = None, 
                     dynamic_axes: dict | None = None, 
                     IO_names: dict | None = None,
                     enable_verbose: bool = False) -> str:
        """Export the model to ONNx format using TorchScript backend."""

        # Prepare export folder and compose name
        onnx_model_name = self._make_model_filename_and_folder(onnx_model_name=onnx_model_name)

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
        # 6) whether to execute constant folding for optimization
        # 7) Model input name
        # 8) Model output name

        torch.onnx.export(self.torch_model,               
                        input_tensor,                      
                        self.onnx_filepath,
                        export_params=True,                         
                        opset_version=self.opset_version,           
                        do_constant_folding=True,                   
                        input_names=IO_names['input'],              
                        output_names=IO_names['output'],            
                        dynamic_axes=dynamic_axes,
                        verbose=enable_verbose, report=self.generate_report)

        print(f"Model exported to ONNx format: {self.onnx_filepath}")

        if self.run_export_validation:
            # Reload the model from disk
            self.onnx_model = self.onnx_load(onnx_filepath=self.onnx_filepath)
            
            self.onnx_validate(self.onnx_model,
                               test_sample=numpy_to_torch(self.dummy_input_sample))

        if self.run_onnx_simplify:
            self.onnx_filepath, model_simplified = self._run_onnx_simplify(self.onnx_model)
            print(f"ONNX model simplified and saved: {self.onnx_filepath}")

        return self.onnx_filepath
    
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def torch_dynamo_export(self, 
                            input_tensor: torch.Tensor | None = None, 
                            onnx_model_name: str = 'onnx_dynamo_export', 
                            dynamic_axes: dict | None = None,
                            IO_names: dict | None = None,
                            enable_verbose: bool = False) -> None:
        """Export the model to ONNx format using TorchDynamo."""

        # Prepare export folder and compose name
        onnx_model_name = self._make_model_filename_and_folder(
            onnx_model_name=onnx_model_name)

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
        
        onnx_program = torch.onnx.export(self.torch_model, 
                                         input_tensor,
                                        export_params=True,
                                        opset_version=self.opset_version,
                                        do_constant_folding=True,
                                        input_names=['input'],
                                        output_names=['output'],
                                        dynamic_axes=self.dynamic_axes, 
                                        dynamo=True, report=self.generate_report,
                                         verbose=enable_verbose)

        # Call model optimization
        onnx_program.optimize()

        # Save optimized model (serialized ONNx model)
        onnx_program.save(self.onnx_filepath)
        print(f"Model exported to ONNx format using TorchDynamo: {self.onnx_filepath}")

        if self.run_export_validation:
            # Reload the model from disk
            self.onnx_model = self.onnx_load(onnx_filepath=self.onnx_filepath)

            self.onnx_validate(onnx_model=self.onnx_model,
                               test_sample=numpy_to_torch(self.dummy_input_sample))

        if self.run_onnx_simplify:
            # Reload the model from disk
            self.onnx_filepath, model_simplified = self._run_onnx_simplify(self.onnx_model)
            print(f"ONNX model simplified and saved: {self.onnx_filepath}")

        return self.onnx_filepath

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def convert_to_onnx_opset(self, 
                              onnx_model : onnx.ModelProto = None, 
                              onnx_opset_version : int = None) -> onnx.ModelProto:
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
    def onnx_validate(self, 
                      onnx_model: onnx.ModelProto | str, 
                      test_sample : torch.Tensor | numpy.ndarray = None, 
                      output_sample : torch.Tensor | numpy.ndarray = None) -> None:
        """Validate the ONNx model using onnx.checker.check_model."""

        # If onnx_model is a string, load the model from the file
        if isinstance(onnx_model, str):
            if not os.path.isfile(onnx_model):
                raise FileNotFoundError(f"ONNX model file not found: {onnx_model}")
            onnx_model = onnx.load(onnx_model)

        elif not isinstance(onnx_model, onnx.ModelProto):
            raise TypeError(f"Invalid ONNX model class: {type(onnx_model)}")

        print('\033[94mValidating model using ONNx checker.check_model... \033[0m', end=' ')
        onnx.checker.check_model(onnx_model, full_check=True)
        print('\033[92mPASSED.\033[0m')
        
        if test_sample is not None:
            print('\033[94mValidating model inference using onnxruntime...\033[0m', end=' ')
            from onnxruntime import InferenceSession

            ort_session = InferenceSession(onnx_model.SerializeToString(), providers=["CPUExecutionProvider"])

            # Compute ONNX Runtime output prediction
            ort_inputs = {ort_session.get_inputs()[0].name: torch_to_numpy(tensor=test_sample)} # Assumes input is only one tensor
            ort_outs = ort_session.run(None, ort_inputs)
            print('\033[92mPASSED.\033[0m')

            if output_sample is not None:
                # Compare ONNX Runtime and PyTorch results
                print('\033[94mOutput equivalence test. Using tolerances rtol=1e-03 and atol=1e-06...\033[0m', end=' ')
                assert_allclose(torch_to_numpy(output_sample), ort_outs[0], rtol=1e-03, atol=1e-06)
                print('\033[92mPASSED.\033[0m')

            else:
                print('\033[38;5;208mNo output sample provided for ONNX model validation. Result validation test skipped.\033[0m')
                # TODO (UM) add warning message here

            # TODO (UM) extend validation method (equivalence test)

        else:
            print('\033[38;5;208mNo test sample provided for ONNX model validation. Equivalence test vs torch model skipped.\033[0m')

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # TODO (PC) test method!
    def onnx_compare_timing(self, torch_model : torch.nn.Module, onnx_model: onnx.ModelProto, test_sample : torch.Tensor | numpy.ndarray, num_iterations : int = 100) -> dict:
        
        # Move model to cpu for comparison
        torch_model.to('cpu')

        # Prepare onnxruntime session
        from onnxruntime import InferenceSession
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
        """Method to load ONNx model from disk and convert to torch."""
        # TODO
        pass