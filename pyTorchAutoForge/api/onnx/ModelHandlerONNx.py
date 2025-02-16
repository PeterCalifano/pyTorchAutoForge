import numpy
import torch, onnx, os
from pyTorchAutoForge.modelBuilding.modelClasses import torchModel
from pyTorchAutoForge.utils.utils import AddZerosPadding



class ModelHandlerONNx:
    """
     _summary_

    _extended_summary_
    """
    # CONSTRUCTOR
    def __init__(self, model: torch.nn.Module | torchModel | onnx.ModelProto, dummy_input_sample: torch.Tensor | numpy.ndarray, onnx_export_path: str = '.', opset_version: int = 11) -> None:
        
        # Store shallow copy of model
        if isinstance(model, torch.nn.Module):
            self.torch_model = model
            self.onnx_model = None

        elif isinstance(model, onnx.ModelProto):
            self.torch_model = None
            self.onnx_model = model
        else:
            raise ValueError("Model must be of base type torch.nn.Module or onnx.ModelProto") 

        # Store export details
        self.dummy_input_sample = dummy_input_sample
        self.onnx_export_path = onnx_export_path
        self.opset_version = opset_version

        # Get version of modules installed in working environment
        self.torch_version = torch.__version__
        self.onnx_version = onnx.__version__

    # METHODS
    def torch_export(self) -> None:
        """Export the model to ONNx format."""

        pass

    def torch_dynamo_export(self) -> None:
        """Export the model to ONNx format using TorchDynamo."""

        pass

    def convert_to_onnx_opset(self, onnx_opset_version : int = 11) -> onnx.ModelProto:
        """Convert the model to a different ONNx version."""

        if onnx_opset_version is None:
            onnx_opset_version = self.opset_version

        try: 
            model_proto = onnx.version_converter.convert_version (model=self.model, target_version=onnx_opset_version)
            return model_proto
            
        except Exception as e:
            print(f"Error converting model to opset version {self.onnx_opset_version}: {e}")
            return None


    def save_onnx_proto(self, modelProto: onnx.ModelProto) -> None:
        """Method to save ONNx model proto to disk."""

        modelFilePath = os.path.join(self.onnx_export_path, self.model_filename + '.onnx')
        onnx.save_model(modelProto, modelFilePath.replace('.onnx', '_ver' + str(self.onnx_target_version) + '.onnx'))


    def onnx_load(self) -> onnx.ModelProto:
        """Method to load ONNx model from disk."""

        pass

    def onnx_load_to_torch(self) -> onnx.ModelProto:
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
    pathToModel = modelSaveName+'.onnx'
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

    return modelONNx, convertedModel

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
