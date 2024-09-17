import torch, onnx, os
from pyTorchAutoForge.utils.utils import AddZerosPadding

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
    modelONNx.save(pathToModel)  # NOTE: this is a torch utility, not onnx!

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
