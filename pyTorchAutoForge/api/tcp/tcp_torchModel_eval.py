"""TODO"""

# Python imports
#import sys, os

# Append paths of custom modules
#sys.path.append(os.path.join('/home/peterc/devDir/MachineLearning_PeterCdev/tcpServerPy'))
#sys.path.append(os.path.join('/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch'))
#sys.path.append(os.path.join('/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch/LimbBasedNavigationAtMoon'))

from enum import Enum
from socket import AI_PASSIVE
import numpy as np
import threading, os, sys, optuna
# Custom imports
from pyTorchAutoForge.api.tcp import tcpServerPy
from pyTorchAutoForge.api.tcp.tcpServerPy import DataProcessor, pytcp_server, pytcp_requestHandler, ProcessingMode
import torch, kornia
from torch import nn
from functools import partial
import cv2 as ocv
from pyTorchAutoForge.utils import GetDevice
from pyTorchAutoForge.utils.DeviceManager import GetDeviceMulti 
from typing import Any

# Import model paths
REPO_XFEAT_PATH = '/home/peterc/devDir/ML-repos/accelerated_features_PeterCdev'
sys.path.append(REPO_XFEAT_PATH)

from modules.xfeat import XFeatLightGlueWrapper

REPO_SUPERGLUE_PATH = '/home/peterc/devDir/ML-repos/SuperGluePretrainedNetwork_PeterCdev'
sys.path.append(REPO_SUPERGLUE_PATH)

from match_pairs_custom import DefineSuperPointSuperGlueModel


# Define processing function for model evaluation (OPNAV limb based)

def defineModelForEval_OPNAVlimbBased() -> nn.Module:
    # NOTE: before using this function make sure the paths are correct
    hostname = os.uname().nodename
    trial_ID = None

    torch.set_grad_enabled(mode=False)

    if hostname == 'peterc-desktopMSI':
            OPTUNA_DB_PATH = '/media/peterc/6426d3ea-1f91-40b7-93ab-7f00d034e5cd/optuna_storage'
            studyName = 'fullDiskRangeConvNet_HyperParamsOptim_ModelAdapterLossStrategy_GaussNoiseBlurShift_V6_19'
            filepath = os.path.join(
                "/media/peterc/6426d3ea-1f91-40b7-93ab-7f00d034e5cd/optuna_storage", "optuna_trials_best_models")

    elif hostname == 'peterc-recoil':
        OPTUNA_DB_PATH = '/home/peterc/devDir/operative/operative-develop/optuna_storage'
        studyName = 'fullDiskRangeConvNet_HyperParamsOptim_ModelLossStrategy_IntensityGaussNoiseBlurShift_reducedV6_612450306419870030'
        filepath = os.path.join(
            OPTUNA_DB_PATH, "optuna_trials_best_models")
    else:
        raise ValueError("Hostname not recognized.")

    # Check if the database exists
    if not os.path.exists(os.path.join(OPTUNA_DB_PATH, studyName+'.db')):
        raise ValueError(f"Database {studyName}.db not found.")

    # Load the study from the database
    study = optuna.load_study(study_name=studyName,
                            storage='sqlite:///{studyName}.db'.format(studyName=os.path.join(OPTUNA_DB_PATH, studyName)))

    # Get the trial
    if trial_ID == None:
        evaluation_trial = study.best_trial
    else:
        evaluation_trial = study.trials[trial_ID]

    run_name = evaluation_trial.user_attrs['mlflow_name']

    files = os.listdir(filepath)
    # Find the file that starts with the run_name
    matching_file = next(
        (f for f in files if f.startswith(run_name)), None)

    if matching_file:
        print(f"Matching file: {matching_file}")
    else:
        raise ValueError("No matching file found.")

    model = ReloadModelFromOptuna(
        evaluation_trial, {}, matching_file.replace('.pth', ''), filepath)
    
    device = GetDevice()
    print('Loaded model on device: ', device)
    
    return model.to(device=device)

class EnumFeatureMatchingType(Enum):
    SUPERPOINT_SUPERGLUE = 'SuperPoint_SuperGlue'
    XFEAT_LIGHTGLUE = 'XFeat_LightGlue'


def defineModelEval_FeatureMatching(enumFeatureMatchingType: EnumFeatureMatchingType = EnumFeatureMatchingType.SUPERPOINT_SUPERGLUE, device: 'str' = GetDevice()) -> nn.Module:

    torch.set_grad_enabled(mode=False)

    if enumFeatureMatchingType == EnumFeatureMatchingType.SUPERPOINT_SUPERGLUE:
        # Define SuperPoint + SuperGlue model
        model = DefineSuperPointSuperGlueModel(device)

    elif enumFeatureMatchingType == EnumFeatureMatchingType.XFEAT_LIGHTGLUE:
        # Define XFeat + LightGlue model
        model = XFeatLightGlueWrapper(device)

    else:
        raise ValueError("Feature matching type not valid.")

    return model

def test_TorchWrapperComm_OPNAVlimbBased():
    HOST, PORT = "localhost", 50001
    PORT_MSGPACK = 50002

    model = defineModelForEval_OPNAVlimbBased() # From optuna dbase (hardcoded for testing purposes or quick-n-dirty use)

    '''
    import json
    # Test model using same image as in MATLAB script before running server
    strDataPath = os.path.join("..", "..", "data")
    ui8Image = ocv.imread(os.path.join(strDataPath, "moon_image_testing.png"))

    with open(os.path.join(strDataPath, "moon_labels_testing.json"), 'r') as f:
        strImageLabels = json.load(f)

    # Show image
    ocv.imshow('Input image', ui8Image)
    ocv.waitKey(1000)
    
    # Convert image to tensor
    input_image = torch.tensor(ui8Image, dtype=torch.float32)
    input_image = input_image.permute(2, 0, 1).unsqueeze(0)

    ocv.destroyAllWindows()
    '''

    def forward_wrapper_OPNAVlimbBased(inputData, model, processingMode: ProcessingMode):
        
        if processingMode == ProcessingMode.MULTI_TENSOR:
            # Check input data
            assert isinstance(inputData, list) and len(inputData) == 1

            # Convert input data to torch tensor
            input_image = torch.tensor(inputData[0], dtype=torch.float32)

        elif processingMode == ProcessingMode.MSG_PACK:
            # Check input data
            assert isinstance(inputData, dict) and 'data' in inputData

            # Convert input data to torch tensor
            input_image = torch.tensor(inputData['data'], dtype=torch.float32)

        else:
            raise ValueError("Processing mode not recognized.")

        input_image_ = input_image[0,:,:,:].clone().detach().cpu()
        input_image_toshow = np.array(input_image_.permute(1, 2, 0).numpy().astype('uint8'))

        # Show received image
        #ocv.imshow('Input image', input_image_toshow)
        #ocv.waitKey(1000)
        #ocv.destroyAllWindows()

        # Evaluate model on input data
        with torch.no_grad():

            # Normalize input image to [0, 1] range
            input_image = input_image / 255.0

            # Evaluate model
            model.eval()
            print('Input shape: ', input_image.shape)
            print('Input datatype: ', input_image.dtype)
            output = model(input_image)
            print('Model output:', output)

            # Return output
            return output.detach().cpu().numpy()
        
    predictCentroidRange = partial(forward_wrapper_OPNAVlimbBased, model=model, processingMode=ProcessingMode.MULTI_TENSOR)

    predictCentroidRange_msgpack = partial(
        forward_wrapper_OPNAVlimbBased, model=model, processingMode=ProcessingMode.MSG_PACK)

    # Create a DataProcessor instance
    processor_multitensor = DataProcessor(
        predictCentroidRange, inputTargetType=np.float32, BufferSizeInBytes=1024, ENDIANNESS='little', DYNAMIC_BUFFER_MODE=True, PRE_PROCESSING_MODE=ProcessingMode.MULTI_TENSOR)

    processor_msgpack = DataProcessor(
        predictCentroidRange_msgpack, inputTargetType=np.float32, BufferSizeInBytes=1024, ENDIANNESS='little', DYNAMIC_BUFFER_MODE=True, PRE_PROCESSING_MODE=ProcessingMode.MSG_PACK)

    # Create and start the server in a separate thread
    server = pytcp_server(
        (HOST, PORT), pytcp_requestHandler, processor_multitensor)
    
    server_msgpack = pytcp_server(
        (HOST, PORT_MSGPACK), pytcp_requestHandler, processor_msgpack)
    
    # Run the server in a separate thread
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True # Exit the server thread when the main thread terminates
    server_thread.start() # Start the server thread

    server_thread_msgpack = threading.Thread(
        target=server_msgpack.serve_forever)
    server_thread_msgpack.daemon = True
    server_thread_msgpack.start()

    # NOTE: client is MATLAB-side in this case! Server must be closed manually
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print('Servers are shutting down...')
        server.shutdown()
        server.server_close()
        server_msgpack.shutdown()
        server_msgpack.server_close()
    
def test_TorchWrapperComm_FeatureMatching() -> None:
    HOST, PORT = "localhost", 50003

    network_name = EnumFeatureMatchingType.XFEAT_LIGHTGLUE
    device = GetDeviceMulti()

    # Hardcoded for quick-n-dirty use
    model = defineModelEval_FeatureMatching(enumFeatureMatchingType=network_name, device=device)

    def forward_wrapper_FeatureMatching(inputData, model, processingMode: ProcessingMode) -> dict[Any, Any]:
        if processingMode == ProcessingMode.MULTI_TENSOR:
            
            # Get input images pair
            print('Input data type: ', type(inputData))
            assert isinstance(inputData, list)

            if isinstance(inputData, (list, tuple)):
                assert len(inputData) == 2
                # Convert input data to torch tensor and normalize to [0, 1] range
                
                input_image1 = torch.tensor(data=inputData[0], dtype=torch.float32) / 255.0
                input_image2 = torch.tensor(inputData[1], dtype=torch.float32) / 255.0

                #input_image1 = torch.tensor(inputData[0], dtype=torch.float32)
                #input_image2 = torch.tensor(inputData[1], dtype=torch.float32)
                
            else:
                raise ValueError("Input data type not valid. Must be a list or a tuple.")
            
        else:
            raise ValueError("Processing mode not supported.")

        with torch.no_grad():   
            print('Evaluating model on images of shapes:', input_image1.shape, input_image2.shape)

            # Move data to same device as model 
            input_image1 = input_image1.to(device=device)
            input_image2 = input_image2.to(device=device)

            # DEBUG: show images
            #input_image1_ = input_image1[0,:,:,:].clone().detach().cpu()
            #input_image2_ = input_image2[0,:,:,:].clone().detach().cpu()
            #input_image1_toshow = np.array(input_image1_.permute(1, 2, 0).numpy().astype('uint8'))
            #input_image2_toshow = np.array(input_image2_.permute(1, 2, 0).numpy().astype('uint8'))

            # Show received image
            #ocv.imshow('Input image 1', input_image1_toshow)
            #ocv.imshow('Input image 2', input_image2_toshow)
            #ocv.waitKey()
            #ocv.destroyAllWindows()
            
            ############################## DEBUG ATTEMPT ##############################
            # Normalize images to [0, 1] range
            #input_image1 = input_image1 / 255.0
            #input_image2 = input_image2 / 255.0
            ###########################################################################

            if isinstance(model, XFeatLightGlueWrapper):
                # Renormalize images to [0, 255] range
                input_image1 = input_image1 * 255.0
                input_image2 = input_image2 * 255.0
    
                # Permute tp (H, W, C) format
                input_image1 = input_image1.permute(0, 2, 3, 1)
                input_image2 = input_image2.permute(0, 2, 3, 1)
                
                # Convert to ndarrays
                input_image1 = input_image1[0].detach().cpu().numpy()
                input_image2 = input_image2[0].detach().cpu().numpy()
                
                # DEVNOTE


            # Evaluate model on input data and convert to ndarrays
            predictedMatchesDict = model({'image0': input_image1, 'image1': input_image2}) # FIXME Xfeat Light Glue is failing here

            # DEVNOTE temporary casting to float32 to ensure TensorCommManager casts ok

            if isinstance(model, XFeatLightGlueWrapper):
                # Define output dictionary
                for k, v in predictedMatchesDict.items():
                    if isinstance(v, np.ndarray):
                        predictedMatchesDict[k] = v.astype(np.float32)
                    elif isinstance(v, torch.Tensor):
                        predictedMatchesDict[k] = v.detach().cpu().numpy().astype(np.float32)
                    else:
                        raise ValueError("Output type is invalid.")
            else:
                # Output is tensor
                predictedMatchesDict = {k: v[0].detach().cpu().float().numpy()
                                        for k, v in predictedMatchesDict.items()}
            ###### DEBUGGING ######
            #predictedMatchesDict = {k: v[0].detach().cpu().numpy()
            #                        for k, v in predictedMatchesDict.items()}


            do_ransac_essential = False
            if do_ransac_essential:
                pass # TODO implement ransac step using essential matrix 
            
            print('Returning dictinary with keys:', predictedMatchesDict.keys())
            print('Shapes of values:', [v.shape for v in predictedMatchesDict.values()])

            # Return output dictionary ['keypoints0', 'scores0', 'descriptors0', 'keypoints1', 'scores1', 'descriptors1', 'matches0', 'matches1', 'matching_scores0', 'matching_scores1']
            return predictedMatchesDict
        
    # Define function for data processor
    featureMatcherForward = partial(
        forward_wrapper_FeatureMatching, model=model, processingMode=ProcessingMode.MULTI_TENSOR)

    # Create a DataProcessor instance
    processor_multitensor = DataProcessor(
        featureMatcherForward, inputTargetType=np.float32, BufferSizeInBytes=1024, ENDIANNESS='little', DYNAMIC_BUFFER_MODE=True, PRE_PROCESSING_MODE=ProcessingMode.MULTI_TENSOR)

    # Create and start the server in a separate thread
    server = pytcp_server(
        (HOST, PORT), pytcp_requestHandler, processor_multitensor)

    # Run the server in a separate thread
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True  # Exit the server thread when the main thread terminates
    server_thread.start()  # Start the server thread

    try:
        while True:
            pass
    except KeyboardInterrupt:
        print('Servers are shutting down...')
        server.shutdown()
        server.server_close()
        server_thread.join()


# MAIN SCRIPT (TODO: need to be adapted)
def main() -> None:
    print('\n\n----------------------------------- RUNNING: torchModelOverTCP.py -----------------------------------\n')
    print("MAIN script operations: initialize always-on server --> listen to data from client --> if OK, evaluate model --> if OK, return output to client\n")
    
    # %% TORCH MODEL LOADING
    # Model path
    tracedModelSavePath = '/home/peterc/devDir/MachineLearning_PeterCdev/checkpoints/'
    #tracedModelName = 'HorizonPixCorrector_CNNv2_' + customTorchTools.AddZerosPadding(modelID, 3) + '_cpu'

    #tracedModelName = 'HorizonPixCorrector_CNNv1max_largerCNN_run3_005_cpu' + '.pt'
    #tracedModelName = '/home/peterc/devDir/MachineLearning_PeterCdev/checkpoints/HorizonPixCorrector_CNNv1max_largerCNN_run6/HorizonPixCorrector_CNNv1max_largerCNN_run6_0088_cuda0.pt'


    # Parameters
    # ACHTUNG: check which model is being loaded!
    tracedModelName = tracedModelSavePath + ""

    # Load torch traced model from file
    torchWrapper = pyTorchAutoForge.api.matlab.TorchModelMATLABwrapper(tracedModelName)

    # %% TCP SERVER INITIALIZATION
    HOST, PORT = "127.0.0.1", 50000 # Define host and port (random is ok)

    # Define DataProcessor object for RequestHandler
    numOfBytes = 56*4 # Length of input * number of bytes in double --> not used if DYNAMIC_BUFFER_MODE is True # TODO: modify this
    dataProcessorObj = tcpServerPy.DataProcessor(torchWrapper.forward, np.float32, numOfBytes, 
                                                 ENDIANNESS='little', DYNAMIC_BUFFER_MODE=True, 
                                                 PRE_PROCESSING_MODE=tcpServerPy.ProcessingMode.TENSOR)

    # Initialize TCP server and keep it running
    with tcpServerPy.pytcp_server((HOST, PORT), tcpServerPy.pytcp_requestHandler, dataProcessorObj, bindAndActivate=True) as server:
        try:
            print('\nServer initialized correctly. Set in "serve_forever" mode.')
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nServer is gracefully shutting down =D.")
            server.shutdown()
            server.server_close()



if __name__ == "__main__":
    #test_TorchWrapperComm_OPNAVlimbBased()
    test_TorchWrapperComm_FeatureMatching()
    #main()


