"""TODO"""

# Python imports
#import sys, os

# Append paths of custom modules
#sys.path.append(os.path.join('/home/peterc/devDir/MachineLearning_PeterCdev/tcpServerPy'))
#sys.path.append(os.path.join('/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch'))
#sys.path.append(os.path.join('/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch/LimbBasedNavigationAtMoon'))

import numpy as np
import threading, os, sys, optuna
# Custom imports
from pyTorchAutoForge.api.tcp import tcpServerPy
from pyTorchAutoForge.api.tcp.tcpServerPy import DataProcessor, pytcp_server, pytcp_requestHandler, ProcessingMode
import torch
from torch import nn
from pyTorchAutoForge.modelBuilding.modelClasses import ReloadModelFromOptuna
from functools import partial
import cv2 as ocv

# Define processing function for model evaluation (OPNAV limb based)

def defineModelForEval():
    # NOTE: before using this function make sure the paths are correct
    hostname = os.uname().nodename
    trial_ID = None

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
    
    return model
         
def test_TorchWrapperComm_OPNAVlimbBased():
    HOST, PORT = "localhost", 50001
    PORT_MSGPACK = 50002

    model = defineModelForEval() # From optuna dbase (hardcoded for testing purposes or quick-n-dirty use)

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

    def forward_wrapper(inputData, model, processingMode: ProcessingMode):
        
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
        ocv.imshow('Input image', input_image_toshow)
        ocv.waitKey(1000)
        ocv.destroyAllWindows()

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
        
    predictCentroidRange = partial(forward_wrapper, model=model, processingMode=ProcessingMode.MULTI_TENSOR)

    predictCentroidRange_msgpack = partial(
        forward_wrapper, model=model, processingMode=ProcessingMode.MSG_PACK)

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
    

def test_TorchWrapperComm_FeatureMatching():
    raise NotImplementedError("Feature matching test not implemented yet.")

# MAIN SCRIPT
def main():
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
    test_TorchWrapperComm_OPNAVlimbBased()
    #test_TorchWrapperComm_FeatureMatching()
    #main()


