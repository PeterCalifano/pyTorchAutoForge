# Script created by PeterC 04-06-2024 to train and validate CNN-NN network enhancing pixel extraction for Limb based Navigation
# Reference works:

# Import modules
import sys, os
# Append paths of custom modules
sys.path.append(os.path.join('/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch/LimbBasedNavigationAtMoon'))

import pyTorchAutoForge 
import limbPixelExtraction_CNN_NN, ModelClasses # Custom model classes
import datasetPreparation
from sklearn import preprocessing # Import scikit-learn for dataset preparation

import torch, mlflow, optuna
from torch import nn
from scipy.spatial.transform import Rotation

from torch.utils.data import Dataset
from torch.utils.data import DataLoader # Utils for dataset management, storing pairs of (sample, label)
from torchvision import datasets # Import vision default datasets from torchvision
from torchvision.transforms import ToTensor # Utils

import numpy as np

from torch.utils.tensorboard import SummaryWriter # Key class to use tensorboard with PyTorch. VSCode will automatically ask if you want to load tensorboard in the current session.
import torch.optim as optim

# EXECUTION MODE
USE_MULTIPROCESS = False
USE_NORMALIZED_IMG = True
USE_LR_SCHEDULING = True
USE_SWA = True
TRAIN_ALL = True
REGEN_DATASET = True
USE_TENSOR_LOSS_EVAL = True
MODEL_CLASS_ID = 7
MANUAL_RUN = True # Uses MODEL_CLASS_ID to run a specific model
LOSS_TYPE = 4 # 0: Conic + L2, # 1: Conic + L2 + Quadratic OutOfPatch, # 2: Normalized Conic + L2 + OutOfPatch,            
            # 3: Polar-n-direction distance + OutOfPatch, #4: MSE + OutOfPatch + ConicLoss
RUN_ID = 1
USE_BATCH_NORM = True
# TRAINING and VALIDATION datasets ID in folder (in this order)
#datasetID = [6, 7]
#datasetID = [12, 10] # Datasets with 25x25 patches
#datasetID = [9, 10]  # NOTE: only matters for regeneration of dataset
datasetID = [12, 13]  # NOTE: only matters for regeneration of dataset

def main(idRun:int, idModelClass:int, idLossType:int):

    # SETTINGS and PARAMETERS 
    batch_size = 256 # Defines batch size in dataset
    #outChannelsSizes = [16, 32, 75, 15] 
    
    if idModelClass < 2:
        outChannelsSizes = [256, 128, 256, 64] 

    if idModelClass == 2:
        outChannelsSizes = [32, 16, 16+8, 16, 8] 

    elif idModelClass == 3:
        outChannelsSizes = [256, 128, 64, 32]

    elif idModelClass == 4:
        outChannelsSizes = [2056, 1024, 512, 64]

    elif idModelClass == 5:
        outChannelsSizes = [1024, 1024, 1024, 512, 512, 256, 64, 32]

    elif idModelClass == 6:
        kernelSizes = [5, 3, 3]
        poolKernelSizes = [2, 2, 2]
        outChannelsSizes = [256, 256, 256, 1024, 1024, 128, 64]

    elif idModelClass == 7:

        kernelSizes = [5]
        poolKernelSizes = [2]
        #outChannelsSizes = [256, 256, 128, 910, 860, 500, 910, 950, 380, 620, 415, 810, 645, 550, 230, 710, 420, 515, 60] # CNNvX , 3 conv + 16 nn
        outChannelsSizes = [256, 256, 128, 500, 500, 250, 125, 60]  # CNNvX , 3 conv + 7 nn
        outChannelsSizes = [77, 256, 256, 125, 60]
    else:
        raise ValueError('Model class ID not found.')


    #kernelSizes = [3, 3]
    initialLearnRate = 1E-4
    momentumValue = 0.8


    #TODO: add log and printing of settings of optimizer for each epoch. Reduce the training loss value printings

    # Loss function parameters
    lossParams = {'ConicLossWeightCoeff': 0, 'RectExpWeightCoeff': 1}

    # DEVNOTE: this could go into yaml/dataclass 
    lossParams['paramsTrain'] = {'ConicLossWeightCoeff': 1000, 'RectExpWeightCoeff': 0}
    lossParams['paramsEval'] = {'ConicLossWeightCoeff': 0, 'RectExpWeightCoeff': 0} # Not currently used

    optimizerID = 1 # 0: SGD, 1: Adam
    #UseMaxPooling = True

    device = pyTorchAutoForge.GetDevice()

    # DEVNOTE: to add to config file: path definitions to save models and name
    exportTracedModel = True    
    tracedModelSavePath = 'tracedModelsArchive' 

    # Options to restart training from checkpoint
    if idModelClass == 0:
        #ID = 0
        #runID = "{num:04d}".format(num=ID)
        #modelSavePath = './checkpoints/HorizonPixCorrector_CNNv1_run3'
        modelSavePath = './checkpoints/HorizonPixCorrector_CNNv1max_largerCNN_run' 
        datasetSavePath = './datasets/HorizonPixCorrectorV1'
        #tensorboardLogDir = './tensorboardLogs/tensorboardLog_v1max_largerCNN_run' 
        #tensorBoardPortNum = 6008
        
        modelArchName = 'HorizonPixCorrector_CNNv1max_largerCNN_run' 
        # DEVNOTE: input size should be taken from the dataset object
        inputSize = 56 # TODO: update this according to new model 
        numOfEpochs = 15


    elif idModelClass == 1:
        #ID = 0
        #runID = "{num:04d}".format(num=ID)
        modelSavePath = './checkpoints/HorizonPixCorrector_CNNv2max_largerCNN_run'
        datasetSavePath = './datasets/HorizonPixCorrectorV2'
        #tensorboardLogDir = './tensorboardLogs/tensorboardLog_v2max_largerCNN_run' 
        #tensorBoardPortNum = 6009

        modelArchName = 'HorizonPixCorrector_CNNv2max_largerCNN_run' 
        inputSize = 57 # TODO: update this according to new model
        numOfEpochs = 15

    elif idModelClass == 2:
        #ID = 6
        #runID = "{num:04d}".format(num=ID) 
        modelSavePath = './checkpoints/HorizonPixCorrector_CNNv3max_largerCNNdeeperNN_run' 
        datasetSavePath = './datasets/HorizonPixCorrectorV3'
        #tensorboardLogDir = './tensorboardLogs/tensorboardLog_v3max_largerCNNdeeperNN_run'   
        #tensorBoardPortNum = 6010

        modelArchName = 'HorizonPixCorrector_CNNv3max_largerCNNdeeperNN_run'
        inputSize = 57 # TODO: update this according to new model
        numOfEpochs = 10

    elif idModelClass == 3:
        #ID = 0
        #runID = "{num:04d}".format(num=ID)
        modelSavePath = './checkpoints/HorizonPixCorrector_deepNNv4_run' 
        datasetSavePath = './datasets/HorizonPixCorrectorV4'
        #tensorboardLogDir = './tensorboardLogs/tensorboardLog_deepNNv4_run'   + runID
        #tensorBoardPortNum = 6011

        modelArchName = 'HorizonPixCorrector_deepNNv4_run' 
        inputSize = 58 # TODO: update this according to new model
        numOfEpochs = 5

    elif idModelClass == 4:

        #runID = "{num:04d}".format(num=idRun)
        modelSavePath = './checkpoints/HorizonExtractionEnhancer_ShortCNNv6maxDeeper'
        datasetSavePath = './datasets/HorizonExtractionEnhancer_ShortCNNv6maxDeeper'
        #tensorboardLogDir = './tensorboardLogs/tensorboardLog_ShortCNNv6maxDeeper_run'   + runID
        #tensorBoardPortNum = 6012

        modelArchName = 'HorizonExtractionEnhancer_ShortCNNv6maxDeeper' 
        inputSize = 58 # TODO: update this according to new model
        numOfEpochs = 30

        parametersConfig = {'kernelSizes': [5],
                    'useBatchNorm': USE_BATCH_NORM, 
                    'poolingKernelSize': 2,
                    'alphaDropCoeff': 0.1,
                    'alphaLeaky': 0.01,
                    'patchSize': 7,
                    'LinearInputSkipSize': 9}

    elif idModelClass == 5:
        #runID = "{num:04d}".format(num=idRun)
        modelArchName = 'HorizonExtractionEnhancer_deepNNv8'

        modelSavePath = './checkpoints/' + modelArchName
        datasetSavePath = './datasets/' + modelArchName
        #tensorboardLogDir = './tensorboardLogs/tensorboardLog_ShortCNNv6maxDeeper_run'   + runID
        #tensorBoardPortNum = 6012

        parametersConfig = {'useBatchNorm': True, 'alphaDropCoeff': 0.05, 'LinearInputSize': 58,
                            'outChannelsSizes': outChannelsSizes}

        inputSize = 58 # TODO: update this according to new model
        numOfEpochs = 15

    elif idModelClass == 6:
        #runID = "{num:04d}".format(num=idRun)
        modelArchName = 'HorizonExtractionEnhancer_CNNv7_randomCloudGuessInputPatch25'

        modelSavePath = './checkpoints/' + modelArchName
        datasetSavePath = './datasets/' + modelArchName
        #tensorboardLogDir = './tensorboardLogs/tensorboardLog_ShortCNNv6maxDeeper_run'   + runID
        #tensorBoardPortNum = 6012

        # DEVNOTE: this could be an example of a configuration dictionary for NN 
        # Note that the yaml could be parsed as dataclass instead of being used directly
        parametersConfig = {'useBatchNorm': True, 'alphaDropCoeff': 0, 
                            'LinearInputSkipSize': 9,'outChannelsSizes': outChannelsSizes,
                            'kernelSizes': kernelSizes, 'poolkernelSizes': poolKernelSizes,
                            'patchSize': 25}

        #inputSize = 58
        inputSize = parametersConfig.get('patchSize')**2 + 9
        numOfEpochs = 20

    elif idModelClass == 7:
                #runID = "{num:04d}".format(num=idRun)
        #modelArchName = 'HorizonExtractionEnhancer_CNNvX_randomCloudGuessPatch25'
        modelArchName = 'HorizonExtractionEnhancer_CNNvX_randomCloudGuessPatch7'

        modelSavePath = './checkpoints/' + modelArchName
        datasetSavePath = './datasets/' + modelArchName
        #tensorboardLogDir = './tensorboardLogs/tensorboardLog_ShortCNNv6maxDeeper_run'   + runID
        #tensorBoardPortNum = 6012

        parametersConfig = {'useBatchNorm': True, 'alphaDropCoeff': 0, 
                            'LinearInputSkipSize': 9,'outChannelsSizes': outChannelsSizes,
                            'kernelSizes': kernelSizes, 'poolkernelSizes': poolKernelSizes,
                            'patchSize': 7}
        
        #inputSize = 58
        inputSize = parametersConfig.get('patchSize')**2 + 9
        numOfEpochs = 100


    EPOCH_START = 0
    
    options = {'taskType': 'regression', 
               'device': device, 
               'epochs': numOfEpochs, 
               'saveCheckpoints':True,
               'checkpointsOutDir': modelSavePath,
               'modelName': modelArchName,
               'loadCheckpoint': False,
               'checkpointsInDir': modelSavePath,
               'lossLogName': 'MSE_OutOfPatch',
               #'logDirectory': tensorboardLogDir,
               'epochStart': EPOCH_START,
               #'tensorBoardPortNum': tensorBoardPortNum
               }

    # DEVNOTE: restart training should become easier. Carry information of epoch with the model?
    if options['epochStart'] == 0:
        restartTraining = False
    else:
        restartTraining = True
        # Get last saving of model (NOTE: getmtime does not work properly. Use scandir + list comprehension)
        with os.scandir(modelSavePath) as it:
            modelNamesWithTime = [(entry.name, entry.stat().st_mtime) for entry in it if entry.is_file()]
        modelName = sorted(modelNamesWithTime, key=lambda x: x[1])[-1][0]


    # %% GENERATE OR LOAD TORCH DATASET

    dataPath = os.path.join('/home/peterc/devDir/MATLABcodes/syntheticRenderings/Datapairs')
    dataPath = os.path.join('/home/peterc/devDir/lumio/lumio-prototype/src/dataset_gen/Datapairs')
    #dirNamesRoot = os.listdir(dataPath)

    with os.scandir(dataPath) as it:
        modelNamesWithID = [(entry.name, entry.name[3:6]) for entry in it if entry.is_dir()]
        dirNamesRootTuples = sorted(modelNamesWithID, key=lambda x: int(x[1]))

    dirNamesRoot = [stringVal for stringVal, _ in dirNamesRootTuples]

    assert(len(dirNamesRoot) >= 2)

    # Select one of the available datapairs folders (each corresponding to a labels generation pipeline output)


    assert(len(datasetID) == 2)

    TrainingDatasetTag = dirNamesRoot[datasetID[0]]
    ValidationDatasetTag = dirNamesRoot[datasetID[1]]

    datasetNotFound = not(os.path.isfile(os.path.join(datasetSavePath, 'TrainingDataset_'+TrainingDatasetTag+'.pt'))) or not((os.path.isfile(os.path.join(datasetSavePath, 'ValidationDataset_'+ValidationDatasetTag+'.pt'))))

    if REGEN_DATASET == True or datasetNotFound:
        # REGENERATE the dataset
        # TODO: add printing of loaded dataset information

        for idDataset in range(len(datasetID)):

            if idDataset == 0:
                print('Generating training dataset from: ', dirNamesRoot[datasetID[idDataset]])
            elif idDataset == 1:
                print('Generating validation dataset from: ', dirNamesRoot[datasetID[idDataset]])

            # Get images and labels from IDth dataset
            dataDirPath = os.path.join(dataPath, dirNamesRoot[datasetID[idDataset]])
            dataFilenames = os.listdir(dataDirPath)
            nImages = len(dataFilenames)

            print('Found number of images:', nImages)

            # DEBUG
            #print(dataFilenames)

            # Get nPatches from the first datapairs files
            dataFileID = 0
            dataFilePath = os.path.join(dataDirPath, dataFilenames[dataFileID])
            #print('Loading data from:', dataFilePath)
            tmpdataDict, tmpdataKeys = datasetPreparation.LoadJSONdata(dataFilePath)
            print('All data loaded correctly --> DATASET PREPARATION BEGIN')

            # DEBUG
            #print(tmpdataKeys)

            nPatches = len(tmpdataDict['ui16coarseLimbPixels'][0])
            nSamples = nPatches * nImages

            # NOTE: each datapair corresponds to an image (i.e. nPatches samples)
            # Initialize dataset building variables
            saveID = 0

            inputDataArray  = np.zeros((inputSize, nSamples), dtype=np.float32)
            labelsDataArray = np.zeros((14, nSamples), dtype=np.float32)

            imgID = 0
            for dataPair in dataFilenames:
       
                # Data dict for ith image
                tmpdataDict, tmpdataKeys = datasetPreparation.LoadJSONdata(os.path.join(dataDirPath, dataPair))

                metadataDict = tmpdataDict['metadata']

                #dPosCam_TF           = np.array(metadataDict['dPosCam_TF'], dtype=np.float32)
                dAttDCM_fromTFtoCAM   = np.array(metadataDict['dAttDCM_fromTFtoCAM'], dtype=np.float32)
                #dSunDir_PixCoords    = np.array(metadataDict['dSunDir_PixCoords'], dtype=np.float32)
                dSunDirAngle          = np.array(metadataDict['dSunAngle'], dtype=np.float32)
                dLimbConicCoeffs_PixCoords = np.array(metadataDict['dLimbConic_PixCoords'], dtype=np.float32)
                #dRmoonDEM            = np.array(metadataDict['dRmoonDEM'], dtype=np.float32)
                dConicPixCentre        = np.array(metadataDict['dConicPixCentre'], dtype=np.float32)

                ui16coarseLimbPixels = np.array(tmpdataDict['ui16coarseLimbPixels'], dtype=np.float32)
                ui8flattenedWindows  = np.array(tmpdataDict['ui8flattenedWindows'], dtype=np.float32)
                centreBaseCost2 = np.array(tmpdataDict['dPatchesCentreBaseCost2'], dtype=np.float32)

                try:
                    dTruePixOnConic = np.array(tmpdataDict['dTruePixOnConic'], dtype=np.float32)
                except:
                    dTruePixOnConic = np.array(tmpdataDict['truePixOnConic'], dtype=np.float32)
                #invalidPatchesToCheck = {'ID': [], 'flattenedPatch': []}

                dConicAvgRadiusInPix = np.array(metadataDict['dTargetPixAvgRadius'], dtype=np.float32)

                if USE_NORMALIZED_IMG:
                    normalizationCoeff = 255.0
                else:
                    normalizationCoeff = 1.0

                for sampleID in range(ui16coarseLimbPixels.shape[1]):

                    print("\tProcessing samples: {current:8.0f}/{total:8.0f} of image {currentImgID:5.0f}/{totalImgs:5.0f}".format(current=sampleID+1, 
                    total=ui16coarseLimbPixels.shape[1], currentImgID=imgID+1, totalImgs=nImages), end="\r")

                    # Get flattened patch
                    flattenedWindow = ui8flattenedWindows[:, sampleID]
                    flattenedWindSize = len(flattenedWindow)
                    # Validate patch counting how many pixels are completely black or white

                    #pathIsValid = limbPixelExtraction_CNN_NN.IsPatchValid(flattenedWindow, lowerIntensityThr=10)

                    pathIsValid = True

                    if pathIsValid:
                        ptrToInput = 0

                        # Assign flattened window to input data array
                        inputDataArray[ptrToInput:ptrToInput+flattenedWindSize, saveID]  = flattenedWindow/normalizationCoeff

                        ptrToInput += flattenedWindSize # Update index

                        #inputDataArray[49, saveID]    = dRmoonDEM

                        # Assign Sun direction to input data array
                        inputDataArray[ptrToInput, saveID] = dSunDirAngle
                        ptrToInput += 1 # Update index

                        # Assign Attitude matrix as Modified Rodrigues Parameters to input data array
                        tmpVal = (Rotation.from_matrix(np.array(dAttDCM_fromTFtoCAM))).as_mrp() # Convert Attitude matrix to MRP parameters

                        inputDataArray[ptrToInput:ptrToInput+len(tmpVal), saveID] = tmpVal

                        ptrToInput += len(tmpVal) # Update index

                        #inputDataArray[55:58, saveID] = dPosCam_TF
                        if idModelClass in [1,2,3,4,5,6,7]:
                            inputDataArray[ptrToInput, saveID] = dConicAvgRadiusInPix
                            ptrToInput += dConicAvgRadiusInPix.size # Update index

                            inputDataArray[ptrToInput:ptrToInput+2, saveID] = dConicPixCentre
                            ptrToInput += dConicPixCentre.size # Update index

                        # Assign coarse Limb pixels to input data array
                        inputDataArray[ptrToInput:ptrToInput+len(ui16coarseLimbPixels[:, sampleID]), saveID] = ui16coarseLimbPixels[:, sampleID]
                        ptrToInput += len(ui16coarseLimbPixels[:, sampleID]) # Update index

                        if ptrToInput != inputSize:
                            raise IndexError('ERROR: Specified input data array size does not match with pointer to input allocator.')
                        
                        # Assign labels to labels data array (DO NOT CHANGE ORDER OF VALUES)
                        labelsDataArray[0:9, saveID] = np.ravel(dLimbConicCoeffs_PixCoords)
                        labelsDataArray[9:11, saveID] = ui16coarseLimbPixels[:, sampleID]
                        labelsDataArray[11, saveID] = centreBaseCost2[sampleID]
                        labelsDataArray[12:14, saveID] = dTruePixOnConic[:, sampleID]
                        
                        saveID += 1
                    #else:
                        # Save invalid patches to check
                        #invalidPatchesToCheck['ID'].append(sampleID)
                        #invalidPatchesToCheck['flattenedPatch'].append(flattenedWindow.tolist())
                
                # NOTE: transpose done using .T is not correct, transposition must be done on [0,1] axis only

                #reshapedPatchArray = inputDataArray[0:flattenedWindSize, 0:ui16coarseLimbPixels.shape[1]].reshape(
                #    (int(np.sqrt(flattenedWindSize)), -1, ui16coarseLimbPixels.shape[1]))
                #
                #limbPixelExtraction_CNN_NN.CreateCustomMosaicGray(reshapedPatchArray,
                #                                             ui16coarseLimbPixels[:, :], [1024, 1024], 
                #                                             dTruePixOnConic[:, :], global_points=None, save_path=None)

                imgID += 1

            # Save json with invalid patches to check
            #if idModelClass == 0:
            #    fileName = './invalidPatchesToCheck_Training_.json'
            #elif idModelClass == 1:
            #    fileName = './invalidPatchesToCheck_Validation.json'
            #with open(os.path.join(fileName), 'w') as fileCheck:
            #    invalidPatchesToCheck_string = json.dumps(invalidPatchesToCheck)
            #    json.dump(invalidPatchesToCheck_string, fileCheck)
            #    fileCheck.close()

            # Shrink dataset remove entries which have not been filled due to invalid path
            print('\n')
            print('\tNumber of images loaded from dataset: ', nImages)
            print('\tNumber of samples in dataset: ', nSamples)
            print('\tNumber of removed invalid patches from validity check:', nSamples - saveID)

            inputDataArray = inputDataArray[:, 0:saveID]           
            labelsDataArray = labelsDataArray[:, 0:saveID]

            # Apply standardization to input data # TODO: check if T is necessary
            #ACHTUNG: image may not be standardized? --> TBC
            inputDataArray[flattenedWindSize:, :] = preprocessing.StandardScaler().fit_transform(inputDataArray[flattenedWindSize:, :].T).T

            if idDataset == 0:
                dataDictTraining = {'labelsDataArray': labelsDataArray, 'inputDataArray': inputDataArray}
            elif idDataset == 1:   
                dataDictValidation = {'labelsDataArray': labelsDataArray, 'inputDataArray': inputDataArray}

            print('DATASET PREPARATION COMPLETED.')

        # INITIALIZE DATASET OBJECT # TEMPORARY from one single dataset
        datasetTraining = datasetPreparation.MoonLimbPixCorrector_Dataset(dataDictTraining)
        datasetValidation = datasetPreparation.MoonLimbPixCorrector_Dataset(dataDictValidation)

        # Save dataset as torch dataset object for future use
        limitSize = 4 * (2**30) * 8 # 4 Gbits
        sizeInBytes = np.sum([64 * entry.size for entry in dataDictTraining.values()])

        if sizeInBytes < limitSize:
            if not os.path.exists(datasetSavePath):
                os.makedirs(datasetSavePath)
                pyTorchAutoForge.SaveTorchDataset(datasetTraining, datasetSavePath, datasetName='TrainingDataset_'+TrainingDatasetTag)
                pyTorchAutoForge.SaveTorchDataset(datasetValidation, datasetSavePath, datasetName='ValidationDataset_'+ValidationDatasetTag)
        else:
            print('Dataset size exceeds 4 Gbits. Saving is skipped due to pickle limitation.')
    else:

        if not(os.path.isfile(os.path.join(datasetSavePath, 'TrainingDataset_'+TrainingDatasetTag+'.pt'))) or not((os.path.isfile(os.path.join(datasetSavePath, 'ValidationDataset_'+ValidationDatasetTag+'.pt')))):
            raise ImportError('Dataset files not found. Set REGEN_DATASET to True to regenerate the dataset torch saves.')
        
        # LOAD PRE-GENERATED DATASETS
        trainingDatasetPath = os.path.join(datasetSavePath, 'TrainingDataset_'+TrainingDatasetTag+'.pt')
        validationDatasetPath = os.path.join(datasetSavePath, 'ValidationDataset_'+ValidationDatasetTag+'.pt')
        
        print('Loading training and validation datasets from: \n\t', trainingDatasetPath, '\n\t', validationDatasetPath)

        datasetTraining = pyTorchAutoForge.LoadTorchDataset(datasetSavePath, datasetName='TrainingDataset_'+TrainingDatasetTag)
        datasetValidation = pyTorchAutoForge.LoadTorchDataset(datasetSavePath, datasetName='ValidationDataset_'+ValidationDatasetTag)
    
    ################################################################################################

    # Define dataloaders objects
    trainingDataset   = DataLoader(datasetTraining, batch_size, shuffle=True, num_workers=1, pin_memory=True)
    validationDataset = DataLoader(datasetValidation, batch_size, shuffle=True, num_workers=1, pin_memory=True)

    dataloaderIndex = {'TrainingDataLoader' : trainingDataset, 'ValidationDataLoader': validationDataset}
    
    # LOSS FUNCTION DEFINITION

    if LOSS_TYPE == 0:
        lossFcn = pyTorchAutoForge.CustomLossFcn(limbPixelExtraction_CNN_NN.MoonLimbPixConvEnhancer_LossFcn, lossParams)
    elif LOSS_TYPE == 1:
        lossFcn = pyTorchAutoForge.CustomLossFcn(limbPixelExtraction_CNN_NN.MoonLimbPixConvEnhancer_LossFcnWithOutOfPatchTerm, lossParams)
    elif LOSS_TYPE == 2:
        if USE_TENSOR_LOSS_EVAL:
            lossFcn = pyTorchAutoForge.CustomLossFcn(limbPixelExtraction_CNN_NN.MoonLimbPixConvEnhancer_NormalizedLossFcnWithOutOfPatchTerm_asTensor, lossParams)
        else:
            lossFcn = pyTorchAutoForge.CustomLossFcn(limbPixelExtraction_CNN_NN.MoonLimbPixConvEnhancer_NormalizedLossFcnWithOutOfPatchTerm, lossParams)
    elif LOSS_TYPE == 3:
        lossFcn = pyTorchAutoForge.CustomLossFcn(limbPixelExtraction_CNN_NN.MoonLimbPixConvEnhancer_PolarNdirectionDistanceWithOutOfPatch_asTensor, lossParams)
    elif LOSS_TYPE == 4:
        lossFcn = pyTorchAutoForge.CustomLossFcn(limbPixelExtraction_CNN_NN.MoonLimbPixConvEnhancer_NormalizedConicLossWithMSEandOutOfPatch_asTensor, lossParams)
    else:
        raise ValueError('Loss function ID not found.')

    # START MLFLOW RUN LOGGING

    with mlflow.start_run(nested=False) as mlflow_run:

        mlflow.log_param('TrainingDatasetTag', TrainingDatasetTag)
        mlflow.log_param('ValidationDatasetTag', ValidationDatasetTag)

        run_id = mlflow_run.info.run_id
        run_name = mlflow_run.info.run_name

        # Add run_id to save names
        modelSavePath += ('_' + run_name)
        modelArchName += ('_' + run_name)

        options['checkpointsOutDir'] = modelSavePath
        options['modelName'] = modelArchName

        mlflow.log_param('Output_channels_sizes', list(outChannelsSizes))
        mlflow.log_params(options)

        # MODEL CLASS TYPE
        if idModelClass == 0:
            modelClass = limbPixelExtraction_CNN_NN.HorizonExtractionEnhancerCNNv1max
        elif idModelClass == 1:
            modelClass = limbPixelExtraction_CNN_NN.HorizonExtractionEnhancerCNNv2max
        elif idModelClass == 2:
            modelClass = ModelClasses.HorizonExtractionEnhancer_CNNv3maxDeeper
        elif idModelClass == 3:
            modelClass = ModelClasses.HorizonExtractionEnhancer_FullyConnNNv4iter    
        elif idModelClass == 4:
            modelClass = ModelClasses.HorizonExtractionEnhancer_ShortCNNv6maxDeeper
        elif idModelClass == 5:
            modelClass = ModelClasses.HorizonExtractionEnhancer_deepNNv8
        elif idModelClass == 6:
            modelClass = ModelClasses.HorizonExtractionEnhancer_CNNv7
        elif idModelClass == 7:
            modelClass = ModelClasses.HorizonExtractionEnhancer_CNNvX_fullyParametric
        else:
            raise ValueError('Model class ID using max pooling not found.')
            
    
        # MODEL DEFINITION
        if restartTraining:
            checkPointPath = os.path.join(modelSavePath, modelName)
            if os.path.isfile(checkPointPath):
            
                print('RESTART training from checkpoint: ', checkPointPath)
                #modelEmpty = modelClass(outChannelsSizes, kernelSizes)
                modelCNN_NN = pyTorchAutoForge.LoadTorchModel(None, modelName, modelSavePath, loadAsTraced=True).to(device=device)
                #modelCNN_NN = customTorchTools.LoadTorchModel(modelCNN_NN, modelName)
    
            else:
                raise ValueError('Specified model state file not found. Check input path.')    
        else:
            if idModelClass == 0 or idModelClass == 1:
                modelCNN_NN = modelClass(outChannelsSizes, kernelSizes).to(device=device)
            elif idModelClass == 2:
                modelCNN_NN = modelClass(outChannelsSizes, kernelSizes, USE_BATCH_NORM).to(device=device)
            elif idModelClass == 3:
                modelCNN_NN = modelClass(outChannelsSizes).to(device=device)
            elif idModelClass == 4:
         
                mlflow.log_params(parametersConfig)
                # Kernel sizes: [5]
                modelCNN_NN = modelClass(outChannelsSizes, parametersConfig).to(device=device)
            elif idModelClass > 5:

                mlflow.log_params(parametersConfig)
                # Kernel sizes: [5]
                modelCNN_NN = modelClass(
                    parametersConfig).to(device=device)

        # Print summary of model
        #try:
        #    summary(modelCNN_NN, input_size=(1, inputSize), batch_size=batch_size, device=device)
        #except:
        #    print('torchsummary summary() function failed')
    
        # ######### TEST DEBUG ############
        #testPrediction = modelCNN_NN(torch.tensor(inputDataArray[0:, 0]))
    
        ############    END TEST DEBUG    ############
        #exit()
    
        #try:
        #    modelCNN_NN = torch.compile(modelCNN_NN) # ACHTUNG: compile is not compatible with jit.trace required to save traced model.
        #    print('Model compiled successfully.')
        #except:
        #    print('Model compilation failed. Using eager mode.')
    
        # Define optimizer object specifying model instance parameters and optimizer parameters
        if optimizerID == 0:
            optimizer = torch.optim.SGD(modelCNN_NN.parameters(), lr=initialLearnRate, momentum=momentumValue, weight_decay=0.00087)
        
        elif optimizerID == 1:
            optimizer = torch.optim.Adam(modelCNN_NN.parameters(), lr=initialLearnRate, betas=(0.9, 0.999), 
                                         eps=1e-08, weight_decay=0.00087, amsgrad=False, foreach=None, fused=True)
    
        for param_group in optimizer.param_groups:
            param_group['initial_lr'] = param_group['lr']

        if USE_SWA:
            swa_start_epoch = 15
            # NOTE: swa_lr is the constant value where the learning rate is annealed to following a cosine curve, then it remains fixed there.
            swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, anneal_strategy='cos', anneal_epochs=30, swa_lr=1E-7)  # SWA scheduler definition
            
            swa_model = torch.optim.swa_utils.AveragedModel(modelCNN_NN).to(device) # SWA model definition

            # Add to configuration dictionary
            options['swa_model'] = swa_model
            options['swa_scheduler'] = swa_scheduler
            options['swa_start_epoch'] = swa_start_epoch

        exponentialDecayGamma = 0.8
    
        for name, param in modelCNN_NN.named_parameters():
            mlflow.log_param(name, list(param.size()))
        
        if USE_LR_SCHEDULING:
            #optimizer = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, threshold=0.01, threshold_mode='rel', cooldown=1, min_lr=1E-12, eps=1e-08)
            #lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=exponentialDecayGamma, last_epoch=options['epochStart']-1)
            
            #lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=exponentialDecayGamma, last_epoch=options['epochStart']-1)
            #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=4, T_mult=2, eta_min=1e-9, last_epoch=options['epochStart']-1)
            if USE_SWA:
                #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, last_epoch=options['epochStart']-1)
                #lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, T_max=300, last_epoch=options['epochStart']-1)
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-8, last_epoch=options['epochStart']-1)

            else:
                lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=exponentialDecayGamma, last_epoch=options['epochStart']-1)
                # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=4, T_mult=2, eta_min=1e-9, last_epoch=options['epochStart']-1)

            options['lr_scheduler'] = lr_scheduler
    
        # Log model parameters
        mlflow.log_param('optimizer ID', optimizerID)
        mlflow.log_param('learning_rate', initialLearnRate)
        #mlflow.log_param('ExponentialDecayGamma', exponentialDecayGamma)

        numOfParameters = pyTorchAutoForge.getNumOfTrainParams(modelCNN_NN)
        mlflow.log_param('NumOfTrainParams', numOfParameters)

        # %% TRAIN and VALIDATE MODEL
        '''
        TrainAndValidateModel(dataloaderIndex:dict, model:nn.Module, lossFcn: nn.Module, optimizer, options:dict={'taskType': 'classification', 
                                                                                                                  'device': GetDevice(), 
                                                                                                                  'epochs': 10, 
                                                                                                                  'Tensorboard':True,
                                                                                                                  'saveCheckpoints':True,
                                                                                                                  'checkpointsDir': './checkpoints',
                                                                                                                  'modelName': 'trainedModel',
                                                                                                                  'loadCheckpoint': False,
                                                                                                                  'epochStart': 150}):
        '''
        (bestTrainedModelData, trainingLosses, validationLosses, inputSample) = pyTorchAutoForge.TrainAndValidateModel(dataloaderIndex, modelCNN_NN, lossFcn, optimizer, options)

    try:
    # %% Export trained model to ONNx and traced Pytorch format 
        if exportTracedModel:
           #customTorchTools.ExportTorchModelToONNx(trainedModel, inputSample, onnxExportPath='./checkpoints',
           #                                    onnxSaveName='trainedModelONNx', modelID=options['epochStart']+options['epochs'], onnx_version=14)
            tracedModelSaveName = os.path.join(tracedModelSavePath, modelArchName+'_'+str(pyTorchAutoForge.AddZerosPadding(bestTrainedModelData['epoch'], 3)))                     
            pyTorchAutoForge.SaveTorchModel(bestTrainedModelData['model'].to(device), modelName= tracedModelSaveName, saveAsTraced=True, 
                                            exampleInput=inputSample, targetDevice=device)
    except:
        print('Export of trained model failed.')
    

# %% MAIN SCRIPT
if __name__ == '__main__':
    
    # Set mlflow experiment
    #mlflow.set_experiment("HorizonEnhancerCNN_OptimizationRuns")
    if datasetID[0] == 6 and datasetID[1] == 7:
        mlflow.set_experiment("HorizonEnhancerCNN_OptimizationRuns_TrainValidOnLUMIOdata")
    #elif datasetID[0] == 9 and datasetID[1] == 10:
    #    mlflow.set_experiment("HorizonEnhancerCNN_OptimizationRuns_TrainValidOnRandomCloud")
    elif datasetID[0] == 9 and datasetID[1] == 10:
        mlflow.set_experiment("HorizonEnhancerCNNvX_TrainValidOnRandomCloud25x25")
    elif datasetID[0] == 12 and datasetID[1] == 13:
        mlflow.set_experiment("HorizonEnhancerCNNvX_TrainValidOnRandomCloud7x7_SWA")
    else:
        raise ValueError('Check datasetID. Experiment not found.')
    
    #mlflow.set_experiment("HorizonEnhancerCNN_Optimization_CNNv7_randomCloud")
    # Stop any running tensorboard session before starting a new one
    pyTorchAutoForge.KillTensorboard()

    # Setup multiprocessing for training the two models in parallel
    if USE_MULTIPROCESS == True:

        # Use the "spawn" start method (REQUIRED by CUDA)
        #multiprocessing.set_start_method('spawn')
        #process1 = multiprocessing.Process(target=main, args=(10,1, 4))
        #process2 = multiprocessing.Process(target=main, args=(10,1, 4))
#
        ## Start the processes
        #process1.start()
        #process2.start()
#
        ## Wait for both processes to finish
        #process1.join()
        #process2.join()
        raise ValueError('Multiprocessing not supported for this script. Use manual run instead.')
    
        print("Training complete for both network classes. Check the logs for more information.")
    elif USE_MULTIPROCESS == False and MANUAL_RUN == False:
        
        modelClassesToTrain = [0, 1, 2, 3]
        idLossType = 4
        idRun = 0

        for idModelClass in modelClassesToTrain:
            print('\n\n----------------------------------- RUNNING: TrainAndValidateCNN.py -----------------------------------\n')
            print("MAIN script operations: load dataset --> split dataset --> define dataloaders --> define model --> define loss function --> train and validate model --> export trained model\n")
            main(idRun, idModelClass, idLossType)
    
    else:
        idLossType = LOSS_TYPE
        RUN_OPTUNA_STUDY = False

        if RUN_OPTUNA_STUDY:
            # %% Optuna study configuration
            optunaStudyObj = optuna.create_study(study_name='CIFAR10_CNN_OptimizationExample', direction='maximize')

            # %% Optuna optimization
            #optunaStudyObj.optimize(objective, n_trials=50, timeout=3600)

            # Print the best trial
            print('Number of finished trials:', len(optunaStudyObj.trials)) # Get number of finished trials
            print('Best trial:')

            trial = optunaStudyObj.best_trial # Get the best trial from study object
            print('  Value: {:.4f}'.format(trial.value)) # Loss function value for the best trial

            # Print parameters of the best trial
            print('  Params: ')
            for key, value in trial.params.items():
                print('    {}: {}'.format(key, value))

        else:
            idRun = RUN_ID
            main(idRun, MODEL_CLASS_ID, idLossType)
