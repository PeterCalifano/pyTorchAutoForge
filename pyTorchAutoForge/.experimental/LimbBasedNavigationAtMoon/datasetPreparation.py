# Script to prepare datasets for training/validation/testing of CNN-NN for Limb based navigation enhancement
# Created by PeterC 31-05-2024

import sys, platform, os
import argparse
import csv 
import json

# Auxiliary modules
import numpy as np
import matplotlib as mpl
from scipy.spatial.transform import Rotation
from sklearn import preprocessing

# Torch modules
import pyTorchAutoForge


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Dataset preparation pipeline.')

    parser.add_argument('--loadMode',
                        type=str,
                        default='json',
                        help='Specify the data format of the datapairs to process.')
    
    parser.add_argument('--dataPath',
                        type=str,
                        default='.',
                        help='Path to JSON file containing datapairs to process.')

    parser.add_argument('--outputPath',
                        type=str,
                        default='dataset',
                        help='Path to folder where postprocessed dataset will be saved.')

    return parser.parse_args()


def LoadJSONdata(dataFilePath):

    if not (os.path.isfile(dataFilePath)):
        raise FileExistsError('Data file not found. Check specified dataPath.')
    
    #print('Data file FOUND, loading...')
    with open(dataFilePath, 'r') as json_file:
        try:
            dataJSONdict = json.load(json_file) # Load JSON as dict
        except Exception as exceptInstance:
            raise Exception('ERROR occurred:', exceptInstance.args)
        #print('Data file: LOADED.')

    if isinstance(dataJSONdict, dict):
        # Get JSONdict data keys
        dataKeys = dataJSONdict.keys()
        # Print JSON structure
        
        #print('Loaded JSON file structure:')
        #print(json.dumps(dataJSONdict, indent=2, default=str))

    elif isinstance(dataJSONdict, list):
        raise Exception('Decoded JSON as list not yet handled by this implementation. If JSON comes from MATLAB jsonencode(), make sure you are providing a struct() as input and not a cell.')
    else: 
        raise Exception('ERROR: incorrect JSON file formatting')
    
    return dataJSONdict, dataKeys
    


# %% Custom Dataset class for Moon Limb pixel extraction CNN enhancer - 01-06-2024
# First prototype completed by PC - 04-06-2024 --> to move to new module
class MoonLimbPixCorrector_Dataset():

    def __init__(self, dataDict:dict, datasetType:str='train', transform=None, target_transform=None):
            # Store input and labels sources
            self.labelsDataArray = dataDict['labelsDataArray']
            self.inputDataArray = dataDict['inputDataArray']

            # Initialize transform objects
            self.transform = transform
            self.target_transform = target_transform

            # Set the dataset type (train, test, validation)
            self.datasetType = datasetType

    def __len__(self):
        return np.shape(self.labelsDataArray)[1]

    # def __getLabelsData__(self):
    #     self.labelsDataArray

    def __getitem__(self, index):
        label   = self.labelsDataArray[:, index]
        inputVec = self.inputDataArray[:, index]

        return inputVec, label
    
# %% Moon Limb pixel extraction dataset builder - 08-07-2024


def BuildDataset(datasetInfo:dict, DatasetClass:callable):
    '''
    Function to build a dataset from a set of JSON files containing datapairs.
    Current limitation: some keys are necessary to be present in the JSON files to build the dataset.
    Next version should modify this. Additionally, transforms for the data should be made more generic.
    '''

    # Get information from datasetInfo
    datasetRootPath = datasetInfo['datasetRootName']
    inputDataArraySize = datasetInfo['inputDataArraySize']
    labelsDataArraySize = datasetInfo['labelsDataArraySize']

    datasetSavePath = datasetInfo['datasetSavePath']
    datasetName = datasetInfo['datasetName']
    datasetID = datasetInfo['datasetID']

    keyList_data = datasetInfo['inputDataKeyList']
    keyList_labels = datasetInfo['labelsDataKeyList']

    keyExcludeFromNormalization = datasetInfo.get('keyExcludeFromNormalization', [])
    ApplyNormalization = datasetInfo.get('ApplyNormalization', True)

    keyList = keyList_data + keyList_labels # Concatenate keys for iteration

    datasetType = datasetInfo['datasetType']

    # Scan directory to get the dataset folder corresponding to datasetID
    with os.scandir(datasetRootPath) as it:
        modelNamesWithID = [(entry.name, entry.name[3:6]) for entry in it if entry.is_dir()]
        datasetPathsTuples = sorted(modelNamesWithID, key=lambda x: int(x[1]))

    datasetPaths = [stringVal for stringVal, _ in datasetPathsTuples]

    #datasetFolderPath = os.path.join(datasetRootPath, 'datasets')

    print('Generating dataset from: ', datasetPaths[datasetID])

    # Get images and labels from IDth dataset
    dataDirPath = os.path.join(datasetRootPath, datasetPaths[datasetID])
    dataFilenames = os.listdir(dataDirPath) # NOTE: processing order does not matter
    nImages = len(dataFilenames)

    print('Found number of images:', nImages)

    # DEBUG
    #print(dataFilenames)

    # Get nPatches from the first datapairs files
    dataFileID = 0
    dataFilePath = os.path.join(dataDirPath, dataFilenames[dataFileID])
    #print('Loading data from:', dataFilePath)
    tmpdataDict, tmpdataKeys = LoadJSONdata(dataFilePath)
    print('All data loaded correctly --> DATASET PREPARATION BEGIN')

    # DEBUG
    #print(tmpdataKeys)
    nSamplesList = []

    for id, data in enumerate(tmpdataDict['ui16coarseLimbPixels']):
        nSamplesList.append(len(data))

    nSamples = np.sum(nSamplesList)
    
    # NOTE: each datapair corresponds to an image (i.e. nPatches samples)
    # Initialize dataset building variables
    saveID = 0
    imgID = 0

    inputDataArray  = np.zeros((inputDataArraySize, nSamples), dtype=np.float32)
    labelsDataArray = np.zeros((labelsDataArraySize, nSamples), dtype=np.float32)

    # Entries in datapairs:
    '''
    'dAttDCM_fromTFtoCAM'
    'dSunDir_PixCoords'
    'dPosCam_TF'
    'dLimbConic_PixCoords'
    'ui8flattenedWindows'
    'dPatchesCentreBaseCost2'
    'dTruePixOnConic'
    'ui16coarseLimbPixels'
    'dTargetPixAvgRadius'
    'dConicPixCente'
    'dSunAngle'
    'ui8flattenedEdgeMask'
    '''
    if ApplyNormalization:
        normalizationCoeff = 255.0
    else:
        normalizationCoeff = 1.0
    
    ptrToSamples = 0

    # Add ids to list of rows to include in Scaler transformation
    rowsIDsToIncludeInScaler = np.array([], dtype=np.int32)

    for idK, key in enumerate(keyList_data):
        if not(key in keyExcludeFromNormalization):
            rowsIDsToIncludeInScaler = np.concatenate((rowsIDsToIncludeInScaler, ptrToInput-dataMatrixRowSize))

    for idD, dataPair in enumerate(dataFilenames):
        
        print("\tProcessing datapair of image {currentImgID:5.0f}/{totalImgs:5.0f}".format(currentImgID=imgID+1, totalImgs=nImages), end="\r")

        # Data dict for ith image
        tmpdataDict, tmpdataKeys = LoadJSONdata(os.path.join(dataDirPath, dataPair))

        # Unpack data from JSON
        metadataDict = tmpdataDict['metadata']
        tmpMetadataKeys = metadataDict.keys()

        ptrToInput = 0
        ptrToLabels = 0

        # Get number of samples in current datapair
        NsamplesInDatapair = nSamplesList[idD]

        # Assignment over keys in datapairs (upgrade from assignment over samples)
        for idK, key in enumerate(keyList):
            
            if key in tmpdataKeys:
                # Entry of inputDataArray
                tmpDataMatrix = np.array(metadataDict[key], dtype=np.float32)
                
                # TEMPORARY --> dataset specific
                if key == 'ui8flattenedWindows':
                    # Apply normalization to input data
                    tmpDataMatrix = tmpDataMatrix/normalizationCoeff
                elif key == 'dAttDCM_fromTFtoCAM':
                    # Convert Attitude matrix to MRP parameters
                    tmpDataMatrix = (Rotation.from_matrix(tmpDataMatrix)).as_mrp()
                    # To test: does assignment of this entry is handled automatically through broadcasting?

                # Get size of data matrix (rows)
                dataMatrixRowSize = tmpDataMatrix.shape[0]

                if key in keyList_data:
                    # Assign data matrix to input data array
                    inputDataArray[ptrToInput:ptrToInput+dataMatrixRowSize, ptrToSamples:ptrToSamples+NsamplesInDatapair]  = tmpDataMatrix
                    # Update pointer index
                    ptrToInput += dataMatrixRowSize

                elif key in keyList_labels:
                    # Assign data matrix to labels data array
                    labelsDataArray[ptrToLabels:ptrToLabels+dataMatrixRowSize, ptrToSamples:ptrToSamples+NsamplesInDatapair]  = tmpDataMatrix
                    # Update pointer index
                    ptrToLabels += dataMatrixRowSize

            elif key in tmpMetadataKeys:
                # Entry of labelsDataArray
                tmpDataMatrix = np.array(tmpdataDict[key], dtype=np.float32)

                # Get size of data matrix (rows)
                dataMatrixRowSize = tmpDataMatrix.shape[0]

                if key in keyList_data:
                    # Assign data matrix to input data array
                    inputDataArray[ptrToInput:ptrToInput+dataMatrixRowSize, ptrToSamples:ptrToSamples+NsamplesInDatapair]  = tmpDataMatrix
                    # Update pointer index
                    ptrToInput += dataMatrixRowSize 

                elif key in keyList_labels:
                    # Assign data matrix to labels data array
                    labelsDataArray[ptrToLabels:ptrToLabels+dataMatrixRowSize, ptrToSamples:ptrToSamples+NsamplesInDatapair]  = tmpDataMatrix
                    # Update pointer index
                    ptrToLabels += dataMatrixRowSize

            else:
                raise Exception('Key not found in data dictionary')

        # Update pointer to samples 
        ptrToSamples += NsamplesInDatapair
        imgID += 1

        # DEVNOTE: TODO code below

    if ApplyNormalization:
        # Apply standardization to input data # TODO: check if T is necessary
        #ACHTUNG: image may not be standardized? --> TBC
        inputDataArray[rowsIDsToIncludeInScaler, :] = preprocessing.StandardScaler().fit_transform(
                                                        inputDataArray[rowsIDsToIncludeInScaler, :].T).T

    dataDict = {'labelsDataArray': labelsDataArray, 'inputDataArray': inputDataArray}
    print('DATASET PREPARATION COMPLETED.')

    if DatasetClass is not None:
        # INITIALIZE DATASET OBJECT if not None
        datasetTraining = DatasetClass(dataDict)

        # Save dataset as torch dataset object for future use
        if not os.path.exists(datasetSavePath):
            os.makedirs(datasetSavePath)

        pyTorchAutoForge.SaveTorchDataset(datasetTraining, datasetSavePath, datasetName=datasetName)
    else:
        datasetTraining = None

    return dataDict, datasetTraining

def main(args):
    print('No code to execute as main')
    if args.loadMode == 'json':
        dataDict, dataKeys = LoadJSONdata(args.dataPath)
        

# Script executed as main
if __name__ == '__main__':
    print('Executing script as main')
    args = parse_args()
    main(args)