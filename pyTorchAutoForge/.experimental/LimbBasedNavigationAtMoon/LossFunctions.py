import torch
from pyTorchAutoForge.utils.utils import GetDevice
import numpy as np
from typing import Union

# %% Custom loss function for Moon Limb pixel extraction CNN enhancer - 01-06-2024
def MoonLimbPixConvEnhancer_LossFcn(predictCorrection, labelVector, paramsTrain: dict = None, paramsEval: dict = None):
    # Alternative loss: alfa*||xCorrT * ConicMatr* xCorr||^2 + (1-alfa)*MSE(label, prediction)
    # Get parameters and labels for computation of the loss
    if paramsTrain is None:
        coeff = 0.5
    else:
        coeff = paramsTrain['ConicLossWeightCoeff']

    LimbConicMatrixImg = (labelVector[:, 0:9].T).reshape(
        3, 3, labelVector.size()[0]).T
    patchCentre = labelVector[:, 9:]

    # Evaluate loss
    conicLoss = 0.0  # Weighting violation of Horizon conic equation
    L2regLoss = 0.0

    for idBatch in range(labelVector.size()[0]):

        # Compute corrected pixel
        correctedPix = torch.tensor(
            [0, 0, 1], dtype=torch.float32, device=GetDevice()).reshape(3, 1)
        correctedPix[0:2] = patchCentre[idBatch, :].reshape(
            2, 1) + predictCorrection[idBatch, :].reshape(2, 1)

        conicLoss += torch.matmul(correctedPix.T, torch.matmul(
            LimbConicMatrixImg[idBatch, :, :].reshape(3, 3), correctedPix))

        # L2regLoss += torch.norm(predictCorrection[idBatch]) # Weighting the norm of the correction to keep it as small as possible

    if coeff == 1:
        L2regLoss = 0
    else:
        L2regLoss = torch.norm(predictCorrection, dim=1).sum()

    # Total loss function
    lossValue = coeff * torch.norm(conicLoss)**2 + (1-coeff) * L2regLoss
    return lossValue

# %% Custom loss function for Moon Limb pixel extraction CNN enhancer with strong loss for out-of-patch predictions - 23-06-2024


def MoonLimbPixConvEnhancer_LossFcnWithOutOfPatchTerm(predictCorrection, labelVector, paramsTrain: dict = None, paramsEval: dict = None):
    # Alternative loss: alfa*||xCorrT * ConicMatr* xCorr||^2 + (1-alfa)*MSE(label, prediction)
    # Get parameters and labels for computation of the loss

    if paramsTrain is None:
        coeff = 0.5
    else:
        coeff = paramsTrain['ConicLossWeightCoeff']

    # Temporary --> should come from paramsTrain dictionary
    patchSize = 7
    halfPatchSize = patchSize/2
    slopeMultiplier = 2

    LimbConicMatrixImg = (labelVector[:, 0:9].T).reshape(
        3, 3, labelVector.size()[0]).T
    patchCentre = labelVector[:, 9:]

    if 'RectExpWeightCoeff' in paramsTrain.keys():
        RectExpWeightCoeff = paramsTrain['RectExpWeightCoeff']
    else:
        RectExpWeightCoeff = 1

    # Evaluate loss
    outOfPatchoutLoss = 0.0
    conicLoss = 0.0  # Weighting violation of Horizon conic equation
    L2regLoss = 0.0

    for idBatch in range(labelVector.size()[0]):

        # Compute corrected pixel and conic loss
        correctedPix = torch.tensor(
            [0, 0, 1], dtype=torch.float32, device=GetDevice()).reshape(3, 1)
        correctedPix[0:2] = patchCentre[idBatch, :].reshape(
            2, 1) + predictCorrection[idBatch, :].reshape(2, 1)

        conicLoss += torch.matmul(correctedPix.T, torch.matmul(
            LimbConicMatrixImg[idBatch, :, :].reshape(3, 3), correctedPix))

        # Add average of the two coordinates to the total cost term
        outOfPatchoutLoss += outOfPatchoutLoss_Quadratic(predictCorrection[idBatch, :].reshape(
            2, 1), halfPatchSize=halfPatchSize, slopeMultiplier=slopeMultiplier)

        # L2regLoss += torch.norm(predictCorrection[idBatch]) # Weighting the norm of the correction to keep it as small as possible

    if coeff == 1:
        L2regLoss = 0
    else:
        L2regLoss = torch.norm(predictCorrection, dim=1).sum()

    # Total loss function
    lossValue = coeff * torch.norm(conicLoss)**2 + (1-coeff) * \
        L2regLoss + RectExpWeightCoeff * outOfPatchoutLoss

    return lossValue

# %% Additional Loss term for out-of-patch predictions based on Rectified Exponential function - 23-06-2024


def outOfPatchoutLoss_RectExp(predictCorrection, halfPatchSize=3.5, slopeMultiplier=2):
    if predictCorrection.size()[0] != 2:
        raise ValueError(
            'predictCorrection must have 2 rows for x and y pixel correction')

    # Compute the out-of-patch loss
    numOfCoordsOutOfPatch = 1
    tmpOutOfPatchoutLoss = 0.0

    # Compute the out-of-patch loss
    if abs(predictCorrection[0]) > halfPatchSize:
        tmpOutOfPatchoutLoss += torch.exp(slopeMultiplier *
                                          (predictCorrection[0]**2 - halfPatchSize**2))

    if abs(predictCorrection[1]) > halfPatchSize:
        tmpOutOfPatchoutLoss += torch.exp(slopeMultiplier *
                                          (predictCorrection[1]**2 - halfPatchSize**2))
        numOfCoordsOutOfPatch += 1

    if tmpOutOfPatchoutLoss > 0:
        if tmpOutOfPatchoutLoss.isinf():
            tmpOutOfPatchoutLoss = 1E4

    # Return the average of the two losses
    return tmpOutOfPatchoutLoss/numOfCoordsOutOfPatch

# %% Additional Loss term for out-of-patch predictions based on quadratic function - 25-06-2024


def outOfPatchoutLoss_Quadratic(predictCorrection, halfPatchSize=3.5, slopeMultiplier=0.2):
    if predictCorrection.size()[0] != 2:
        raise ValueError(
            'predictCorrection must have 2 rows for x and y pixel correction')

    # Compute the out-of-patch loss
    numOfCoordsOutOfPatch = 1
    tmpOutOfPatchoutLoss = 0.0

    # Compute the out-of-patch loss
    if abs(predictCorrection[0]) > halfPatchSize:
        tmpOutOfPatchoutLoss += torch.square(
            slopeMultiplier*(predictCorrection[0] - halfPatchSize)**2)

    if abs(predictCorrection[1]) > halfPatchSize:
        tmpOutOfPatchoutLoss += torch.square(
            slopeMultiplier*(predictCorrection[1] - halfPatchSize)**2)
        numOfCoordsOutOfPatch += 1

    if tmpOutOfPatchoutLoss > 0:
        if tmpOutOfPatchoutLoss.isinf():
            raise ValueError('tmpOutOfPatchoutLoss is infinite')

    # Return the average of the two losses
    return tmpOutOfPatchoutLoss/numOfCoordsOutOfPatch


def outOfPatchoutLoss_Quadratic_asTensor(predictCorrection: torch.tensor, halfPatchSize=3.5, slopeMultiplier=0.2):

    if predictCorrection.size()[1] != 2:
        raise ValueError(
            'predictCorrection must have 2 rows for x and y pixel correction')

    device = predictCorrection.device
    batchSize = predictCorrection.size()[0]

    numOfCoordsOutOfPatch = torch.ones(batchSize, 1, device=device)

    # Check which samples have the coordinates out of the patch
    idXmask = abs(predictCorrection[:, 0]) > halfPatchSize
    idYmask = abs(predictCorrection[:, 1]) > halfPatchSize

    numOfCoordsOutOfPatch += idYmask.view(batchSize, 1)

    tmpOutOfPatchoutLoss = torch.zeros(batchSize, 1, device=device)

    # Add contribution for all X coordinates violating the condition
    tmpOutOfPatchoutLoss[idXmask] += torch.square(slopeMultiplier*(
        predictCorrection[idXmask, 0] - halfPatchSize)**2).reshape(-1, 1)
    # Add contribution for all Y coordinates violating the condition
    tmpOutOfPatchoutLoss[idYmask] += torch.square(slopeMultiplier*(
        predictCorrection[idYmask, 1] - halfPatchSize)**2).reshape(-1, 1)

    if any(tmpOutOfPatchoutLoss > 0):
        if any(tmpOutOfPatchoutLoss.isinf()):
            raise ValueError('tmpOutOfPatchoutLoss is infinite')

    # Return the average of the two losses for each entry in the batch
    return torch.div(tmpOutOfPatchoutLoss, numOfCoordsOutOfPatch)

#######################################################################################################
# %% Custom normalized loss function for Moon Limb pixel extraction CNN enhancer - 23-06-2024


def MoonLimbPixConvEnhancer_NormalizedLossFcnWithOutOfPatchTerm(predictCorrection, labelVector, paramsTrain: dict = None, paramsEval: dict = None):

    # Get parameters and labels for computation of the loss
    if paramsTrain is None:
        coeff = 1
    else:
        coeff = paramsTrain['ConicLossWeightCoeff']

    if paramsTrain is None:
        RectExpWeightCoeff = 1
    elif 'RectExpWeightCoeff' in paramsTrain.keys():
        RectExpWeightCoeff = paramsTrain['RectExpWeightCoeff']

    # Temporary --> should come from params dictionary
    patchSize = 7
    halfPatchSize = patchSize/2
    slopeMultiplier = 2

    LimbConicMatrixImg = (labelVector[:, 0:9].T).reshape(
        3, 3, labelVector.size()[0]).T
    patchCentre = labelVector[:, 9:11]
    baseCost2 = labelVector[:, 11]

    # Evaluate loss
    normalizedConicLoss = 0.0  # Weighting violation of Horizon conic equation
    outOfPatchoutLoss = 0.0

    batchSize = labelVector.size()[0]

    for idBatch in range(labelVector.size()[0]):

        # Compute corrected pixel
        correctedPix = torch.tensor(
            [0, 0, 1], dtype=torch.float32, device=GetDevice()).reshape(3, 1)
        correctedPix[0:2] = patchCentre[idBatch, :].reshape(
            2, 1) + predictCorrection[idBatch, :].reshape(2, 1)

        normalizedConicLoss += ((torch.matmul(correctedPix.T, torch.matmul(
            LimbConicMatrixImg[idBatch, :, :].reshape(3, 3), correctedPix)))**2/baseCost2[idBatch])

        # Add average of the two coordinates to the total cost term
        outOfPatchoutLoss += outOfPatchoutLoss_Quadratic(predictCorrection[idBatch, :].reshape(
            2, 1), halfPatchSize=halfPatchSize, slopeMultiplier=slopeMultiplier)

    if coeff == 1:
        L2regLoss = 0
    else:
        L2regLoss = torch.norm(predictCorrection, dim=1).sum()

    # Total loss function
    lossValue = coeff * (normalizedConicLoss/batchSize) + (1-coeff) * \
        L2regLoss + RectExpWeightCoeff * outOfPatchoutLoss

    return lossValue


# %% Custom normalized loss function for Moon Limb pixel extraction CNN enhancer tensorized evaluation - 27-06-2024
def MoonLimbPixConvEnhancer_NormalizedLossFcnWithOutOfPatchTerm_asTensor(predictCorrection, labelVector, params: dict = None):

    # Get parameters and labels for computation of the loss
    TrainingMode = params.get('TrainingMode', True)
    paramsTrain = params.get(
        'paramsTrain', {'ConicLossWeightCoeff': 1, 'RectExpWeightCoeff': 1})
    paramsEval = params.get(
        'paramsEval', {'ConicLossWeightCoeff': 1, 'RectExpWeightCoeff': 0})

    if TrainingMode:
        RectExpWeightCoeff = paramsTrain.get('RectExpWeightCoeff', 1)
        ConicLoss2WeightCoeff = paramsTrain.get('ConicLossWeightCoeff', 1)
    else:
        RectExpWeightCoeff = paramsEval.get('RectExpWeightCoeff', 1)
        ConicLoss2WeightCoeff = paramsEval.get('ConicLossWeightCoeff', 1)

    # Temporary --> should come from params dictionary
    patchSize = params.get('patchSize', 7)
    slopeMultiplier = params.get('slopeMultiplier', 2)
    halfPatchSize = patchSize/2

    # Temporary --> should come from params dictionary
    patchSize = 7
    halfPatchSize = patchSize/2
    slopeMultiplier = 2

    # Extract data from labelVector
    batchSize = labelVector.size()[0]
    device = labelVector.device

    LimbConicMatrixImg = torch.tensor((labelVector[:, 0:9].T).reshape(
        3, 3, labelVector.size()[0]).T, dtype=torch.float32, device=device)

    patchCentre = labelVector[:, 9:11]
    baseCost2 = labelVector[:, 11]

    # Evaluate loss terms
    # normalizedConicLoss = torch.zeros(batchSize, 1, 1, dtype=torch.float32, device=device) # Weighting violation of Horizon conic equation
    # outOfPatchoutLoss = torch.zeros(batchSize, 1, 1, dtype=torch.float32, device=device)

    # Compute corrected pixel
    correctedPix = torch.zeros(
        batchSize, 3, 1, dtype=torch.float32, device=device)

    correctedPix[:, 2, 0] = 1
    correctedPix[:, 0:2, 0] = patchCentre + predictCorrection

    normalizedConicLoss = torch.div((torch.bmm(correctedPix.transpose(1, 2), torch.bmm(
        LimbConicMatrixImg, correctedPix)))**2, baseCost2.reshape(batchSize, 1, 1))

    # Add average of the two coordinates to the total cost term
    outOfPatchoutLoss = outOfPatchoutLoss_Quadratic_asTensor(
        predictCorrection, halfPatchSize=halfPatchSize, slopeMultiplier=slopeMultiplier).reshape(batchSize, 1, 1)

    if ConicLoss2WeightCoeff == 1:
        L2regLoss = 0
    else:
        L2regLoss = torch.norm(predictCorrection, dim=1).sum()

    # Total loss function
    lossValue = ConicLoss2WeightCoeff * (normalizedConicLoss.sum()) + (
        1-ConicLoss2WeightCoeff) * L2regLoss + RectExpWeightCoeff * outOfPatchoutLoss.sum()

    # Return sum of loss for the whole batch
    return lossValue/batchSize


# %% Polar-n-direction loss function for Moon Limb pixel extraction CNN enhancer tensorized evaluation - 27-06-2024
def MoonLimbPixConvEnhancer_PolarNdirectionDistanceWithOutOfPatch_asTensor(predictCorrection, labelVector, paramsTrain: dict = None, paramsEval: dict = None):
    # Get parameters and labels for computation of the loss

    if paramsTrain is None:
        RectExpWeightCoeff = 1
    elif 'RectExpWeightCoeff' in paramsTrain.keys():
        RectExpWeightCoeff = paramsTrain['RectExpWeightCoeff']

    slopeMultiplier = 4

    # Temporary --> should come from params dictionary
    patchSize = 7
    halfPatchSize = patchSize/2

    # Extract data from labelVector
    batchSize = labelVector.size()[0]
    device = labelVector.device

    LimbConicMatrixImg = torch.tensor((labelVector[:, 0:9].T).reshape(
        3, 3, labelVector.size()[0]).T, dtype=torch.float32, device=device)
    patchCentre = labelVector[:, 9:11]

    # Evaluate loss terms
    outOfPatchoutLoss = torch.zeros(
        batchSize, 1, 1, dtype=torch.float32, device=device)

    # Compute corrected pixel
    correctedPix = torch.zeros(
        batchSize, 3, 1, dtype=torch.float32, device=device)

    correctedPix[:, 2, 0] = 1
    correctedPix[:, 0:2, 0] = patchCentre + predictCorrection

    # Compute the Polar-n-direction distance
    polarNdirectionDist = ComputePolarNdirectionDistance_asTensor(
        LimbConicMatrixImg, correctedPix)

    # Add average of the two coordinates to the total cost term
    outOfPatchoutLoss += outOfPatchoutLoss_Quadratic_asTensor(
        predictCorrection, halfPatchSize=halfPatchSize, slopeMultiplier=slopeMultiplier).reshape(batchSize, 1, 1)

    # Total loss function
    lossValue = polarNdirectionDist + RectExpWeightCoeff * outOfPatchoutLoss

    # Return sum of loss for the whole batch
    return torch.sum(lossValue/batchSize)


def ComputePolarNdirectionDistance_asTensor(CconicMatrix: Union[np.array, torch.tensor, list],
                                            pointCoords: Union[np.array, torch.tensor]):
    '''
    Function to compute the Polar-n-direction distance of a point from a conic in the image plane represented by its [3x3] matrix using torch tensors operations.
    '''
    device = pointCoords.device
    # Shape point coordinates as tensor
    batchSize = pointCoords.shape[0]

    pointHomoCoords_tensor = torch.zeros(
        (batchSize, 3, 1), dtype=torch.float32, device=device)
    pointHomoCoords_tensor[:, 0:2, 0] = torch.stack(
        (pointCoords[:, 0, 0], pointCoords[:, 1, 0]), dim=1).reshape(batchSize, 2)
    pointHomoCoords_tensor[:, 2, 0] = 1

    # Reshape Conic matrix to tensor
    CconicMatrix = CconicMatrix.view(batchSize, 3, 3).to(device)

    CbarMatrix_tensor = torch.zeros(
        (batchSize, 3, 3), dtype=torch.float32, device=device)
    CbarMatrix_tensor[:, 0:2, 0:3] = CconicMatrix[:, 0:2, 0:3]

    Gmatrix_tensor = torch.bmm(CconicMatrix, CbarMatrix_tensor)
    Wmatrix_tensor = torch.bmm(
        torch.bmm(CbarMatrix_tensor.transpose(1, 2), CconicMatrix), CbarMatrix_tensor)

    # Compute Gdist2, CWdist and Cdist
    Cdist_tensor = torch.bmm(pointHomoCoords_tensor.transpose(
        1, 2), torch.bmm(CconicMatrix, pointHomoCoords_tensor))

    Gdist_tensor = torch.bmm(pointHomoCoords_tensor.transpose(
        1, 2), torch.bmm(Gmatrix_tensor, pointHomoCoords_tensor))
    Gdist2_tensor = torch.bmm(Gdist_tensor, Gdist_tensor)

    Wdist_tensor = torch.bmm(pointHomoCoords_tensor.transpose(
        1, 2), torch.bmm(Wmatrix_tensor, pointHomoCoords_tensor))
    CWdist_tensor = torch.bmm(Cdist_tensor, Wdist_tensor)

    # Get mask for the condition
    idsMask = Gdist2_tensor >= CWdist_tensor

    notIdsMask = Gdist2_tensor < CWdist_tensor

    # Compute the square distance depending on if condition
    sqrDist_tensor = torch.zeros(batchSize, dtype=torch.float32, device=device)

    sqrDist_tensor[idsMask[:, 0, 0]] = Cdist_tensor[idsMask] / (Gdist_tensor[idsMask] * (
        1 + torch.sqrt(1 + (Gdist2_tensor[idsMask] - CWdist_tensor[idsMask]) / Gdist2_tensor[idsMask]))**2)
    sqrDist_tensor[notIdsMask[:, 0, 0]] = 0.25 * \
        (Cdist_tensor[notIdsMask]**2 / Gdist_tensor[notIdsMask])

    # Return mean over the whole batch
    return abs(sqrDist_tensor)


# %% MSE + Conic Loss function for Moon Limb pixel extraction CNN enhancer tensorized evaluation - 27-06-2024
def MoonLimbPixConvEnhancer_NormalizedConicLossWithMSEandOutOfPatch_asTensor(predictCorrection, labelVector,
                                                                             params: dict = None):

    # Get parameters and labels for computation of the loss
    TrainingMode = params.get('TrainingMode', True)
    paramsTrain = params.get(
        'paramsTrain', {'ConicLossWeightCoeff': 1, 'RectExpWeightCoeff': 1})
    paramsEval = params.get(
        'paramsEval', {'ConicLossWeightCoeff': 1, 'RectExpWeightCoeff': 0})

    if TrainingMode:
        RectExpWeightCoeff = paramsTrain.get('RectExpWeightCoeff', 1)
        ConicLoss2WeightCoeff = paramsTrain.get('ConicLossWeightCoeff', 1)
    else:
        RectExpWeightCoeff = paramsEval.get('RectExpWeightCoeff', 1)
        ConicLoss2WeightCoeff = paramsEval.get('ConicLossWeightCoeff', 1)

    # Temporary --> should come from params dictionary
    patchSize = params.get('patchSize', 7)
    slopeMultiplier = params.get('slopeMultiplier', 2)
    halfPatchSize = patchSize/2

    # Extract data from labelVector
    batchSize = labelVector.size()[0]
    device = predictCorrection.device

    # Step 1: Select the first 9 columns for all rows
    # Step 2: Permute the dimensions to match the transposition (swap axes 0 and 1)
    # Step 3: Reshape the permuted tensor to the specified dimensions
    # Step 4: Permute again to match the final transposition (swap axes 0 and 2)

    LimbConicMatrixImg = ((labelVector[:, 0:9].permute(1, 0)).reshape(
        3, 3, labelVector.size()[0]).permute(2, 0, 1)).clone().to(device)

    assert (LimbConicMatrixImg.shape[0] == batchSize)

    patchCentre = labelVector[:, 9:11]  # Patch centre coordinates (pixels)
    # Base cost for the conic constraint evaluated at patch centre
    baseCost2 = labelVector[:, 11]
    # Target prediction for the pixel correction
    targetPrediction = labelVector[:, 12:14]

    # Initialize arrays for the loss terms
    # normalizedConicLoss = torch.zeros(batchSize, 1, dtype=torch.float32, device=device) # Weighting violation of Horizon conic equation
    # outOfPatchoutLoss = torch.zeros(batchSize, 1, dtype=torch.float32, device=device)

    # Compute corrected pixel
    correctedPix = torch.zeros(
        batchSize, 3, 1, dtype=torch.float32, device=device)
    correctedPix[:, 2, 0] = 1
    correctedPix[:, 0:2, 0] = patchCentre + predictCorrection

    # Compute the normalized conic loss
    if ConicLoss2WeightCoeff != 0:
        # normalizedConicLoss = torch.div( ((torch.bmm(correctedPix.transpose(1,2), torch.bmm(LimbConicMatrixImg, correctedPix)) )**2).reshape(batchSize, 1), baseCost2.reshape(batchSize, 1))
        unnormalizedConicLoss = ((torch.bmm(correctedPix.transpose(1, 2), torch.bmm(
            LimbConicMatrixImg, correctedPix)))**2).reshape(batchSize, 1)

    else:
        unnormalizedConicLoss = torch.zeros(
            batchSize, 1, dtype=torch.float32, device=device)

    # Compute the MSE loss term
    mseLoss = torch.nn.functional.mse_loss(
        correctedPix[:, 0:2, 0], targetPrediction, size_average=None, reduce=None, reduction='mean')

    if RectExpWeightCoeff != 0:
        # Add average of the two coordinates to the out of patch cost term
        outOfPatchLoss = outOfPatchoutLoss_Quadratic_asTensor(
            predictCorrection, halfPatchSize=halfPatchSize, slopeMultiplier=slopeMultiplier)
    else:
        outOfPatchLoss = torch.zeros(
            batchSize, 1, dtype=torch.float32, device=device)

    # Total loss function
    normalizedConicLossTerm = ConicLoss2WeightCoeff * \
        torch.sum(unnormalizedConicLoss)/batchSize
    outOfPatchLossTerm = torch.sum(
        (RectExpWeightCoeff * outOfPatchLoss))/batchSize
    lossValue = normalizedConicLossTerm + mseLoss + outOfPatchLossTerm

    # Return sum of loss for the whole batch
    return {'lossValue': lossValue, 'normalizedConicLoss': normalizedConicLossTerm, 'mseLoss': mseLoss, 'outOfPatchoutLoss': outOfPatchLossTerm}
