'''
Script to test the Polar-n-direction distance function for the evaluation of the distance of a point from a conic in the image place represented by its [3x3] matrix.
Created by PeterC, 26-06-2024. Reference: “[14] Y. Wu, H. Wang, F. Tang, and Z. Wang, “Efficient conic fitting with an analytical polar-n-direction geometric distance,
” Pattern Recognition, vol. 90, pp. 415-423, 2019.” 
DEVNOTE: old implementation, not actively maintained.
'''

# Import the required modules
import matplotlib.pyplot as plt
import torch
from pyTorchAutoForge.utils.DeviceManager import GetDeviceMulti
import numpy as np

def ComputePolarNdirectionDistance(conicMatrix: np.ndarray | torch.Tensor | list, 
                                   pointCoords: np.ndarray | torch.Tensor | list):
    '''
    Function to compute the Polar-n-direction distance of a point from a conic in the image plane represented by its [3x3] matrix.
    '''    
    # Convert the input to homogeneous coordinates
    pointHomoCoords = torch.tensor([pointCoords[0], pointCoords[1], 1], dtype=torch.float32).reshape(3,1)

    # Compute auxiliary variables
    CbarMatrix = torch.zeros((3,3))
    CbarMatrix[0:2,0:3] = conicMatrix.reshape(3,3)[0:2,0:3]

    Gmatrix = torch.matmul(conicMatrix, CbarMatrix)
    Wmatrix = torch.matmul(torch.matmul(CbarMatrix.T, conicMatrix), CbarMatrix)

    # Compute Gdist2, CWdist and Cdist
    Cdist = torch.matmul(pointHomoCoords.T, torch.matmul(conicMatrix, pointHomoCoords))
    
    Gdist = ( torch.matmul(pointHomoCoords.T, torch.matmul(Gmatrix, pointHomoCoords)) )
    Gdist2 = Gdist * Gdist
    
    Wdist = ( torch.matmul(pointHomoCoords.T, torch.matmul(Wmatrix, pointHomoCoords)) )
    CWdist = Cdist * Wdist

    # Compute the square distance depending on if condition
    if Gdist2 >= CWdist:

        sqrDist = Cdist / (Gdist * ( 1+torch.sqrt(1 + (Gdist2 - CWdist)/Gdist2) )**2)

    else:
        sqrDist = 0.25 * (Cdist**2 / Gdist)


    ##### DEBUG PRINTS ##########
    #print('CbarMatrix:', CbarMatrix)
    #print('Gmatrix:', Gmatrix)
    #print('Wmatrix:', Wmatrix)
    ############################

    return sqrDist

def ComputePolarNdirectionDistance_asTensor(conicMatrix: np.ndarray | torch.Tensor | list, 
                                   pointCoords: np.ndarray | torch.Tensor | list):
    '''
    Function to compute the Polar-n-direction distance of a point from a conic in the image plane represented by its [3x3] matrix using torch tensors operations.
    '''

    device = pointCoords.device
    # Shape point coordinates as tensor
    batchSize = pointCoords.shape[0]

    pointHomoCoords_tensor = torch.zeros((batchSize,3,1), dtype=torch.float32, device=device)
    pointHomoCoords_tensor[:, 0:2, 0] = torch.stack((pointCoords[:, 0], pointCoords[:, 1]), dim=1)
    pointHomoCoords_tensor[:, 2, 0] = 1

    # Reshape Conic matrix to tensor
    conicMatrix = conicMatrix.view(batchSize, 3, 3).to(device)

    CbarMatrix_tensor = torch.zeros((batchSize, 3, 3), dtype=torch.float32, device=device)
    CbarMatrix_tensor[:, 0:2,0:3] = conicMatrix[:, 0:2,0:3]

    Gmatrix_tensor = torch.bmm(conicMatrix, CbarMatrix_tensor)
    Wmatrix_tensor = torch.bmm(torch.bmm(CbarMatrix_tensor.transpose(1,2), conicMatrix), CbarMatrix_tensor)

    # Compute Gdist2, CWdist and Cdist
    Cdist_tensor = torch.bmm( pointHomoCoords_tensor.transpose(1, 2), torch.bmm(conicMatrix, pointHomoCoords_tensor) ) 
    
    Gdist_tensor = torch.bmm( pointHomoCoords_tensor.transpose(1, 2), torch.bmm(Gmatrix_tensor, pointHomoCoords_tensor) )
    Gdist2_tensor = torch.bmm(Gdist_tensor, Gdist_tensor)
    
    Wdist_tensor = torch.bmm(pointHomoCoords_tensor.transpose(1, 2), torch.bmm(Wmatrix_tensor, pointHomoCoords_tensor)) 
    CWdist_tensor = torch.bmm(Cdist_tensor, Wdist_tensor)


    # Get mask for the condition
    idsMask = Gdist2_tensor >= CWdist_tensor

    print('Any false in idsMask? ', torch.any(idsMask == False).item())

    notIdsMask = Gdist2_tensor < CWdist_tensor

    # Compute the square distance depending on if condition
    sqrDist_tensor = torch.zeros(batchSize, dtype=torch.float32, device=device)

    sqrDist_tensor[idsMask[:,0,0]] = Cdist_tensor[idsMask] / ( Gdist_tensor[idsMask] * ( 1 + torch.sqrt(1 + (Gdist2_tensor[idsMask] - CWdist_tensor[idsMask]) / Gdist2_tensor[idsMask]) )**2)
    sqrDist_tensor[notIdsMask[:,0,0]] = 0.25 * (Cdist_tensor[notIdsMask]**2 / Gdist_tensor[notIdsMask])

    return sqrDist_tensor


def main():

    import matplotlib

    # Determine default backend (select based on session)
    if matplotlib.is_interactive():
        matplotlib.use(backend='TkAgg')
    else:
        matplotlib.use(backend='Agg')

    print("Test the Polar-n-direction distance function for the evaluation of the distance of a point from a conic in the image place represented by its [3x3] matrix.")

    # %% TEST: check which point matplotlib and openCV uses as origin of the image
    # TODO

    # %% TEST: loss function
    # Construct meshgrid over [7x7] patch for evaluation
    Npoints = 500
    xSpace = np.linspace(0, 1024, Npoints)
    ySpace = np.linspace(0, 1024, Npoints)

    X, Y   = np.meshgrid(xSpace, ySpace)

    # Define the conic matrix as [9x1] vector
    conicMatrix = torch.tensor([2.0041524863091946E-6,
            0,
            -0.0010261260729903081,
            0,
            2.0041524863091946E-6,
            -0.0010261260729903076,
            -0.0010261260729903081,
            -0.0010261260729903076,
            1]).reshape(1, 3, 3).repeat(Npoints**2, 1, 1)
    
    # TEST: For loop version
    sqrDist_loop = np.zeros((Npoints, Npoints))

    for idX in range(xSpace.size):
        for idY in range(ySpace.size):
            sqrDist_loop[idX][idY] = ComputePolarNdirectionDistance(conicMatrix[0,:,:], torch.tensor([xSpace[idX], ySpace[idY]]))

    # TEST: Tensor evaluation version

    X = torch.tensor(X, dtype=torch.float32, device=GetDeviceMulti())
    Y = torch.tensor(Y, dtype=torch.float32, device=GetDeviceMulti())

    pixelCoords = torch.stack((X.flatten(), Y.flatten()), dim=1)

    # Evaluate the loss function over the meshgrid
    sqrDist_tensor = ComputePolarNdirectionDistance_asTensor(conicMatrix, pixelCoords)

    sqrDist_np = np.array(sqrDist_tensor.cpu().detach().numpy()).reshape(Npoints, Npoints).T

    # Plot the loss function values
    Nlevels = 100

    fig, ax = plt.subplots()

    X, Y = np.array(X.cpu().detach().numpy()), np.array(Y.cpu().detach().numpy())
    ax.contourf(X, Y, sqrDist_np, levels=Nlevels)
    
    # Plot distance as surface 
    fig2, ax2 = plt.subplots(subplot_kw={"projection": "3d"})
    
    ax2.plot_surface(X, Y, abs(sqrDist_np), cmap='viridis', edgecolor='none')

    # Plot loop version
    fig3, ax3 = plt.subplots(subplot_kw={"projection": "3d"})
    
    ax3.plot_surface(X, Y, sqrDist_loop**2, cmap='viridis', edgecolor='none')


    plt.show(block=True)

    diff = np.array(sqrDist_loop) - np.array(sqrDist_np)


if __name__ == "__main__":
    main()