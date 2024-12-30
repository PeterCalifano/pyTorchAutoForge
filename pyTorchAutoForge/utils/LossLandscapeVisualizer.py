import matplotlib.pyplot as plt
import numpy as np
import torch


# TODO
class LossLandscapeVisualizer():
    def __init__(self) -> None:
        pass



# %% Loss visualization Functions - PeterC - 06-07-2024 (EXPERIMENT)

# Example usage
# Assuming you have a dataloader `data_loader` and loss function `loss_fn`, call the function:
# plot_loss_landscape(mofrl, data_loader, loss_fn)

def GetRandomTensorDirection(model):
    direction = []
    # For each entry in parameters (a tensor!) in the model, create a random tensor with the same shape
    for param in model.parameters():  # What does parameters() return specifically?
        if param.requires_grad:
            direction.append(torch.randn_like(param))
    return direction


def FilterNormalizeDirections(d1, d2, model):
    norm_d1, norm_d2 = [], []
    for p1, p2, param in zip(d1, d2, model.parameters()):
        if param.requires_grad:
            norm = torch.norm(param)
            norm_d1.append(p1 / torch.norm(p1) * norm)
            norm_d2.append(p2 / torch.norm(p2) * norm)
    return norm_d1, norm_d2


def DisplaceModelParamsAlongDirections(model, direction, alpha):
    index = 0
    for param in model.parameters():
        if param.requires_grad:
            param.data.add_(direction[index], alpha=alpha)
            index += 1

# Auxiliary function (do not move to customTorchTools)


def EvalLossFcn(model, data_loader, loss_fn):
    model.eval()
    loss = 0.0
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)
            loss += loss_fn(outputs, targets).item()
    return loss / len(data_loader)


def Plot2DlossLandscape(model, dataloader, lossFcn, dir1Range=(-1, 1), dir2Range=(-1, 1), gridSize=0.1):
    # Get two sets of random directions and normalize them using filter normalization
    d1 = GetRandomTensorDirection(model)
    d2 = GetRandomTensorDirection(model)

    d1, d2 = FilterNormalizeDirections(d1, d2, model)

    # Pre-allocate plot grid
    gammaVals = np.arange(dir1Range[0], dir1Range[1], gridSize)
    etaVals = np.arange(dir2Range[0], dir2Range[1], gridSize)
    losses = np.zeros((len(gammaVals), len(etaVals)))

    originalModelState = [param.clone() for param in model.parameters()]

    for i, x in enumerate(gammaVals):
        for j, y in enumerate(etaVals):

            # Compute displaced model parameters along the two directions for a given x and y
            DisplaceModelParamsAlongDirections(model, d1, x)
            DisplaceModelParamsAlongDirections(model, d2, y)

            # Evaluate loss using displaced model
            losses[i, j] = EvalLossFcn(model, dataloader, lossFcn)

            # NOTE: Not sure what this does?
            for param, original_param in zip(model.parameters(), originalModelState):
                param.data.copy_(original_param)

    # Generate mesh grid for plotting
    X, Y = np.meshgrid(gammaVals, etaVals)
    Z = losses.T  # Transpose to match the correct orientation

    # Plot 2D contour of the loss landscape
    plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar()
    plt.xlabel('Direction 1')
    plt.ylabel('Direction 2')
    plt.title('Loss Landscape')
    plt.show()
