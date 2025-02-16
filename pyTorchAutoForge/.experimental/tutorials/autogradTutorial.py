# Script implementing the PyTorch autograd tutoriPytorch basics) as exercise by PeterC - 02-07-2024
# Reference: https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html
import torch


# KEY NOTE: We can only obtain the grad properties for the leaf nodes of the computational graph, which have requires_grad property set to True. 
#       For all other nodes in our graph, gradients will not be available. Leaf nodes are all tensors involved in the computation for which 
#       requires_grad=True.
# KEY NOTE: We can only perform gradient calculations using backward once on a given graph, for performance reasons. 
#       If we need to do several backward calls on the same graph, we need to pass retain_graph=True to the backward call.
#       The reason is that PyTorch builds the graph dynamically each time. Additionally, calling backward multiple times 
#       will accumulate the gradients in the leaf nodes, leading to the wrong results if zero_grad() is not called.
#       DAGs are dynamic in PyTorch An important thing to note is that the graph is recreated from scratch; after each .backward() call, autograd starts populating a new graph. 
#       This is exactly what allows you to use control flow statements in your model; you can change the shape, size and operations at every iteration if needed.

def main():

    print('------------------------------- TEST: PyTorch autograd tutorial -------------------------------')
    inputTensor = torch.ones(5)  # input tensor
    inputTensor.requires_grad = True  # Set requires_grad to track forward and backward gradients for the "self" tensor
    # Example use: parameters that are trainable, which must be tracked in the computational graphs must have required_grad=True

    targetOutput = torch.zeros(3)  # expected output

    weightMatrix = torch.randn(5, 3, requires_grad=True)
    biasVector = torch.randn(3, requires_grad=True)

    # Perform computation of the linear layer
    layerOutput = torch.matmul(inputTensor, weightMatrix) + biasVector

    # Compute the loss
    loss = torch.nn.functional.binary_cross_entropy_with_logits(layerOutput, targetOutput)

    # Compute gradients of the loss
    loss.backward()

    print('Gradient of the loss wrt weightMatrix: ', weightMatrix.grad)
    print('Gradient of the loss wrt biasVector: ', biasVector.grad)

    # Detach from graph to stop tracking history 
    weightsDetached = weightMatrix.detach() # This does NOT create a copy of the data, but detaches the data itself.
    clonerWeights = weightMatrix.clone().detach() # This creates a copy of the data, and detaches the copy from the graph.

    # Jacobian Product example and demonstration of why one has to zero the gradients after each backward call
    inp = torch.eye(4, 5, requires_grad=True)
    out = (inp+1).pow(2).t()

    # By calling backward with an argument, the Jacobian along the direction of the vector is computed, instead of the whole Jacobian matrix
    out.backward(torch.ones_like(out), retain_graph=True)
    print(f"First call\n{inp.grad}")

    out.backward(torch.ones_like(out), retain_graph=True)
    print(f"\nSecond call\n{inp.grad}")

    inp.grad.zero_()

    out.backward(torch.ones_like(out), retain_graph=True)
    print(f"\nCall after zeroing gradients\n{inp.grad}")