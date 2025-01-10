# Test installation by printing version in python
#echo -e "\e[32m-------------------------------------------------- Testing installations --------------------------------------------------\n\e[0m"
#echo -e "\n \e[32m \tTesting torch, torchvision, tensorrt, torch-tensorrt...\n"
#python -c "import torch; import torchvision; print('Torch Version:', torch.__version__); print('TorchVision Version:', torchvision.__version__); print('CUDA #Available:', torch.cuda.is_available());"
#python -c "import tensorrt; print('TensorRT Version:', tensorrt.__version__)"
#python -c "import torch_tensorrt; print('Torch-TensorRT Version:', torch_tensorrt.__version__)"
#
#echo -e "\n \e[32m \tTesting spiking_networks modules...\n"
#python -c "import norse; print('Norse Version:', norse.__version__)"
#python -c "import tonic; print('Tonic Version:', tonic.__version__)"
#
#echo -e "\n \e[32m \tTesting pyTorchAutoForge...\n"
#python -c "import pyTorchAutoForge; print('pyTorchAutoForge Version:', pyTorchAutoForge.__version__)"
#
#echo -e "\n \e[32m \tTesting nvidia-modelopt...\n"
#python -c "import modelopt.torch.quantization.extensions as ext; ext.precompile()" # Attempt to load modelopt and compile extensions


if __name__ == "__main__":

    import torch
    import torchvision
    import tensorrt
    import torch_tensorrt
    import pyTorchAutoForge

    print("-------------------------------------------------- Testing installation of venvTorch for Jetson --------------------------------------------------\n")

    print("Testing torch, torchvision, tensorrt...\n")
    print('\tTorch Version:', torch.__version__)
    print('\tTorchVision Version:', torchvision.__version__)
    print('\tCUDA Available:', torch.cuda.is_available())
    print('\tTensorRT Version:', tensorrt.__version__)

    print("\nTesting pyTorchAutoForge...\n")
    print('\tpyTorchAutoForge Version:', pyTorchAutoForge.__version__)

    print("\nTesting torch-tensorrt...\n")

    print('\tTorch-TensorRT Version:', torch_tensorrt.__version__)

    print("\nTesting spiking_networks modules...\n")
    #print('\tNorse Version:', norse.__version__)
    #print('\tTonic Version:', tonic.__version__)

    print("\nTesting nvidia-modelopt...\n")
    import modelopt
    import modelopt.torch.quantization.extensions as ext
    print('\tModelOpt Version:', modelopt.__version__)
    #ext.precompile()