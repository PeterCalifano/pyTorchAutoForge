
# Import modules
import pyTorchAutoForge  # Custom torch tools

def main():

    # SETTINGS and PARAMETERS 
    batch_size = 16*2 # Defines batch size in dataset
    TRAINING_PERC = 0.80
    #outChannelsSizes = [16, 32, 75, 15] 
    outChannelsSizes = [16, 32, 75, 15] 
    kernelSizes = [3, 1]
    learnRate = 1E-10
    momentumValue = 0.001
    optimizerID = 1 # 0
    device = pyTorchAutoForge.GetDevice()
    exportTracedModel = True


if __name__ == '__main__':
    main()