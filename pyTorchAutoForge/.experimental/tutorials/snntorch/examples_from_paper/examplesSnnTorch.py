# Script created by PeterC to reproduce examples in reference paper, 06-06-2024
# REFERENCE: 

import torch 
import snntorch as snn
from snntorch import spikegen


# %% LIF neuron step solution (Equations 4-5)
def LIFneuronAnalyticSol(X, U, W, beta=0.9, thr=1):
    S = 0 # Initialization of output spike
    # Analytical time varying solution
    U = beta*U + W*X - S*thr # Single timestep
    S = int(S > thr) # Determine sif spiked

    return S, U




# Main function
def main():
    print('TODO')

    # %% Example instantiation of neurons in snntorch
    LIF_neuron = snn.Leaky(beta=0.9)
    IF_neuron = snn.Leaky(beta=1.0) # This causes the leaky term to vanish

    CUBA_neuron = snn.Synaptic(beta=0.9, alpha=0.8)
    RecLIFv1_neuron= snn.RLeaky(beta=0.9, all_to_all=True) # all-to-all recurrent lif neuron  
    RecLIFv2_neuron = snn.RLeaky(beta=0.9, all_to_all=False) # one-to-one recurrent lif neuron
    SpikingLSTM_neuron = snn.SLSTM(input_size=10, hidden_size=1000) # spiking LSTM: 10 inputs, 1000 outputs

    # %% Example of use of spikegen module for INPUT ENCODING
    inputSignal = torch.rand(100)
    numOfTimesteps = 5

    timeEncodedInput = spikegen.latency(inputSignal, numOfTimesteps)
    rateEncodedInput = spikegen.rate(inputSignal, numOfTimesteps)
    deltaModEncodedInput = spikegen.delta(inputSignal, numOfTimesteps)

    # %% Example of rate based loss functions
    from snntorch import functional as SF
    SF.ce_rate_loss() # cross-entropy spike rate
    SF.mse_rate_loss() # mean square spike rate
    SF.ce_max_membrane_loss() # maximum membrane
    SF.mse_membrane_loss() # mean square membrane

    # %% Example of latency based loss functions
    SF.ce_temporal_loss()
    SF.mse_temporal_loss()
    SF.mse_membrane_loss()

# Run main
if __name__=='__main__':
    main()

