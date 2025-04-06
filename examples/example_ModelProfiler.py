import numpy as np
from pyTorchAutoForge.evaluation import ModelProfiler
import torch
import os 
from pyTorchAutoForge.api.torch import SaveModel

saveTraced = True
def main():

    raise NotImplementedError("This example is not working yet, need to fix the model loading and saving")

    # Load model from state dictionary
    home_dir = os.path.expanduser("~")
    checkpoints_path = f"{home_dir}/devDir/nav-frontend/.experimental/checkpoints/"
    state_name = "classy-skink-613_epoch_3888_cpu"
    #state_name = "funny-donkey-284_epoch_4959_cpu"

    nn_model_state = checkpoints_path + state_name + ".pth"

    output_prof_folder = f"{home_dir}/devDir/nav-frontend/.experimental/profiling_results/"
    os.makedirs(output_prof_folder, exist_ok=True)

    output_prof_filename = os.path.join(
        output_prof_folder, f"{state_name}_profiling_results.json")
    fc_layer_sizes = [128, 64, 8, 32, 16]
    disable_bn = True

    # Load model
    model = LoadShallowResNetSkipToOut(input_layer_size=14, fc_layer_sizes=fc_layer_sizes, output_layer_size=2, skip_indices=[1, 2], checkpoint_path=nn_model_state, disable_bn=disable_bn).to('cpu')

    ############ DEVTMP ############
    # Attempt to load model from dict --> need to save the model directly calling model.save()
    #model = torch.load(nn_model_state)
    ################################

    # Define profiler object
    profiler = ModelProfiler(model, input_shape_or_sample=(1, 14), device='cpu', record_shapes=True, with_stack=True, output_prof_filename=output_prof_filename)

    # Print model summary
    model_stats = profiler.make_summary()

    # Run profiler
    output_prof = profiler.run_prof()

    # Save traced and make netron diagram
    if saveTraced:
        # Save model as traced
        SaveModel(model, checkpoints_path + state_name,
                    save_as_traced=True, example_input = torch.rand(1, 14), target_device='cpu') 
        
        profiler.make_netron_diagram(checkpoints_path + state_name + ".pt")
    

if __name__ == "__main__":
    main()