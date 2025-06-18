# TODO
import argparse as argp
import os, sys
import logging

try:
    import torch
    from pyTorchAutoForge.evaluation import ModelProfilerHelper
    from pyTorchAutoForge.api.torch import LoadModel, SaveModel, AutoForgeModuleSaveMode
    from torch.profiler import profile, record_function, ProfilerActivity
    from pyTorchAutoForge.utils.argument_parsers import ParseShapeString, ConfigArgumentParser
    import onnx
    from onnx2pytorch import ConvertModel as OnnxToTorch
except ImportError as e:
    print("Required libraries are not installed. Please install pyTorchAutoForge and its dependencies.")
    sys.exit(1)
    
# Set up logging # TODO use logger to print info
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# CLI callable script to load a model, profile it and export to onnx if requested.
# TODO add support to load config from yml, using ConfigArgumentParser in PTAF
argparser = argp.ArgumentParser(
    description="Program to profile a PyTorch/ONNx model and optionally export it to ONNX format and/or torch dynamo traced model.")

argparser.add_argument(
    '-l', '--library_add',
    type=str,
    nargs='+',
    default=None,
    help="One or more paths to custom libraries to add to sys.path"
)

argparser.add_argument(
    '-i', '--import_module',
    type=str,
    nargs='+',
    default=None,
    help="One or more import statements to execute before running the script"
)

argparser.add_argument(
    "--model_path",
    "-p",
    type=str,
    required=True,
    help="Path to the PyTorch/ONNx model file (.pt, .pth, .onnx).")

argparser.add_argument(
    "--onnx_export",
    "-o",
    action='store_true',
    help="Export the model to ONNX format.")

argparser.add_argument(
    "--traced_export",
    "-t",
    action='store_true',
    help="Export the model as a traced script using torch dynamo API.")

argparser.add_argument(
    "--shape_input_sample",
    "-s",
    type=ParseShapeString,
    required=True,
    help="Input tensor shape, comma-separated, e.g. --shape 1,3,224,224")

argparser.add_argument("--device",
                    "-d",
                    type=str,
                    default="cpu",
                    help="Device to run the model on (default: 'cpu').")

argparser.add_argument(
    "--netron",
    "-n",
    action='store_true',
    help="Open the model in Netron for visualization after profiling and export.")

# %% Main program    
def main():

    # Parse command line arguments
    args = argparser.parse_args()

    # If a custom library is specified, add it to the system path
    if args.library_add:
        for lib_path in args.library_add:
            if lib_path not in sys.path:
                sys.path.insert(0, lib_path)
        logging.info(f"Added library paths: {args.library_add}")

    # If import_module is specified, import the module
    if args.import_module:
        import importlib

        for class_path in args.import_module:
            module_name, class_name = class_path.rsplit('.', 1)
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            # Inject into globals so Unpickler can find it
            globals()[class_name] = cls
            print(f"Registered {class_name} from {module_name} for pickle.")


    # %% Output paths definitions
    # Get paths from CLI input
    root_dir_path = os.path.dirname(p=args.model_path)
    model_filename = os.path.basename(p=args.model_path)
    model_name_noext = os.path.splitext(model_filename)[0]

    if not os.path.exists(root_dir_path):
        logging.error(f"Error: specified model path '{args.model_path}' does not exist.")
        sys.exit(1)

    export_output_path = os.path.join(root_dir_path, "exported_models")
    traced_output_path = os.path.join(export_output_path, model_name_noext)
    onnx_output_path = os.path.join(export_output_path, model_name_noext)

    # Imports and overrides
    try:
        from pyTorchAutoForge.api.onnx import ModelHandlerONNx
    except ImportError as e:
        print("ONNX support in pyTorchAutoForge.api.onnx is not available. Export will be disabled.")
        args.onnx_export = False

    if not(torch.cuda.is_available()) and args.device.startswith('cuda'):
        print("CUDA is not available on this system. Switching to CPU.")
        args.device = 'cpu'

    load_as_traced = False
    if args.model_path.endswith('.pt'):
        load_as_traced = True

    # Determine input sample shape
    input_sample_shape = args.shape_input_sample
    input_sample = torch.randn(input_sample_shape)
    
    # Options recap and printing
    print("\033[96;1m\033[1m\n-------- Model profiling and export program options --------\033[0m")
    print(f"\033[96;1m\033[1mModel path:\033[0m {args.model_path}")
    print(
        f"\033[96;1m\033[1m\033[96;1m\033[1mONNX export:\033[0m {args.onnx_export}")
    print(f"\033[96;1m\033[1mTraced export:\033[0m {args.traced_export}")
    print(f"\033[96;1m\033[1mInput sample shape:\033[0m {input_sample_shape}")
    print(f"\033[96;1m\033[1mDevice:\033[0m {args.device}")
    print(f"\033[96;1m\033[1mExport path:\033[0m {onnx_output_path}")
    print(f"\033[96;1m\033[1mNetron visualization:\033[0m {args.netron}\n\n")
    
    # Check if model exist
    if not os.path.exists(args.model_path):
        print("\033[91m" + f"Model file '{args.model_path}' does not exist. Please provide a valid model path." + "\033[0m")
        sys.exit(1)

    # %% Model profiling
    # Load model
    if args.model_path.endswith('.onnx'):
        onnx_model = onnx.load(args.model_path)

        # Convert model to torch using onnx2pytorch package
        model = OnnxToTorch(onnx_model)

    elif args.model_path.endswith('.pth') or args.model_path.endswith('.pt'):

        # Get model from file
        model = LoadModel(model=None,
                          model_filename=args.model_path,
                          load_as_traced=load_as_traced)
    else:
        raise ValueError(
            f"\033[91mUnsupported model file format: {args.model_path}. Supported formats are .pt, .pth, and .onnx.\033[0m")
    model.eval()
    print("\t--> Model loaded from: ", args.model_path)

    # Determine activities to profile
    profile_activities = []
    profile_activities.append(ProfilerActivity.CPU)

    if torch.cuda.is_available() and args.device.startswith('cuda'):
        profile_activities.append(ProfilerActivity.CUDA)

    # Run model profiler
    model_profiler = ModelProfilerHelper(model=model,
                                         input_shape_or_sample=input_sample,
                                         device=args.device,
                                         activities=tuple(profile_activities))

    obj_profile = model_profiler.run_prof()
    model_summary = model_profiler.make_summary()

    # %% Model tracing and export to ONNX
    try: 
        if not load_as_traced and args.traced_export:
            logging.info("Exporting model as TRACED_DINAMO...")
            SaveModel(model=model,
                    model_filename=traced_output_path,
                    save_mode=AutoForgeModuleSaveMode.TRACED_DINAMO,
                    example_input=input_sample)
            logging.info(f"Model traced and saved to: {traced_output_path}.pt")
        else:   
            logging.info("Model was loaded as a traced model. Skipping tracing...")

    except Exception as e:
        logging.error(f"Error during model tracing: {e}")
        load_as_traced = False


    onnx_filepath = None
    try:
        if args.onnx_export and not args.model_path.endswith('.onnx') and args.onnx_export:
            # TODO: try to add onnx simply 

            logging.info("Exporting model to ONNX format...")
            onnx_handler = ModelHandlerONNx(model=model,
                                            dummy_input_sample=input_sample,
                                            opset_version=11,
                                            onnx_export_path=onnx_output_path)
            
            # FIXME: adaptive pooling seems to have export issues and torch_dynamo_export does not work
            #onnx_filepath = onnx_handler.torch_export()
            onnx_filepath = onnx_handler.torch_dynamo_export()
            
            model_exported_to_onnx = True
            logging.info(f"Model exported to ONNX format at: {onnx_filepath}")
        else:
            model_exported_to_onnx = False

    except Exception as e:
        print("\033[91m" + f"Error during ONNX export: {e}" + "\033[0m")
        onnx_filepath = None
        model_exported_to_onnx = False

    # %% Open netron diagram
    if args.netron:
        try:
            import netron
            logging.info("Generating and opening Netron diagram...")
            if model_exported_to_onnx and onnx_filepath is not None:
                netron_filepath = onnx_filepath

            elif load_as_traced or args.traced_export:
                netron_filepath = traced_output_path + ".pt"

            else:
                raise ValueError(
                    f"\033[91mNo model available for Netron visualization. Please export the model to ONNX or trace it.\033[0m")

            model_profiler.make_netron_diagram(model_path=netron_filepath)
        
        except ImportError:
            print("\033[91m" + "Netron is not installed. Please install it to visualize the model." + "\033[0m")



if __name__ == "__main__":
    main()