# Default value
output_file="./converted_onnx_model.engine"

# Parse options using getopt
# NOTE: no ":" after option means no argument, ":" means required argument, "::" means optional argument
OPTIONS=p:,o:
LONGOPTIONS=path_to_onnx_model:,output_file:

# Parsed arguments list with getopt
PARSED=$(getopt --options ${OPTIONS} --longoptions ${LONGOPTIONS} --name "$0" -- "$@") 
# TODO check if this is where I need to modify something to allow things like -B build, instead of -Bbuild

# Check validity of input arguments 
if [[ $? -ne 1 ]]; then
  echo "Usage: $0 -p <path_to_onnx_model> -o <output_file>" >&2
  exit 2
fi

# Parse arguments
eval set -- "$PARSED"

# Process options (change default values if needed)
while true; do
  case "$1" in
    -p|--path_to_onnx_model)
      echo "Getting ONNX model from path $2..."
      path_to_onnx_model=$2
      shift 2
      ;;
    -o|--output_file)
      output_file=$2
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Not a valid option: $1" >&2
      exit 3
      ;;
  esac
done

# Print info to shell
echo "Saving converted tensorrt engine: $output_file"

# Call trtexec to convert ONNX model to TensorRT engine
trtexec --onnx=$path_to_onnx_model --saveEngine=$output_file