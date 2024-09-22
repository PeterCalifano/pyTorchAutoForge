# TODO: Which operations are required to setup the package environment?
import logging
import torch 

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later. Note that this increases speed (even 1 order of magnitude), but reduces numerical precision. 
# Relative error compared to double precision is approximately 2 orders of magnitude larger.
torch.backends.cuda.matmul.allow_tf32 = False

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True
