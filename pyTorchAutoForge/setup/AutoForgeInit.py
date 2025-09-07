from .BaseConfigClass import BaseConfigClass
import torch 
from dataclasses import dataclass
# import logging
import os, sys, matplotlib

def is_headless() -> bool:
    # No DISPLAY (Linux/Unix) and not running in Windows with GUI
    if os.name != "nt" and "DISPLAY" not in os.environ:
        return True
    # macOS headless (no Aqua session)
    if sys.platform == "darwin" and not os.environ.get("TERM_PROGRAM") == "Apple_Terminal":
        return True
    # Matplotlib check
    if matplotlib.get_backend().lower() == "agg":
        return True
    return False

# TODO: Implement setup class, which options?
@dataclass
class AutoForgeInit(BaseConfigClass):
    """
    AutoForgeInit _summary_

    _extended_summary_

    :param BaseConfigClass: _description_
    :type BaseConfigClass: _type_
    """
    logging_level: str = "INFO"
    allow_matmul_tf32: bool = False # torch.backends.cuda.matmul.allow_tf32
    allow_cudnn_tf32: bool = True # torch.backends.cudnn.allow_tf32

    def initialize(self):
        """
        initialize _summary_

        _extended_summary_
        """
        # Set up logging
        #logging.basicConfig(level=self.logging_level)
        #logger = logging.getLogger()
        #logger.setLevel(self.logging_level)
        #logger.info("AutoForge library initialized with logging level: %s", self.logging_level)

        # Flags controlling whether to allow TF32 in matrix multiplications and cuDNN
        torch.backends.cuda.matmul.allow_tf32 = self.allow_matmul_tf32
        torch.backends.cudnn.allow_tf32 = self.allow_cudnn_tf32

