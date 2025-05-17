import numpy as np
from numpy._typing._array_like import NDArray

def build_sample_image() -> NDArray:
    """
    Create a sample binary image with two regions: an ellipse and a square.
    
    Returns:
        np.ndarray: A binary image with two regions.
    """
    # Create a simple binary image with a single square region.
    image = np.zeros((512, 512), dtype=bool)

    # Draw an ellipse centered at (256,256)
    cy, cx = 256, 256
    ry, rx = 40, 80  
    Y, X = np.ogrid[:512, :512]
    mask = ((Y - cy)**2) / ry**2 + ((X - cx)**2) / rx**2 <= 1
    image[mask] = True

    # Draw a small square region
    image[300:355, 275:400] = True

    return image.astype(np.uint8) * 255  # Convert to uint8 for visualization
