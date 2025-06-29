from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

def Project_PCA(data: np.ndarray, n_components: int) -> np.ndarray:
    """
    Projects a dataset onto the first 'n_components' principal components.

    Parameters:
        data (np.ndarray): Input data of shape (N_samples, N_feats).
        n_components (int): Number of principal components to keep (default=6).

    Returns:
        np.ndarray: Transformed data of shape (N_samples, n_components).
    """
    assert len(
        data.shape) == 2, "Input data must be a 2D array (N_samples, N_feats)"

    # Standardize data to have zero mean and unit variance
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Apply PCA
    pca = PCA(n_components=n_components)
    projected_data = pca.fit_transform(data_scaled)

    return projected_data
