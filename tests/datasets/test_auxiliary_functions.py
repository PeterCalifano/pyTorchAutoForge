from pyTorchAutoForge.datasets.auxiliary_functions import LoadDatasetToMem
from pyTorchAutoForge.datasets.LabelsClasses import PTAF_Datakey

import os
import numpy as np

def test_LoadDatasetToMem():

    import matplotlib
    import matplotlib.pyplot as plt

    hostname = os.uname()[1]
    if hostname == "lagrange" or os.getenv("TMUX") is not None or not plt.isinteractive():
        matplotlib.use('agg')  # or 'Qt5Agg'
    else:
        if matplotlib.is_interactive():
            matplotlib.use('TkAgg')  # or 'TkAgg'
            plt.ion()
        else:
            matplotlib.use(backend='Agg')

    env_variable = os.getenv("DATASETS")

    if env_variable is None:
        raise ValueError(
            "Environment variable SCRATCH or DATASETS not set. Please set it to the path of the datasets folder.")

    dataset_root = (os.path.join(env_variable, 'OPERATIVE-archive'),
                    os.path.join(env_variable, 'UniformlyScatteredPointCloudsDatasets/Ceres'))

    dataset_names = ("Dataset_UniformAzElPointCloud_ABRAM_Ceres_420_ID0",)
    dataset_origin_tag = ('dataset_gen_lib_abram',)

    data_keys = (PTAF_Datakey.CENTRE_OF_FIGURE,
                 PTAF_Datakey.RANGE_TO_COM,
                 PTAF_Datakey.BBOX_XYWH,
                 PTAF_Datakey.PHASE_ANGLE,
                 PTAF_Datakey.REFERENCE_SIZE)

    data_container = LoadDatasetToMem(dataset_name=dataset_names,
                                      datasets_root_folder=dataset_root,
                                      dataset_origin_tag=dataset_origin_tag,
                                      lbl_vector_data_keys=data_keys)

    # Loop over images and get the ten images with lowest median intensity
    num_imgs = data_container.images.shape[0]
    num_worst = np.min((10, num_imgs))

    # Compute median intensity for each image
    stats = []
    for img, lbl in data_container:
        nonzero_pixels = img[img > 0]
        med = np.median(nonzero_pixels) if len(nonzero_pixels) > 0 else 0
        stats.append((med, img, lbl))

    # Sort by increasing median and take first n
    stats.sort(key=lambda x: x[0])
    lowest = stats[:num_worst]

    # Show images with lowest median intensity in a grid layout
    cols = min(num_worst, 5)
    rows = int(np.ceil(num_worst / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = axes.flatten() if num_worst > 1 else [axes]

    for ax, (med, img, lbl) in zip(axes, lowest):

        img_to_show = img if img.ndim == 3 else img.squeeze()
        ax.imshow(img_to_show, cmap='gray')
        ax.set_title(f"{lbl}\nmed={med:.1f}")
        ax.axis('off')

    for ax in axes[len(lowest):]:
        ax.axis('off')
    plt.tight_layout()
    plt.show()
