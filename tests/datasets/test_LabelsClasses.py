import pathlib
import tempfile
import pytest
import yaml

from pyTorchAutoForge.datasets.LabelsClasses import (
    LabelsContainer,
    GeometricLabels,
    AuxiliaryLabels,
    KptsHeatmapsLabels,
)

def test_to_dict_and_from_dict():
    # Create a container with custom values
    geom = GeometricLabels(
        ui32_image_size=(1296, 966),
        centre_of_figure=(522.6458, 590.4096),
        distance_to_obj_centre=1.31744e8,
        length_units='m',
        bound_box_coordinates=(433.0, 501.0, 145.0, 170.0),
        bbox_coords_order='xywh',
        obj_apparent_size_in_pix=88.1364,
        object_reference_size=1737420.0,
        object_ref_size_units='m',
        obj_projected_ellipsoid_matrix=[[0.0, 0.0, 0.0]]
    )
    aux = AuxiliaryLabels(
        phase_angle_in_deg=73.3925,
        light_direction_rad_angle_from_x=-2.60326,
        object_shape_matrix_cam_frame=[[0.0, 0.0, 0.0]]
    )
    kpts = KptsHeatmapsLabels(
        num_of_kpts=5,
        heatmap_size=(64, 64),
        heatmap_datatype='single'
    )
    container = LabelsContainer(
        geometric=geom,
        auxiliary=aux,
        kpts_heatmaps=kpts
    )
    d = container.to_dict()
    new_container = LabelsContainer.from_dict(d)

    assert new_container.geometric.ui32_image_size == container.geometric.ui32_image_size
    assert new_container.auxiliary.phase_angle_in_deg == container.auxiliary.phase_angle_in_deg
    assert new_container.kpts_heatmaps.num_of_kpts == container.kpts_heatmaps.num_of_kpts

def test_yaml_serialization_and_deserialization(tmp_path):
    # Create a container with custom values
    geom = GeometricLabels(
        ui32_image_size=(1024, 768),
        centre_of_figure=(512.0, 384.0),
        distance_to_obj_centre=100.0,
        length_units='cm',
        bound_box_coordinates=(100.0, 150.0, 200.0, 250.0),
        bbox_coords_order='xywh',
        obj_apparent_size_in_pix=50.0,
        object_reference_size=300.0,
        object_ref_size_units='cm',
        obj_projected_ellipsoid_matrix=[[1.0, 0.0, 0.0]]
    )
    aux = AuxiliaryLabels(
        phase_angle_in_deg=45.0,
        light_direction_rad_angle_from_x=0.785,
        object_shape_matrix_cam_frame=[[1.0, 0.0, 0.0]]
    )
    kpts = KptsHeatmapsLabels(
        num_of_kpts=8,
        heatmap_size=(32, 32),
        heatmap_datatype='double'
    )
    container = LabelsContainer(
        geometric=geom,
        auxiliary=aux,
        kpts_heatmaps=kpts
    )
    # Save container to YAML file using tmp_path
    file_path = tmp_path / "test.yaml"
    container.save_to_yaml(str(file_path))
    loaded_container = LabelsContainer.load_from_yaml(str(file_path))

    assert loaded_container.geometric.ui32_image_size == container.geometric.ui32_image_size
    assert loaded_container.auxiliary.phase_angle_in_deg == container.auxiliary.phase_angle_in_deg
    assert loaded_container.kpts_heatmaps.num_of_kpts == container.kpts_heatmaps.num_of_kpts

    # Delete temporary file
    if file_path.exists():
        file_path.unlink()

def test_labels_containers_instantiation() -> None:
    # Create a container with custom values
    geom = GeometricLabels(
        ui32_image_size=(1296, 966),
        centre_of_figure=(522.6458, 590.4096),
        distance_to_obj_centre=1.31744e8,
        length_units='m',
        bound_box_coordinates=(433.0, 501.0, 145.0, 170.0),
        bbox_coords_order='xywh',
        obj_apparent_size_in_pix=88.1364,
        object_reference_size=1737420.0,
        object_ref_size_units='m',
        obj_projected_ellipsoid_matrix=[[0.0]*3 for _ in range(3)]
    )
    aux = AuxiliaryLabels(
        phase_angle_in_deg=73.3925,
        light_direction_rad_angle_from_x=-2.60326,
        object_shape_matrix_cam_frame=[[0.0]*3 for _ in range(3)]
    )
    kpts = KptsHeatmapsLabels()
    container = LabelsContainer(
        geometric=geom, auxiliary=aux, kpts_heatmaps=kpts)

    # Serialize to YAML string
    yaml_str = container.to_yaml()
    print("YAML Output:\n", yaml_str)


def test_lbl_containers_loading_from_yaml():
    # Test sample here: "/tests/.test_samples/test_labels/000001.yaml"

    this_file_path = pathlib.Path(__file__).resolve().parent
    root_path = this_file_path / pathlib.Path("../.test_samples")
    lbl_number = 1
    lbl_path = "labels"
    lbl_filename = f"{lbl_number:06d}.yaml"

    filepath = pathlib.Path(root_path) / lbl_path / lbl_filename

    # Load from YAML file
    container = LabelsContainer.load_from_yaml(filepath)

    import pprint
    print("Loaded Container:")
    pprint.pprint(container, indent=2)


if __name__ == "__main__":
    test_labels_containers_instantiation()
    test_lbl_containers_loading_from_yaml()
