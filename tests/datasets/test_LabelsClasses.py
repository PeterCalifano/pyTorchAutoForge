import pathlib
import tempfile
import pytest
import yaml

from pyTorchAutoForge.datasets.LabelsClasses import (
    LabelsContainer,
    GeometricLabels,
    AuxiliaryLabels,
    KptsHeatmapsLabels,
    PTAF_Datakey,
    Parse_ptaf_datakeys,
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


    container_dict = container.to_dict()
    print("Container as dict:", container_dict)

    # Convert back to LabelsContainer from dict
    new_container = LabelsContainer.from_dict(container_dict)
    print(new_container)

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


def test_parse_ptaf_datakeys_valid_input():
    """Test Parse_ptaf_datakeys with valid label strings."""
    # Test single valid label
    result = Parse_ptaf_datakeys(["IMAGE"])
    assert result == (PTAF_Datakey.IMAGE,)
    
    # Test multiple valid labels
    result = Parse_ptaf_datakeys(["IMAGE", "MASK", "BBOX"])
    assert result == (PTAF_Datakey.IMAGE, PTAF_Datakey.MASK, PTAF_Datakey.BBOX)
    
    # Test all valid labels (dynamically generated from enum)
    all_labels = [key.name for key in PTAF_Datakey]
    result = Parse_ptaf_datakeys(all_labels)
    assert len(result) == len(all_labels)
    assert all(isinstance(key, PTAF_Datakey) for key in result)


def test_parse_ptaf_datakeys_case_insensitive():
    """Test Parse_ptaf_datakeys handles mixed case input correctly."""
    # Test lowercase
    result = Parse_ptaf_datakeys(["image", "mask"])
    assert result == (PTAF_Datakey.IMAGE, PTAF_Datakey.MASK)
    
    # Test mixed case
    result = Parse_ptaf_datakeys(["Image", "MaSk", "bBoX"])
    assert result == (PTAF_Datakey.IMAGE, PTAF_Datakey.MASK, PTAF_Datakey.BBOX)


def test_parse_ptaf_datakeys_empty_input():
    """Test Parse_ptaf_datakeys with empty input."""
    result = Parse_ptaf_datakeys([])
    assert result == ()


def test_parse_ptaf_datakeys_unknown_label():
    """Test Parse_ptaf_datakeys raises ValueError for unknown labels."""
    with pytest.raises(ValueError) as exc_info:
        Parse_ptaf_datakeys(["UNKNOWN_LABEL"])
    
    assert "Unknown PTAF_Datakey: UNKNOWN_LABEL" in str(exc_info.value)
    
    # Test that the original KeyError is chained
    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, KeyError)


def test_parse_ptaf_datakeys_partial_invalid():
    """Test Parse_ptaf_datakeys raises error when one label in sequence is invalid."""
    with pytest.raises(ValueError) as exc_info:
        Parse_ptaf_datakeys(["IMAGE", "INVALID", "MASK"])
    
    assert "Unknown PTAF_Datakey: INVALID" in str(exc_info.value)


if __name__ == "__main__":
    #test_labels_containers_instantiation()
    #test_lbl_containers_loading_from_yaml()
    test_to_dict_and_from_dict()
