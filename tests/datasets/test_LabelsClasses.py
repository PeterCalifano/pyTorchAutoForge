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
    yaml_dict = yaml.safe_load(yaml_str)
    assert "geometric" in yaml_dict
    assert "auxiliary" in yaml_dict
    assert "kpts_heatmaps" in yaml_dict
    assert yaml_dict["geometric"]["ui32ImageSize"] == [1296, 966]
    assert yaml_dict["geometric"]["charBBoxCoordsOrder"] == "xywh"


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

    assert container.geometric.ui32_image_size == (1296, 966)
    assert container.geometric.bbox_coords_order == "xywh"
    assert container.geometric.centre_of_figure == pytest.approx(
        (522.6458119377869, 590.409585241252)
    )
    assert container.geometric.distance_to_obj_centre == pytest.approx(1.3174400439758018e8)
    assert container.auxiliary.phase_angle_in_deg == pytest.approx(73.39248871205007)


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


def test_to_yaml_uses_yaml_aliases():
    geom = GeometricLabels(
        ui32_image_size=(10, 20),
        centre_of_figure=(1.0, 2.0),
        distance_to_obj_centre=3.0,
        length_units="m",
        bound_box_coordinates=(4.0, 5.0, 6.0, 7.0),
        bbox_coords_order="xywh",
        obj_apparent_size_in_pix=8.0,
        object_reference_size=9.0,
        object_ref_size_units="m",
        obj_projected_ellipsoid_matrix=[[1.0, 0.0, 0.0]],
    )
    aux = AuxiliaryLabels(
        phase_angle_in_deg=15.0,
        light_direction_rad_angle_from_x=0.5,
        object_shape_matrix_cam_frame=[[1.0, 0.0, 0.0]],
    )
    kpts = KptsHeatmapsLabels(
        num_of_kpts=3,
        heatmap_size=(32, 32),
        heatmap_datatype="single",
    )
    container = LabelsContainer(geometric=geom, auxiliary=aux, kpts_heatmaps=kpts)

    yaml_dict = yaml.safe_load(container.to_yaml())
    assert "geometric" in yaml_dict
    assert "ui32ImageSize" in yaml_dict["geometric"]
    assert "dCentreOfFigure" in yaml_dict["geometric"]
    assert "auxiliary" in yaml_dict
    assert "dPhaseAngleInDeg" in yaml_dict["auxiliary"]
    assert "kpts_heatmaps" in yaml_dict
    assert "ui32HeatmapSize" in yaml_dict["kpts_heatmaps"]


def test_from_dict_with_yaml_aliases_converts_types():
    data = {
        "geometric": {
            "ui32ImageSize": [10, 20],
            "dCentreOfFigure": [1.0, 2.0],
            "dDistanceToObjCentre": "3.5",
            "charLengthUnits": "m",
            "dBoundBoxCoordinates": [4.0, 5.0, 6.0, 7.0],
            "charBBoxCoordsOrder": "xywh",
            "dObjApparentSizeInPix": "8.25",
            "dObjectReferenceSize": "9.0",
            "dObjectRefSizeUnits": "m",
            "dObjProjectedEllipsoidMatrix": [[1.0, 0.0, 0.0]],
        },
        "auxiliary": {
            "dPhaseAngleInDeg": "10.0",
            "dLightDirectionRadAngleFromX": "0.5",
            "dObjectShapeMatrix_CamFrame": [[1.0, 0.0, 0.0]],
        },
        "kpts_heatmaps": {
            "ui32NumOfKpts": 2,
            "ui32HeatmapSize": [16, 24],
            "charHeatMapDatatype": "double",
        },
    }

    container = LabelsContainer.from_dict(data, yaml_aliases=True)

    assert container.geometric.ui32_image_size == (10, 20)
    assert isinstance(container.geometric.ui32_image_size, tuple)
    assert container.geometric.distance_to_obj_centre == 3.5
    assert isinstance(container.geometric.distance_to_obj_centre, float)
    assert container.geometric.bound_box_coordinates == (4.0, 5.0, 6.0, 7.0)
    assert container.kpts_heatmaps.heatmap_size == (16, 24)


def test_bbox_order_getattr_guard():
    geom = GeometricLabels(
        bound_box_coordinates=(1.0, 2.0, 3.0, 4.0),
        bbox_coords_order="xywh",
    )
    container = LabelsContainer(geometric=geom)

    assert getattr(container, "BBOX_XYWH") == geom.bound_box_coordinates
    with pytest.raises(AssertionError):
        _ = getattr(container, "BBOX_XYXY")


def test_get_lbl_1d_vector_size():
    size, sizes_dict = LabelsContainer.get_lbl_1d_vector_size(
        (PTAF_Datakey.CENTRE_OF_FIGURE, PTAF_Datakey.BBOX_XYWH)
    )
    assert size == 6
    assert sizes_dict == {"CENTRE_OF_FIGURE": 2, "BBOX_XYWH": 4}


def test_get_labels_with_datakeys():
    geom = GeometricLabels(
        centre_of_figure=(1.0, 2.0),
        distance_to_obj_centre=3.0,
        bound_box_coordinates=(4.0, 5.0, 6.0, 7.0),
    )
    container = LabelsContainer(geometric=geom)

    labels = container.get_labels(
        (PTAF_Datakey.CENTRE_OF_FIGURE, PTAF_Datakey.RANGE_TO_COM)
    )
    assert labels.tolist() == [1.0, 2.0, 3.0]


if __name__ == "__main__":
    #test_labels_containers_instantiation()
    #test_lbl_containers_loading_from_yaml()
    test_to_dict_and_from_dict()
    pass
