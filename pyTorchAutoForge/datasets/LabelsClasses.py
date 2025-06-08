from dataclasses import dataclass, field, fields
from typing import Any, Type, TypeVar
import yaml
import pathlib

T = TypeVar('T', bound='BaseLabelsContainer')

@dataclass
class BaseLabelsContainer:
    """
    Base container offering YAML serialization/deserialization with support for
    mapping between YAML keys and dataclass attributes via metadata aliases.
    """

    def to_yaml(self) -> str:
        """
        Serialize the container to a YAML string including only non-empty fields,
        using metadata aliases for keys.
        """
        def prune(data: Any) -> Any:
            if isinstance(data, BaseLabelsContainer):
                return prune({f.metadata.get('yaml', f.name): getattr(data, f.name)
                              for f in fields(data)
                              if getattr(data, f.name) not in (None, '', [], {}, ())})
            if isinstance(data, dict):
                result = {}
                for k, v in data.items():
                    pruned_v = prune(v)
                    if pruned_v not in (None, '', [], {}, ()):
                        result[k] = pruned_v
                return result
            if isinstance(data, (list, tuple)):
                pruned_list = [prune(v) for v in data]
                return pruned_list if pruned_list else None
            return data

        pruned = prune(self)
        return yaml.safe_dump(pruned)

    @classmethod
    def from_dict(cls: Type[T], data: dict[str, Any]) -> T:
        """
        Instantiate a container from a dict using metadata aliases,
        recursively constructing nested BaseLabelsContainer types.
        """
        init_kwargs: dict[str, Any] = {}
        for f in fields(cls):
            alias = f.metadata.get('yaml', f.name)
            if alias in data:
                value = data[alias]
                if hasattr(f.type, 'from_dict') and isinstance(value, dict):
                    init_kwargs[f.name] = f.type.from_dict(
                        value)  # type: ignore
                else:
                    init_kwargs[f.name] = value
        return cls(**init_kwargs)  # type: ignore

    @classmethod
    def load_from_yaml(cls: Type[T], path: str | pathlib.Path) -> T:
        """
        Load a container instance from a YAML file, mapping YAML keys to attributes.
        """
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def save_to_yaml(self, path: str) -> None:
        """
        Save the container to a YAML file.
        """
        with open(path, 'w') as f:
            f.write(self.to_yaml())


@dataclass
class GeometricLabels(BaseLabelsContainer):

    ui32_image_size: tuple[int, int] = field(
        default=(0, 0), metadata={'yaml': 'ui32ImageSize'})
    
    centre_of_figure: tuple[float, float] = field(
        default=(0.0, 0.0), metadata={'yaml': 'dCentreOfFigure'})
    
    distance_to_obj_centre: float = field(
        default=0.0, metadata={'yaml': 'dDistanceToObjCentre'})
    
    length_units: str = field(default='', metadata={'yaml': 'charLengthUnits'})

    bound_box_coordinates: tuple[float, float, float, float] = field(
        default=(0.0, 0.0, 0.0, 0.0), metadata={'yaml': 'dBoundBoxCoordinates'})
    
    bbox_coords_order: str = field(default='xywh', metadata={
                                   'yaml': 'charBBoxCoordsOrder'})
    
    obj_apparent_size_in_pix: float = field(
        default=0.0, metadata={'yaml': 'dObjApparentSizeInPix'})
    
    object_reference_size: float = field(
        default=0.0, metadata={'yaml': 'dObjectReferenceSize'})
    
    object_ref_size_units: str = field(
        default='m', metadata={'yaml': 'dObjectRefSizeUnits'})
    
    obj_projected_ellipsoid_matrix: list[list[float]] = field(default_factory=list,
                                                              metadata={'yaml': 'dObjProjectedEllipsoidMatrix'})


@dataclass
class AuxiliaryLabels(BaseLabelsContainer):
    phase_angle_in_deg: float = field(
        default=-1.0, metadata={'yaml': 'dPhaseAngleInDeg'})
    
    light_direction_rad_angle_from_x: float = field(
        default=0.0, metadata={'yaml': 'dLightDirectionRadAngleFromX'})
    
    object_shape_matrix_cam_frame: list[list[float]] = field(
        default_factory=list, metadata={'yaml': 'dObjectShapeMatrix_CamFrame'})


@dataclass
class KptsHeatmapsLabels(BaseLabelsContainer):
    num_of_kpts: int = field(default=0, metadata={'yaml': 'ui32NumOfKpts'})

    heatmap_size: tuple[int, int] = field(
        default=(0, 0), metadata={'yaml': 'ui32HeatmapSize'})
    
    heatmap_datatype: str = field(default='single', metadata={
                                  'yaml': 'charHeatMapDatatype'})


# %% Container object to group all labels
@dataclass
class LabelsContainer(BaseLabelsContainer):
    geometric: GeometricLabels = field(default_factory=GeometricLabels,
                                       metadata={'yaml': 'geometric'})
    auxiliary: AuxiliaryLabels = field(default_factory=AuxiliaryLabels,
                                       metadata={'yaml': 'auxiliary'})
    kpts_heatmaps: KptsHeatmapsLabels = field(default_factory=KptsHeatmapsLabels,
                                              metadata={'yaml': 'kpts_heatmaps'})

# Runnable example
def example() -> None:
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


def example_loading():

    root_path = "/media/peterc/Main/linux_data/datasets/UniformlyScatteredPointCloudsDatasets/Moon/Dataset_UniformAzElPointCloud_ABRAM_Moon_5000_ID5"
    lbl_number = 1
    lbl_path = "labels"
    lbl_filename = f"{lbl_number:06d}.yaml"
    
    filepath = pathlib.Path(root_path) / lbl_path / lbl_filename

    # Load from YAML file
    container = LabelsContainer.load_from_yaml(filepath)

    print("Loaded Container:")
    print(container)

if __name__ == "__main__":
    example()
    example_loading()