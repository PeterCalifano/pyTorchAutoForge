from dataclasses import dataclass, field, fields
from typing import Any, Type, TypeVar
import yaml
import pathlib
from typing import get_origin, get_args

T = TypeVar('T', bound='BaseLabelsContainer')

@dataclass
class BaseLabelsContainer:
    """
    Base container offering YAML serialization/deserialization with support for
    mapping between YAML keys and dataclass attributes via metadata aliases.
    """

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the container and all nested BaseLabelsContainer fields
        into a plain dict using attribute names.
        """

        out_dict: dict[str, Any] = {}
        
        for f in fields(self):

            value = getattr(self, f.name)

            if isinstance(value, BaseLabelsContainer):
                out_dict[f.name] = value.to_dict()
            else:
                out_dict[f.name] = value

        return out_dict
    
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
    def from_dict(cls: Type[T], data: dict[str, Any], yaml_aliases: bool = False) -> T:
        """
        Instantiate a container from a dict using metadata aliases,
        recursively constructing nested BaseLabelsContainer types.
        """
        init_kwargs: dict[str, Any] = {}

        for f in fields(cls):
            if yaml_aliases:
                alias = f.metadata.get('yaml', f.name)

                if alias in data:
                    raw_value = data[alias]

                    # Nested container
                    if hasattr(f.type, 'from_dict') and isinstance(raw_value, dict):
                        value = f.type.from_dict(raw_value, yaml_aliases=yaml_aliases)  # type: ignore
                    else:
                        value = raw_value
                        # Coerce lists to tuples if field annotation is Tuple
                        origin = get_origin(f.type)
                        if origin is tuple and isinstance(value, list):
                            value = tuple(value)

            else:
                value = data.get(f.name, None)

                # Nested container
                if hasattr(f.type, 'from_dict') and isinstance(value, dict):
                    value = f.type.from_dict(value, yaml_aliases=yaml_aliases)  # type: ignore

            init_kwargs[f.name] = value

        return cls(**init_kwargs)  # type: ignore

    @classmethod
    def load_from_yaml(cls: Type[T], path: str | pathlib.Path) -> T:
        """
        Load a container instance from a YAML file, mapping YAML keys to attributes.
        """
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data, yaml_aliases=True)

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

    # Convenience getters for common fields:
    @property
    def centre_of_figure(self) -> tuple[float, float]:
        return self.geometric.centre_of_figure

    @property
    def distance_to_obj_centre(self) -> float:
        return self.geometric.distance_to_obj_centre

    @property
    def ui32_image_size(self) -> tuple[int, int]:
        return self.geometric.ui32_image_size

    @property
    def bound_box_coordinates(self) -> tuple[float, float, float, float]:
        return self.geometric.bound_box_coordinates

    @property
    def obj_apparent_size_in_pix(self) -> float:
        return self.geometric.obj_apparent_size_in_pix

    @property
    def obj_projected_ellipsoid_matrix(self) -> list[list[float]]:
        return self.geometric.obj_projected_ellipsoid_matrix

    @property
    def light_direction_rad_angle_from_x(self) -> float:
        return self.auxiliary.light_direction_rad_angle_from_x

    @property
    def phase_angle_in_deg(self) -> float:
        return self.auxiliary.phase_angle_in_deg
