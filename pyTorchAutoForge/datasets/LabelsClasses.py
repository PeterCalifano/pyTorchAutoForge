from dataclasses import dataclass, field, fields
from typing import Any, Type, TypeVar
import yaml
import pathlib
from typing import get_origin, get_args
import numpy as np
import enum

T = TypeVar('T', bound='BaseLabelsContainer')

try:
    from kornia.constants import DataKey
except ImportError:
    print('\033[93m' + "kornia not installed, images augmentation functionalities won't be available." + '\033[0m')

class PTAF_Datakey(enum.Enum):
    """
    Enumeration class for dataset keys. Interchangable with kornia datakeys (included in this enumeration class).
    """
    IMAGE = 0
    INPUT = 0
    MASK = 1
    BBOX = 2
    BBOX_XYXY = 3
    BBOX_XYWH = 4
    KEYPOINTS = 5
    CLASS = 6
    RANGE_TO_COM = 7  # Range to center of mass of object
    REFERENCE_SIZE = 8  # Reference size of the object, e.g. diameter or radius
    PHASE_ANGLE = 9  # Phase angle of the scene
    CENTRE_OF_FIGURE = 10  # Centre of figure of the object
    APPARENT_SIZE = 11  # Apparent size of the object in pixels

    def get_lbl_vector_size(self):
        # Define sizes for data keys
        """
        Get the size of the label vector based on the data key.
        """
        sizes = {
            PTAF_Datakey.IMAGE: -1,
            PTAF_Datakey.INPUT: -1,
            PTAF_Datakey.MASK: -1,
            PTAF_Datakey.BBOX: 4,  # x1, y1, x2, y2
            PTAF_Datakey.BBOX_XYXY: 4,  # x1, y1, x2, y2
            PTAF_Datakey.BBOX_XYWH: 4,  # x, y, width, height
            PTAF_Datakey.KEYPOINTS: 2,  # x, y for each keypoint
            PTAF_Datakey.CLASS: 1,
            PTAF_Datakey.RANGE_TO_COM: 1,
            PTAF_Datakey.REFERENCE_SIZE: 1,
            PTAF_Datakey.PHASE_ANGLE: 1,
            PTAF_Datakey.CENTRE_OF_FIGURE: 2,  # x, y coordinates of the centre of figure
            PTAF_Datakey.APPARENT_SIZE: 1  # Apparent size in pixels
        }

        return sizes.get(self, None)
    
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

                        elif ("float" in str(f.type) and not ("list" in str(f.type) or "tuple" in str(f.type))) and not isinstance(value, float):
                            # Check if value is convertible to float
                            try:
                                value = float(value)
                            except (TypeError, ValueError):
                                raise TypeError(f"Cannot convert value '{value}' of type {type(value)} to float for field '{f.name}'")

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

    # Dispatcher of property method from PTAF_Datakey enum:
    def __getattr__(self, item: str) -> Any:
        """
        Dispatches attribute access for PTAF_Datakey enum members to the appropriate
        label field within the container.

        Args:
            item (str): The PTAF_Datakey enum member or its string representation.

        Raises:
            AttributeError: If the provided item does not correspond to a supported PTAF_Datakey.

        Returns:
            Any: The value of the corresponding label field.
        """
        # Convert item to PTAF_Datakey if it's a string
        if isinstance(item, str):
            try:
                item = PTAF_Datakey[item]
            except KeyError:
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'.")
        
        if item == PTAF_Datakey.BBOX or item == PTAF_Datakey.BBOX_XYXY or item == PTAF_Datakey.BBOX_XYWH:
            
            # If bbox is specified with given coordinates, check it
            if item == PTAF_Datakey.BBOX_XYXY:
                assert(self.geometric.bbox_coords_order == 'xyxy'), \
                    "BBOX_XYXY requires bbox_coords_order to be 'xyxy'."
                
            elif item == PTAF_Datakey.BBOX_XYWH:
                assert(self.geometric.bbox_coords_order == 'xywh'), \
                    "BBOX_XYWH requires bbox_coords_order to be 'xywh'."
                
            return self.geometric.bound_box_coordinates
    
        elif item == PTAF_Datakey.CENTRE_OF_FIGURE:
            return self.geometric.centre_of_figure
        
        elif item == PTAF_Datakey.REFERENCE_SIZE:
            return float(self.geometric.object_reference_size)

        elif item == PTAF_Datakey.RANGE_TO_COM:
            return float(self.geometric.distance_to_obj_centre)

        elif item == PTAF_Datakey.PHASE_ANGLE:
            return float(self.auxiliary.phase_angle_in_deg)

        elif item == PTAF_Datakey.APPARENT_SIZE:
            return float(self.geometric.obj_apparent_size_in_pix)

        else:
            # Raise AttributeError to maintain standard behavior
            raise AttributeError(f"'{self.__class__.__name__}' object has no definition for PTAF_Datakey '{item}'. Make sure it exists or add it.")

    def get_labels(self, data_keys: tuple[PTAF_Datakey | str, ...] | str | PTAF_Datakey) -> list[Any]:
        """
        Get a list of label values corresponding to the provided data keys.
        
        Args:
            data_keys (tuple[PTAF_Datakey | str, ...]): The keys for which to retrieve label values.

        Returns:
            list[Any]: A list of label values corresponding to the provided keys.
        """

        if isinstance(data_keys, (str, PTAF_Datakey)):
            data_keys = (data_keys,) # Make tuple to ensure it is iterable

        return np.concatenate([np.array(getattr(self, str(key.name))).ravel() for key in data_keys])


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
    
    @classmethod
    def get_lbl_1d_vector_size(cls, data_keys: tuple[PTAF_Datakey | str, ...]):
        """
        Calculate the size of the label vector based on the provided data keys.
        """
        size = 0
        sizes_dict = {}
        for key in data_keys:
                        
            if key in (PTAF_Datakey.IMAGE, PTAF_Datakey.INPUT, PTAF_Datakey.MASK):
                raise ValueError("is not a valid label key.")

            # Check key is a valid one for which size is defined
            if isinstance(key, PTAF_Datakey):
                size += key.get_lbl_vector_size()
                sizes_dict[key.name] = key.get_lbl_vector_size()
            else:
                raise TypeError(f"Unsupported key type: {type(key)}")
        
        return size, sizes_dict