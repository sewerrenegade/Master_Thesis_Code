import typing
import sys
sys.path.append('/home/milad/Desktop/Master_Thesis/code/Master_Thesis_Code')
 

from datasets.SCEMILA.base_image_SCEMILA import SCEMILA_base
from distance_functions.functions.basic_distance_functions import CosineSimilarity, EuclideanDistance, L1Distance, LinfinityDistance, LpDistance
from distance_functions.functions.cubical_complex_distance import CubicalComplexImageDistanceFunction
from distance_functions.functions.perceptual_lpsis_distance import PerceptualLoss
from distance_functions.functions.random_convolutions_distance import RandomProjectionModel


MODULES = {
    "Euclidean Distance": EuclideanDistance,
    "Cosine Similarity Distance" : CosineSimilarity,
    "L1 Distance" :L1Distance,
    "L-infinity Distance" : LinfinityDistance,
    "Lp Distance": LpDistance,
    "Cubical Complex Distance": CubicalComplexImageDistanceFunction,
    "Perceptual Distance":PerceptualLoss,
    "Random Convolutions Distance": RandomProjectionModel
    
}


NAME_KEY: str = "name"
CONFIG_KEY: str = "config"


def get_module(name: str, config: typing.Dict[str, typing.Any]):
    """Recursively deserializes objects registered in MODULES."""
    if name not in MODULES:
        raise KeyError(
            f"{name} not found in registered modules. Available are {MODULES.keys()}."
        )
    cls = MODULES[name]
    for key, value in config.items():
        if isinstance(value, dict) and NAME_KEY in value and CONFIG_KEY in value:
            config[key] = get_module(value[NAME_KEY], value[CONFIG_KEY])
    return cls(**config)