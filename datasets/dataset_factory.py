import typing
import sys
from datasets.SCEMILA.base_image_SCEMILA import SCEMILA_MIL_base, SCEMILA_base
sys.path.append('/home/milad/Desktop/Master_Thesis/code/Master_Thesis_Code')
 
from datasets.SCEMILA.base_fnl34_features import SCEMILA_fnl34_feature_base
from datasets.data_synthesizers.data_sythesizer import SinglePresenceMILSynthesizer, DoublePresenceSythesizer
from datasets.MNIST.MNIST_dataloader import MNIST
from datasets.FashionMNIST.FashionMNIST_base import FashionMNIST_MIL_base, FashionMNIST_base
from datasets.MNIST.MNIST_base import MNIST_MIL_base, MNISTBase
from datasets.CIFAR10.CIFAR10_base import CIFAR10_MIL_base, CIFAR10_base
from datasets.Acevedo.acevedo_base import Acevedo_MIL_base, Acevedo_base
# from datasets.SCEMILA.base_image_SCEMILA import SCEMILAimage_base,SCEMILA_fnl34_feature_base,SCEMILA_DinoBloom_feature_base
#from datasets.SCEMILA import *
from datasets.SCEMILA.SCEMILA_lightning_wrapper import SCEMILA


MODULES = {
    "MNIST": MNIST,
    "SCEMILA" : SCEMILA,
    "SinglePresenceSythesizer" :SinglePresenceMILSynthesizer,
    #"DoublePresenceSythesizer" : DoublePresenceSythesizer
}
BASE_MODULES = {
    "FashionMNIST": FashionMNIST_base,
    "MNIST": MNISTBase,
    "CIFAR10": CIFAR10_base,
    "SCEMILA/fnl34_feature_data": SCEMILA_fnl34_feature_base,
    "SCEMILA/image_data": SCEMILA_base,
    "Acevedo":Acevedo_base,
    "MIL_FashionMNIST": FashionMNIST_MIL_base,
    "MIL_MNIST": MNIST_MIL_base,
    "MIL_CIFAR10": CIFAR10_MIL_base,
    #"MIL_SCEMILA/fnl34_feature_data": SCEMILA_fnl34_feature_base,
    #"MIL_SCEMILA/image_data": SCEMILAimage_base,
    "MIL_Acevedo":Acevedo_MIL_base,
    "MIL_SCEMILA": SCEMILA_MIL_base
    
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