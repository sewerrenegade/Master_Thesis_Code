import typing
import sys
sys.path.append('/home/milad/Desktop/Master_Thesis/code/Master_Thesis_Code')
 
from datasets.data_synthesizers.MNIST_data_sythesizer import SinglePresenceSythesizer, DoublePresenceSythesizer
from datasets.MNIST.MNIST_dataloader import MNIST
from datasets.FashionMNIST.FashionMNIST_base import FashionMNIST_base
from datasets.MNIST.MNIST_base import MNIST_base
from datasets.CIFAR10.CIFAR10_base import CIFAR10_base
# from datasets.SCEMILA.base_image_SCEMILA import SCEMILAimage_base,SCEMILA_fnl34_feature_base,SCEMILA_DinoBloom_feature_base
from datasets.SCEMILA import *
from datasets.SCEMILA.SCEMILA_lightning_wrapper import SCEMILA


MODULES = {
    "MNIST": MNIST,

    "SCEMILA" : SCEMILA,
    "SinglePresenceSythesizer" :SinglePresenceSythesizer,
    "DoublePresenceSythesizer" : DoublePresenceSythesizer
}
BASE_MODULES = {
    "FashionMNIST": FashionMNIST_base,
    "MNIST": MNIST_base,
    "CIFAR10": CIFAR10_base,
    "SCEMILA/fnl34_feature_data": SCEMILA_fnl34_feature_base,
    "SCEMILA/image_data": SCEMILAimage_base,
    "SCEMILA/dinobloom_feature_data": SCEMILA_DinoBloom_feature_base
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