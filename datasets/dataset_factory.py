import typing

from datasets.data_synthesizers.MNIST_data_sythesizer import SinglePresenceSythesizer, DoublePresenceSythesizer
from datasets.MNIST.MNIST_dataloader import MNIST
from datasets.SCEMILA.SCEMILA_lightning_wrapper import SCEMILA

MODULES = {
    "MNIST": MNIST,
    "SCEMILA" : SCEMILA,
    "SinglePresenceSythesizer" :SinglePresenceSythesizer,
    "DoublePresenceSythesizer" : DoublePresenceSythesizer
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