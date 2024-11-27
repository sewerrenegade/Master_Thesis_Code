import importlib
import typing

MODULES = {
    "MNIST": "datasets.MNIST.MNIST_dataloader.MNIST",
    "SCEMILA": "datasets.SCEMILA.SCEMILA_lightning_wrapper.SCEMILA",
    "SinglePresenceSythesizer": "datasets.data_synthesizers.data_sythesizer.SinglePresenceMILSynthesizer",
}

BASE_MODULES = {
    "FashionMNIST": "datasets.FashionMNIST.FashionMNIST_base.FashionMNIST_base",
    "MNIST": "datasets.MNIST.MNIST_base.MNISTBase",
    "CIFAR10": "datasets.CIFAR10.CIFAR10_base.CIFAR10_base",
    "SCEMILA/fnl34_feature_data": "datasets.SCEMILA.base_fnl34_features.SCEMILA_fnl34_feature_base",
    "SCEMILA/image_data": "datasets.SCEMILA.base_image_SCEMILA.SCEMILA_base",
    "Acevedo": "datasets.Acevedo.acevedo_base.Acevedo_base",
    "MIL_FashionMNIST": "datasets.FashionMNIST.FashionMNIST_base.FashionMNIST_MIL_base",
    "MIL_MNIST": "datasets.MNIST.MNIST_base.MNIST_MIL_base",
    "MIL_CIFAR10": "datasets.CIFAR10.CIFAR10_base.CIFAR10_MIL_base",
    "MIL_Acevedo": "datasets.Acevedo.acevedo_base.Acevedo_MIL_base",
    "MIL_SCEMILA": "datasets.SCEMILA.base_image_SCEMILA.SCEMILA_MIL_base",
}

NAME_KEY: str = "name"
CONFIG_KEY: str = "config"


def lazy_import(path: str):
    """Lazy imports a module and retrieves the required class or function."""
    module_path, class_name = path.rsplit(".", 1)  # Split into module and class
    module = importlib.import_module(module_path)  # Import the module dynamically
    return getattr(module, class_name)  # Get the class or function from the module


def get_module(name: str, config: typing.Dict[str, typing.Any]):
    """Recursively deserializes objects registered in MODULES."""
    if name not in MODULES:
        raise KeyError(
            f"{name} not found in registered modules. Available are {MODULES.keys()}."
        )
    cls_path = MODULES[name]
    cls = lazy_import(cls_path)  # Dynamically load the class
    for key, value in config.items():
        if isinstance(value, dict) and NAME_KEY in value and CONFIG_KEY in value:
            config[key] = get_module(value[NAME_KEY], value[CONFIG_KEY])
    return cls(**config)