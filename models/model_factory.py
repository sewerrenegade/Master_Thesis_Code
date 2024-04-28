
from models.salome_models.mil import CV_MIL
from models.salome_models.topo_reg_mil import CV_TopoRegMIL
from models.SCEMILA.model import AMiL
from torch.optim import SGD
from models.salome_models.optimizer import Adam,SGD
from models.salome_models.scheduler import ConstantScheduler


MODULES = {
    "CV_MIL": CV_MIL,
    "CV_TopoRegMIL": CV_TopoRegMIL,
    "Adam": Adam,
    "ConstantScheduler": ConstantScheduler,
    "SCEMILA":AMiL,
    "SGD" : SGD
}

NAME_KEY: str = "name"
CONFIG_KEY: str = "config"


def get_module(name: str, config):
    """Recursively deserializes objects registered in MODULES."""
    if name not in MODULES:
        raise KeyError(
            f"{name} not found in registered modules. Available are {MODULES.keys()}."
        )
    cls = MODULES[name]
    for key, value in config.items():
        if(isinstance(value, dict)) and NAME_KEY in value and CONFIG_KEY in value:
                config[key] = get_module(value[NAME_KEY], value[CONFIG_KEY])
    return cls(**config)