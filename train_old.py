

# from experiments.augmentation_sensitivity import produce_experiment_elements


# # produce_experiment_elements()
# import torch.nn as nn
# from torchvision import models
# resnet34 = models.resnet34()
# res34 = list(resnet34.children())
# resnet50 = models.resnet50()
# res50 = list(resnet50.children())
# pass
import numpy as np
vals = [0.98166, 0.98074, 0.96297, 0.97071, 0.96439, 0.9483, 0.97307, 0.96623]
print(f"avg: {np.mean(vals)} \n std: {np.std(vals)}")