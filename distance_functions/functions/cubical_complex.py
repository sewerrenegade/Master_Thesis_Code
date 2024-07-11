from models.topology_models.topo_tools.sliced_wasserstein_distance import SlicedWassersteinDistance
from models.topology_models.topo_tools.distances import WassersteinDistance
import torch.nn as nn
from models.topology_models.topo_tools.cubical_complex import CubicalComplex
import torch
import torchvision.transforms as transforms

class CubicalComplexImageEncoder(nn.Module):
    def __init__(self):
        super(CubicalComplexImageEncoder,self).__init__()
        self.device='cpu'
        self.cube_complex_encoder = CubicalComplex()
        self.wasserstein_distance = WassersteinDistance()
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Converts the image to a PyTorch tensor (HWC -> CHW, [0, 255] -> [0, 1])
        ])
    def forward(self, x):
        a = x[0]#first picture
        b = x[1]#second picture
        if not isinstance(x,torch.Tensor):
            a = torch.Tensor(x[0])
            b = torch.Tensor(x[1])
        cub_complexs_0 = self.cube_complex_encoder(a)
        cub_complexs_1 = self.cube_complex_encoder(b)
        distance = self.calculate_distance_between_complexes(cub_complexs_0,cub_complexs_1)
        return distance


    def calculate_distance_between_complexes(self,cub0,cub1):
        nb_channels = len(cub0)
        distances = []
        for channel_index in range(nb_channels):
            distances.append(self.wasserstein_distance(cub0[channel_index],cub1[channel_index]))
        return torch.sum(torch.stack(distances))



