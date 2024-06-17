from models.topology_models.topo_tools.sliced_wasserstein_distance import SlicedWassersteinDistance
from models.topology_models.topo_tools.distances import WassersteinDistance
import torch.nn as nn
from models.topology_models.topo_tools.cubical_complex import CubicalComplex
import torch

class CubicalComplexImageEncoder(nn.Module):
    def __init__(self, image_dim=[28,28],):
        super(CubicalComplexImageEncoder,self).__init__()
        self.device='cpu'#self.device='cuda:0'
        self.image_dim = image_dim
        self.cube_complex_encoder = CubicalComplex(dim = len(image_dim))
        self.wasserstein_distance = WassersteinDistance(q = 2)
        #self.requires_grad_ = True

    def forward(self, x):
        input = x.to(self.device)
        #print(f"Gradient of input {x.grad_fn}")
        cub_complexs = self.cube_complex_encoder(input)
        distances = self.calculate_distance_matrix(cub_complexs)
        distances.requires_grad_(True)

        return distances.to('cuda:0')


    def calculate_distance_matrix(self,elements):
        distance_matrix = torch.zeros(len(elements),len(elements),device=self.device)

        for i in range(len(elements)):
            for j in range(i + 1, len(elements)):
                x, y = elements[i][0], elements[j][0]
                distance = self.wasserstein_distance(x, y)
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
        return distance_matrix



