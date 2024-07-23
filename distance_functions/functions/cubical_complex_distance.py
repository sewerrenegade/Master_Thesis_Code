from models.topology_models.topo_tools.data import PersistenceInformation
from models.topology_models.topo_tools.sliced_wasserstein_distance import SlicedWassersteinDistance
from models.topology_models.topo_tools.distances import WassersteinDistance,MaximumMeanDiscrepancyDistance
import torch.nn as nn
from models.topology_models.topo_tools.cubical_complex import CubicalComplex
import torch
import torchvision.transforms as transforms
import numpy as np

class CubicalComplexImageDistanceFunction(nn.Module):
    def __init__(self,calculate_holes = True,join_channels = False,distribution_distance = "WasserStein"):
        super(CubicalComplexImageDistanceFunction,self).__init__()
        self.name = "Cubical Complex Distance"
        self.device='cpu'
        self.calculate_holes = calculate_holes
        self.join_channels = join_channels
        self.cube_complex_encoder = CubicalComplex()
        self.distribution_distance_name = distribution_distance
        if distribution_distance ==  "WasserStein":
            self.distribution_distance = WassersteinDistance()
        elif distribution_distance == "MaximumMeanDiscrepancy":
            self.distribution_distance = MaximumMeanDiscrepancyDistance()
        else:
            raise NotImplementedError(f"Unknown distribution distance: {distribution_distance}")
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Converts the image to a PyTorch tensor (HWC -> CHW, [0, 255] -> [0, 1])
        ])
    
    def get_settings(self):
        return {'calculate_holes': self.calculate_holes,
                'join_channels':self.join_channels,
                'distribution_distance':self.distribution_distance_name,
                }
    def forward(self, x, y =None):
        if y is None:
            assert len(x) == 2, "Input list must contain exactly two elements."
            a = x[0]#first picture
            b = x[1]#second picture
        else:
            a = x
            b = y
        if not isinstance(a,torch.Tensor):
            a = torch.Tensor(a)
            b = torch.Tensor(b)
        else:
            assert not a.is_cuda
        cub_complexs_0 = self.cube_complex_encoder(a)
        cub_complexs_1 = self.cube_complex_encoder(b)
        if not self.calculate_holes:
            cub_complexs_0 = self.remove_first_dimension_features(cub_complexs_0)
            cub_complexs_1 = self.remove_first_dimension_features(cub_complexs_1)
        if self.join_channels:
            cub_complexs_0 = self.merge_cubical_complex_persistence_info_across_channels(cub_complexs_0)
            cub_complexs_1 = self.merge_cubical_complex_persistence_info_across_channels(cub_complexs_1)

            
        distance = self.calculate_distance_between_complexes(cub_complexs_0,cub_complexs_1)
        return distance


    def calculate_distance_between_complexes(self,cub0,cub1):
        nb_channels = len(cub0)
        distances = []
        if not self.join_channels:
            for channel_index in range(nb_channels):
                distances.append(self.distribution_distance(cub0[channel_index],cub1[channel_index]))
        else:
            for channel_index in range(nb_channels):
                distances.append(self.distribution_distance(cub0[channel_index],cub1[channel_index]))
        return torch.sum(torch.stack(distances))

    def remove_first_dimension_features(self,list_of_per_infos):
        new_per_infos = []
        for channel_per_info_index in range(len(list_of_per_infos)):
            new_per_infos.append(list_of_per_infos[channel_per_info_index][:-1])
        return new_per_infos
            
    def merge_cubical_complex_persistence_info_across_channels(self,per_infos):
        new_per_info = []
        dims = len(per_infos[0])
        for dim in range(dims):
            pixel_pairings = torch.cat([per_infos[i][dim].pairing for i in range(len(per_infos))])
            persistence_diagram = torch.cat([per_infos[i][dim].diagram for i in range(len(per_infos))])
            dimension = dim
            new_per_info.append(
                PersistenceInformation(
                    pairing=pixel_pairings, # these are the pixels that turn on/off in order to create/destory topo features
                    diagram=persistence_diagram, # these are the persistence values at which the topo features are craeted/destoryed
                    dimension=dimension # for a normal image (not volumetric) this is 0 or 1; 0 for connected components and 1 for rings/holes
        ))
        return [new_per_info] #this is to imply that it is single channel now.
