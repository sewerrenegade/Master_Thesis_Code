import torch
import torch.nn as nn



class PCA_Distance(nn.Module):
    def __init__(self, latent_dim=128,output_space_dim = 64):
        super(PCA_Distance,self).__init__()
        self.device='cuda:0'#self.device='cpu'#
        self.in_dim = latent_dim
        self.out_dim = output_space_dim


    def forward(self, x):
        input = x.to(self.device)
        U,S,V = torch.pca_lowrank(A = input,q= self.out_dim)
        down_projected = V
        distance_matrix = torch.norm(down_projected[:, None] - down_projected, dim=2, p=2)
        return distance_matrix
    
class Random_Distance(nn.module):
    def __init__(self, latent_dim=128, mean=5, var=1):
        super(Random_Distance,self).__init__()
        self.device='cuda:0'#self.device='cpu'#
        self.in_dim = latent_dim
        self.mean = mean
        self.var = var

    def forward(self,x):
        shape = x.shape[-1]
        return torch.randn(shape, shape)#wack

class UMap_Distance(nn.Module):
    def __init__(self,latent_dim = 128,output_space_dim = 64):
        super(UMap_Distance,self).__init__()
