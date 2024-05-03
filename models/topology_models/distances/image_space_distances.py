import torch
import torch.nn as nn
import lpips
import numpy as np
import json

from models.topology_models.topo_tools.cubical_complex import CubicalComplex
from models.topology_models.topo_tools.sliced_wasserstein_distance import SlicedWassersteinDistance
from models.topology_models.topo_tools.distances import WassersteinDistance


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
        print(f"Gradient of input {x.grad_fn}")
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


class ReconstructionProjectionModel(nn.Module):
    def __init__(self, input_dim=[1,28,28],path_to_model = ""):
        super(ReconstructionProjectionModel,self).__init__()
        self.device='cuda:0'
        self.encoder = nn.Sequential(
            # 28 x 28
            nn.Conv2d(1, 4, kernel_size=5),
            # 4 x 24 x 24
            nn.ReLU(True),
            nn.Conv2d(4, 8, kernel_size=5),
            nn.ReLU(True),
            # 8 x 20 x 20 = 3200
            nn.Flatten(),
            nn.Linear(3200, 10),
            # 10
            nn.Softmax(),
            )
        self.decoder = nn.Sequential(
            # 10
            nn.Linear(10, 400),
            # 400
            nn.ReLU(True),
            nn.Linear(400, 4000),
            # 4000
            nn.ReLU(True),
            nn.Unflatten(1, (10, 20, 20)),
            # 10 x 20 x 20
            nn.ConvTranspose2d(10, 10, kernel_size=5),
            # 24 x 24
            nn.ConvTranspose2d(10, 1, kernel_size=5),
            # 28 x 28
            nn.Sigmoid(),
            )
        if path_to_model != "":
            self.load_state_dict(torch.load(path_to_model))

    def forward(self, x):
        #x = torch.unsqueeze(x, dim=0)
        enc = self.encoder(x)
        dec = self.decoder(enc)
        #x = torch.squeeze(x,dim=0)
        return dec
    
    def get_latent_code_for_input(self,input):
        return self.encoder(input)
    


class PerceptualLoss(nn.Module):
    """
    Perceptual Loss function module, which returns perceptual loss distance matrix for a given input batch
    """
    def __init__(self, device='cuda:0', net='vgg', **kwargs):
        super(PerceptualLoss, self).__init__(**kwargs)
        self.loss_fn = lpips.LPIPS(net=net)
        self.device = device

    def forward(self, x):
        """
        Compute perceptual loss for each pair of samples of a batch to return a distance matrix
        """
        #get triangular indices for computing a distance matrix
        n = x.shape[0]
        inds = self._get_index_pairs(n)
        
        #broadcast data such that each pair of the batch is represented
        batch0 = x[inds[0]]  
        batch1 = x[inds[1]]

        #compute loss/distance pair-wise on the batch level
        loss = self.loss_fn(batch0, batch1) 
        loss = loss.view(loss.shape[0])

        #reshape output into a proper distance matrix
        D = self._batch_to_matrix(loss, inds, n)
        return D 

    def _get_index_pairs(self, n):
        """
        return all pairs of indices of two 1d index tensors
        """
        inds = torch.triu_indices(n, n)
        return inds
    
    def _batch_to_matrix(self, x, inds, n):
        """
        Reshape batched result to distance matrix
        """
        D = torch.zeros(n, n, dtype=torch.float32, device=self.device)
        D[inds[0], inds[1]] = x
        return self._triu_to_full(D)

    def _triu_to_full(self, D):
        """
        Convert triu (upper triangular) matrix to full, symmetric matrix.
        Assumes square input matrix D
        """
        diagonal = torch.eye(D.shape[0], dtype=torch.float32, 
                device=self.device) * torch.diag(D) #eye matrix with diagonal entries of D 
        D = D + D.T - diagonal # add transpose minus diagonal values 
        # to move from upper triangular to full matrix 
        return D

class RandomProjectionModel(nn.Module):
    """Conv architecture (initially for CIFAR)."""
    def __init__(self, input_dim=[3,32,32]):
        super().__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.projection = nn.Sequential(
            nn.Conv2d(input_dim[0], 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4] ~768 dim
            nn.ReLU(),
            )
    def forward(self, x):
        """Compute latent representation using convolutional net."""
        batch_dim = x.shape[0]
        x = self.projection(x)
        return x.view(batch_dim, -1)