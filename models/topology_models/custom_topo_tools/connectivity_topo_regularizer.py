
import torch.nn as nn
from torch import cat,randperm,stack,Tensor
from numpy import ndarray

from models.topology_models.custom_topo_tools.milad_topo import ConnectivityEncoderCalculator
# topo function
class TopologicalZeroOrderLoss(nn.Module):
    """Topological signature."""

    def __init__(self, to_gpu=False):
        """Topological signature computation.

        Args:
            p: Order of norm used for distance computation
            use_cycles: Flag to indicate whether cycles should be used
                or not.
        """
        super().__init__()
        self.to_gpu = to_gpu
        self.signature_calculator = ConnectivityEncoderCalculator
        self.l1_loss = nn.L1Loss()

    def calulate_space_connectivity_encoding(self,distance_matrix):
        topo_encoder = self.signature_calculator(distance_matrix)
        topo_encoder.calculate_connectivity()
        return topo_encoder

    @staticmethod
    def to_numpy(obj):
        if isinstance(obj, ndarray):
            # If it's already a NumPy array, return as is
            return obj
        elif isinstance(obj, Tensor):
            # Detach the tensor from the computation graph, move to CPU if necessary, and convert to NumPy
            return obj.detach().cpu().numpy()
        else:
            raise TypeError("Input must be a NumPy array or a PyTorch tensor")
        
    def calculate_loss_of_s1_on_s2(self,topo_encoding_space_1,distances1,topo_encoding_space_2,distances2):
        differentiable_scale_of_equivalent_edges_in_space_1 = []
        differentiable_scale_of_equivalent_edges_in_space_2 = []

        # Iteratively compute corresponding edges and scales in space 2
        for edge_indices in topo_encoding_space_1.persistence_pairs:
            # Get the equivalent feature in space 2 for each pair of points
            equivalent_feature_in_space_2 = topo_encoding_space_2.what_connected_these_two_points(edge_indices[0], edge_indices[1])
            
            # Extract the corresponding persistence pair in space 2
            equivalent_edge_in_space_2 = equivalent_feature_in_space_2["persistence_pair"]
            
            # Calculate the scale for the equivalent edge in space 2
            scale_of_equivalent_edge_in_space_2 = distances2[equivalent_edge_in_space_2[0], equivalent_edge_in_space_2[1]] / topo_encoding_space_2.distance_of_persistence_pairs[-1]
            scale_of_edge_in_space_1 = distances1[edge_indices[0], edge_indices[1]] / topo_encoding_space_1.distance_of_persistence_pairs[-1]
            # Store the result directly (no need to convert to a new tensor)
            differentiable_scale_of_equivalent_edges_in_space_1.append(scale_of_edge_in_space_1)
            differentiable_scale_of_equivalent_edges_in_space_2.append(scale_of_equivalent_edge_in_space_2)
        differentiable_scale_of_equivalent_edges_in_space_1 = stack(differentiable_scale_of_equivalent_edges_in_space_1)
        differentiable_scale_of_equivalent_edges_in_space_2 = stack(differentiable_scale_of_equivalent_edges_in_space_2)
        return self.l1_loss(differentiable_scale_of_equivalent_edges_in_space_1,differentiable_scale_of_equivalent_edges_in_space_2)

    def forward(self, distances1, distances2):
        """Return topological distance of two pairwise distance matrices.

        Args:
            distances1: Distance matrix in space 1
            distances2: Distance matrix in space 2

        Returns:
            distance, dict(additional outputs)
        """
        nondiff_distances1 = TopologicalZeroOrderLoss.to_numpy(distances1)
        nondiff_distances2 = TopologicalZeroOrderLoss.to_numpy(distances2)
        topo_encoding_space_1 = self.calulate_space_connectivity_encoding(nondiff_distances1)
        topo_encoding_space_2 = self.calulate_space_connectivity_encoding(nondiff_distances2)
        
        loss_2_on_1 = self.calculate_loss_of_s1_on_s2(topo_encoding_space_1=topo_encoding_space_1,
                                                      distances1=distances1,
                                                      topo_encoding_space_2=topo_encoding_space_2,
                                                      distances2=distances2)
        
        loss_1_on_2 = self.calculate_loss_of_s1_on_s2(topo_encoding_space_1=topo_encoding_space_2,
                                                      distances1=distances2,
                                                      topo_encoding_space_2=topo_encoding_space_1,
                                                      distances2=distances1)
        return loss_1_on_2 + loss_2_on_1,[]