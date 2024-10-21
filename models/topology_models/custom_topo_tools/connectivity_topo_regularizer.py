
import torch.nn as nn
from torch import stack,tensor,Tensor
from numpy import ndarray

from models.topology_models.custom_topo_tools.milad_topo import ConnectivityEncoderCalculator
# topo function
class TopologicalZeroOrderLoss(nn.Module):
    """Topological signature."""
    LOSS_ORDERS = [1,2]
    PER_FEATURE_LOSS_SCALE_ESTIMATION_METHODS =["match_scale_order","match_scale_distribution"]

    def __init__(self,p=1,scale_matching_method="match_scale_order"):
        """Topological signature computation.

        Args:
            p: Order of norm used for distance computation
            use_cycles: Flag to indicate whether cycles should be used
                or not.
        """
        super().__init__()
        assert p in TopologicalZeroOrderLoss.LOSS_ORDERS
        self.p = p
        self.scale_matching_method = self.set_scale_matching_method(scale_matching_method)
        self.signature_calculator = ConnectivityEncoderCalculator
        self.loss_fnc = self.get_torch_p_order_function()

    def set_scale_matching_method(self,scale_matching_method):
        if scale_matching_method in TopologicalZeroOrderLoss.PER_FEATURE_LOSS_SCALE_ESTIMATION_METHODS:
            return scale_matching_method
        else:
            raise ValueError(f"Scale matching methode {scale_matching_method} does not exist")
        
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
        
    def calculate_loss_of_s1_on_s2(self,topo_encoding_space_1:ConnectivityEncoderCalculator,distances1,topo_encoding_space_2:ConnectivityEncoderCalculator,distances2):
        differentiable_scale_of_equivalent_edges_in_space_1 = []
        differentiable_scale_of_equivalent_edges_in_space_2 = []

        for index,edge_indices in enumerate(topo_encoding_space_1.persistence_pairs):
            equivalent_feature_in_space_2 = topo_encoding_space_2.what_connected_these_two_points(edge_indices[0], edge_indices[1])
            equivalent_edge_in_space_2 = equivalent_feature_in_space_2["persistence_pair"]
            if self.scale_matching_method == TopologicalZeroOrderLoss.PER_FEATURE_LOSS_SCALE_ESTIMATION_METHODS[0]:
                scale_of_edge_in_space_1 = tensor(topo_encoding_space_2.scales[index])
            elif self.scale_matching_method == TopologicalZeroOrderLoss.PER_FEATURE_LOSS_SCALE_ESTIMATION_METHODS[1]:
                scale_of_edge_in_space_1 = distances1[edge_indices[0], edge_indices[1]] / topo_encoding_space_1.distance_of_persistence_pairs[-1]

            scale_of_equivalent_edge_in_space_2 = distances2[equivalent_edge_in_space_2[0], equivalent_edge_in_space_2[1]] / topo_encoding_space_2.distance_of_persistence_pairs[-1]

            
            differentiable_scale_of_equivalent_edges_in_space_1.append(scale_of_edge_in_space_1)
            differentiable_scale_of_equivalent_edges_in_space_2.append(scale_of_equivalent_edge_in_space_2)
        differentiable_scale_of_equivalent_edges_in_space_1 = stack(differentiable_scale_of_equivalent_edges_in_space_1)
        differentiable_scale_of_equivalent_edges_in_space_2 = stack(differentiable_scale_of_equivalent_edges_in_space_2)
        return self.loss_fnc(differentiable_scale_of_equivalent_edges_in_space_1,differentiable_scale_of_equivalent_edges_in_space_2)

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
    
    def get_torch_p_order_function(self):
        if self.p ==1 :
            return nn.L1Loss()
        elif self.p == 2:
            return nn.MSELoss()
        else:
            raise ValueError(f"This loss {self.p} is not supported")
        