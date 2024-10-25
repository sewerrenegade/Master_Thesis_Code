
import torch.nn as nn
from torch import stack,tensor,Tensor,long,abs
from numpy import ndarray
import numpy as np

from models.topology_models.custom_topo_tools.topo_encoder import ConnectivityEncoderCalculator
# topo function
class TopologicalZeroOrderLoss(nn.Module):
    """Topological signature."""
    LOSS_ORDERS = [1,2]
    PER_FEATURE_LOSS_SCALE_ESTIMATION_METHODS =["match_scale_order","match_scale_distribution","moor_method","modified_moor_method","deep"]

    def __init__(self,method="match_scale_distribution",p=1):
        """Topological signature computation.

        Args:
            p: Order of norm used for distance computation
            use_cycles: Flag to indicate whether cycles should be used
                or not.
        """
        super().__init__()
        assert p in TopologicalZeroOrderLoss.LOSS_ORDERS
        self.p = p
        self.signature_calculator = ConnectivityEncoderCalculator
        self.loss_fnc = self.get_torch_p_order_function()
        self.topo_feature_loss = self.get_topo_feature_approach(method)
        import torch

    def deep_scale_distribution_matching_loss_of_s1_on_s2(self,topo_encoding_space_1:ConnectivityEncoderCalculator,distances1,topo_encoding_space_2:ConnectivityEncoderCalculator,distances2):
        if distances2.requires_grad:
            # Initialize loss on the GPU
            loss = tensor(0.0, device=distances2.device)
            pairwise_distances_influenced = 0

            important_edge_for_each_scale = []
            # Iterate over persistence pairs in space 1
            for index, edge_indices in enumerate(topo_encoding_space_1.persistence_pairs):
                print(f"{index}/{len(topo_encoding_space_1.persistence_pairs)}")
                
                component_birth_in_s1_due_to_pers_pair = topo_encoding_space_1.get_component_birthed_at_index(index)
                scale_in_s1 = topo_encoding_space_1.scales[index]
                index_of_scale_in_s1 = topo_encoding_space_2.get_index_of_scale_closest_to(scale_in_s1)
                
                # Fetch relevant sets in space 2 for the current scale
                relevant_sets_in_s2 = topo_encoding_space_2.get_components_that_contain_these_points_at_this_index_or_scale(
                    relevant_points=component_birth_in_s1_due_to_pers_pair, index_of_scale=index_of_scale_in_s1
                )
                
                to_push_out_at_this_scale = []
                healthy_subsets = {}

                for component_in_s2_name, member_verticies in relevant_sets_in_s2.items():
                    # Intersect sets directly on the GPU if possible
                    good_verticies = np.intersect1d(member_verticies, component_birth_in_s1_due_to_pers_pair)
                    # Find vertices not in the birth component, append the persistence pair
                    for vertex in member_verticies:
                        if vertex not in component_birth_in_s1_due_to_pers_pair:
                            pair_info = topo_encoding_space_2.what_connected_this_point_to_this_set(
                                point=vertex, set=good_verticies
                            )["persistence_pair"]
                            to_push_out_at_this_scale.append(pair_info)
                    healthy_subsets[component_in_s2_name] = tensor(good_verticies, dtype=long, device=distances2.device)

                # Get edges needed to connect the components
                pairs_to_join_healthy_subsets = topo_encoding_space_2.what_edges_needed_to_connect_these_components(healthy_subsets)
                unique_to_push_out_at_this_scale = list(set(to_push_out_at_this_scale))
                
                # Combine important pairs
                important_pairs = pairs_to_join_healthy_subsets + unique_to_push_out_at_this_scale
                
                # If there are important pairs, perform loss calculation
                if len(important_pairs) != 0:
                    pairwise_distances_influenced += len(important_pairs)
                    important_edge_for_each_scale.append((scale_in_s1,important_pairs))
            for scale_edges_pair in important_edge_for_each_scale:
                # Convert important pairs to a tensor on GPU for indexing
                important_pairs_tensor =tensor(np.array(scale_edges_pair[1]), dtype=long, device=distances2.device)
                scale = tensor(scale_edges_pair[0], device=distances2.device)
                # Select and compute the differences on GPU
                selected_diff_distances = distances2[important_pairs_tensor[:, 0], important_pairs_tensor[:, 1]] / topo_encoding_space_2.distance_of_persistence_pairs[-1]

                # Compute the loss at this scale and accumulate
                loss_at_this_scale = abs(selected_diff_distances - scale) ** self.p
                loss = loss + loss_at_this_scale.sum()

            # Normalize the loss by the total number of pairs influenced
            return loss / pairwise_distances_influenced if pairwise_distances_influenced > 0 else tensor(0.0, device=distances2.device)
        else:
            return tensor(0.0, device=distances2.device)

    def deep_scale_distribution_matching_loss_of_s1_on_s2_old(self,topo_encoding_space_1:ConnectivityEncoderCalculator,distances1,topo_encoding_space_2:ConnectivityEncoderCalculator,distances2):
        if True:#distances2.requires_grad:
            loss = tensor(0.0, device=distances2.device)
            pairwise_distances_influenced = 0
            for index,edge_indices in enumerate(topo_encoding_space_1.persistence_pairs):
                print(f"{index}/{len(topo_encoding_space_1.persistence_pairs)}")
                component_birth_in_s1_due_to_pers_pair = topo_encoding_space_1.get_component_birthed_at_index(index)
                scale_in_s1 = topo_encoding_space_1.scales[index]
                index_of_scale_in_s1 = topo_encoding_space_2.get_index_of_scale_closest_to(scale_in_s1)
                relevant_sets_in_s2 = topo_encoding_space_2.get_components_that_contain_these_points_at_this_index_or_scale(relevant_points= component_birth_in_s1_due_to_pers_pair,index_of_scale= index_of_scale_in_s1)
                to_push_out_at_this_scale = []
                healthy_subsets = {}
                for component_in_s2_name,member_verticies in relevant_sets_in_s2.items():
                    good_verticies = np.intersect1d(member_verticies,component_birth_in_s1_due_to_pers_pair)
                    healthy_subsets[component_in_s2_name] = good_verticies
                    for vertex in member_verticies:
                        if not vertex in component_birth_in_s1_due_to_pers_pair:
                            to_push_out_at_this_scale.append(topo_encoding_space_2.what_connected_this_point_to_this_set(point=vertex,set=good_verticies)["persistence_pair"])
                pairs_to_join_healthy_subsets = topo_encoding_space_2.what_edges_needed_to_connect_these_components(healthy_subsets)
                unique_to_push_out_at_this_scale = list(set(to_push_out_at_this_scale))
                important_pairs = pairs_to_join_healthy_subsets + unique_to_push_out_at_this_scale
                if len(important_pairs) != 0:
                    pairwise_distances_influenced = pairwise_distances_influenced + len(important_pairs)
                    important_pairs = tensor(important_pairs, dtype=long) 
                    selected_diff_distances = distances2[important_pairs[:, 0], important_pairs[:, 1]]/ topo_encoding_space_2.distance_of_persistence_pairs[-1]
                    loss_at_this_scale = abs(selected_diff_distances - scale_in_s1)**self.p
                    loss = loss + loss_at_this_scale.sum()

            return loss/pairwise_distances_influenced
        else:
            return tensor(0.0, device=distances2.device)


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
        
        loss_2_on_1 = self.topo_feature_loss(topo_encoding_space_1=topo_encoding_space_1,
                                                      distances1=distances1,
                                                      topo_encoding_space_2=topo_encoding_space_2,
                                                      distances2=distances2)
        
        loss_1_on_2 = self.topo_feature_loss(topo_encoding_space_1=topo_encoding_space_2,
                                                      distances1=distances2,
                                                      topo_encoding_space_2=topo_encoding_space_1,
                                                      distances2=distances1)
        return loss_1_on_2 + loss_2_on_1,[]
    
    
    def get_topo_feature_approach(self,method):
        self.method = self.set_scale_matching_method(method)
        if self.method == TopologicalZeroOrderLoss.PER_FEATURE_LOSS_SCALE_ESTIMATION_METHODS[0]:
            topo_fnc =  self.scale_order_matching_loss_of_s1_on_s2
        elif self.method == TopologicalZeroOrderLoss.PER_FEATURE_LOSS_SCALE_ESTIMATION_METHODS[1]:
            topo_fnc =  self.scale_distribution_matching_loss_of_s1_on_s2
        elif self.method == TopologicalZeroOrderLoss.PER_FEATURE_LOSS_SCALE_ESTIMATION_METHODS[2]:
            topo_fnc =  self.moor_method_calculate_loss_of_s1_on_s2
        elif self.method == TopologicalZeroOrderLoss.PER_FEATURE_LOSS_SCALE_ESTIMATION_METHODS[3]:
            topo_fnc =  self.modified_moor_method_calculate_loss_of_s1_on_s2
        elif self.method == TopologicalZeroOrderLoss.PER_FEATURE_LOSS_SCALE_ESTIMATION_METHODS[4]:
            topo_fnc = self.deep_scale_distribution_matching_loss_of_s1_on_s2
        print(f"Using {self.method} to calculate per topo feature loss")
        return topo_fnc

    def set_scale_matching_method(self,scale_matching_method):
        if scale_matching_method in TopologicalZeroOrderLoss.PER_FEATURE_LOSS_SCALE_ESTIMATION_METHODS:
            return scale_matching_method
        else:
            raise ValueError(f"Scale matching methode {scale_matching_method} does not exist")
        
    def calulate_space_connectivity_encoding(self,distance_matrix):
        topo_encoder = self.signature_calculator(distance_matrix)
        topo_encoder.calculate_connectivity()
        return topo_encoder

    # using numpy to make sure autograd of torch is not disturbed
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
        
    def moor_method_calculate_loss_of_s1_on_s2(self,topo_encoding_space_1:ConnectivityEncoderCalculator,distances1,topo_encoding_space_2:ConnectivityEncoderCalculator,distances2):
        differentiable_scale_of_equivalent_edges_in_space_1 = []
        differentiable_scale_of_equivalent_edges_in_space_2 = []

        for index,edge_indices in enumerate(topo_encoding_space_1.persistence_pairs):
            scale_of_edge_in_space_1 = distances1[edge_indices[0], edge_indices[1]]
            scale_of_edge_in_space_2 = distances2[topo_encoding_space_1.persistence_pairs[index][0],topo_encoding_space_1.persistence_pairs[index][0]]
            differentiable_scale_of_equivalent_edges_in_space_1.append(scale_of_edge_in_space_1)
            differentiable_scale_of_equivalent_edges_in_space_2.append(scale_of_edge_in_space_2)
        differentiable_scale_of_equivalent_edges_in_space_1 = stack(differentiable_scale_of_equivalent_edges_in_space_1)
        differentiable_scale_of_equivalent_edges_in_space_2 = stack(differentiable_scale_of_equivalent_edges_in_space_2)
        return self.loss_fnc(differentiable_scale_of_equivalent_edges_in_space_1,differentiable_scale_of_equivalent_edges_in_space_2)
    def modified_moor_method_calculate_loss_of_s1_on_s2(self,topo_encoding_space_1:ConnectivityEncoderCalculator,distances1,topo_encoding_space_2:ConnectivityEncoderCalculator,distances2):
        differentiable_scale_of_equivalent_edges_in_space_1 = []
        differentiable_scale_of_equivalent_edges_in_space_2 = []

        for index,edge_indices in enumerate(topo_encoding_space_1.persistence_pairs):
            scale_of_edge_in_space_1 = distances1[edge_indices[0], edge_indices[1]]
            scale_of_edge_in_space_2 = distances2[topo_encoding_space_1.persistence_pairs[index][0],topo_encoding_space_1.persistence_pairs[index][0]]
            differentiable_scale_of_equivalent_edges_in_space_1.append(scale_of_edge_in_space_1)
            differentiable_scale_of_equivalent_edges_in_space_2.append(scale_of_edge_in_space_2)
        differentiable_scale_of_equivalent_edges_in_space_1 = stack(differentiable_scale_of_equivalent_edges_in_space_1)
        differentiable_scale_of_equivalent_edges_in_space_2 = stack(differentiable_scale_of_equivalent_edges_in_space_2)/ topo_encoding_space_2.distance_of_persistence_pairs[-1]
        return self.loss_fnc(differentiable_scale_of_equivalent_edges_in_space_1,differentiable_scale_of_equivalent_edges_in_space_2)
    
    def scale_order_matching_loss_of_s1_on_s2(self,topo_encoding_space_1:ConnectivityEncoderCalculator,distances1,topo_encoding_space_2:ConnectivityEncoderCalculator,distances2):
        differentiable_scale_of_equivalent_edges_in_space_1 = []
        differentiable_scale_of_equivalent_edges_in_space_2 = []

        for index,edge_indices in enumerate(topo_encoding_space_1.persistence_pairs):
            equivalent_feature_in_space_2 = topo_encoding_space_2.what_connected_these_two_points(edge_indices[0], edge_indices[1])
            equivalent_edge_in_space_2 = equivalent_feature_in_space_2["persistence_pair"]
            scale_of_edge_in_space_1 = tensor(topo_encoding_space_2.scales[index]).to(distances1.device) #will break numpy inputs; and is non differentiable
            scale_of_edge_in_space_2 = distances2[equivalent_edge_in_space_2[0], equivalent_edge_in_space_2[1]] / topo_encoding_space_2.distance_of_persistence_pairs[-1]            
            differentiable_scale_of_equivalent_edges_in_space_1.append(scale_of_edge_in_space_1)
            differentiable_scale_of_equivalent_edges_in_space_2.append(scale_of_edge_in_space_2)
        differentiable_scale_of_equivalent_edges_in_space_1 = stack(differentiable_scale_of_equivalent_edges_in_space_1)
        differentiable_scale_of_equivalent_edges_in_space_2 = stack(differentiable_scale_of_equivalent_edges_in_space_2)
        return self.loss_fnc(differentiable_scale_of_equivalent_edges_in_space_1,differentiable_scale_of_equivalent_edges_in_space_2)
    
    def scale_distribution_matching_loss_of_s1_on_s2(self,topo_encoding_space_1:ConnectivityEncoderCalculator,distances1,topo_encoding_space_2:ConnectivityEncoderCalculator,distances2):
        differentiable_scale_of_equivalent_edges_in_space_1 = []
        differentiable_scale_of_equivalent_edges_in_space_2 = []
        for index,edge_indices in enumerate(topo_encoding_space_1.persistence_pairs):
            equivalent_feature_in_space_2 = topo_encoding_space_2.what_connected_these_two_points(edge_indices[0], edge_indices[1])
            equivalent_edge_in_space_2 = equivalent_feature_in_space_2["persistence_pair"]
            scale_of_edge_in_space_1 = distances1[edge_indices[0], edge_indices[1]] / topo_encoding_space_1.distance_of_persistence_pairs[-1]
            scale_of_edge_in_space_2 = distances2[equivalent_edge_in_space_2[0], equivalent_edge_in_space_2[1]] / topo_encoding_space_2.distance_of_persistence_pairs[-1]
            differentiable_scale_of_equivalent_edges_in_space_1.append(scale_of_edge_in_space_1)
            differentiable_scale_of_equivalent_edges_in_space_2.append(scale_of_edge_in_space_2)
        differentiable_scale_of_equivalent_edges_in_space_1 = stack(differentiable_scale_of_equivalent_edges_in_space_1)
        differentiable_scale_of_equivalent_edges_in_space_2 = stack(differentiable_scale_of_equivalent_edges_in_space_2)
        return self.loss_fnc(differentiable_scale_of_equivalent_edges_in_space_1,differentiable_scale_of_equivalent_edges_in_space_2)
    


    
    def get_torch_p_order_function(self):
        if self.p ==1 :
            return nn.L1Loss()
        elif self.p == 2:
            return nn.MSELoss()
        else:
            raise ValueError(f"This loss {self.p} is not supported")
        