import numpy as np


#WARNING: none of this is differentiable, torch can not track gradients through this... use this only to calculate and match topological features
class ConnectivityEncoderCalculator:
    def __init__(self, distance_mat):
        assert isinstance(distance_mat, np.ndarray)
        assert distance_mat.shape[0] == distance_mat.shape[1]
        assert len(distance_mat.shape) == 2
        self.distance_matrix = distance_mat
        self.n_vertices = distance_mat.shape[0]
        zero_scale_topo = np.arange(self.n_vertices, dtype=int)
        self.topo_scale_evolution =[zero_scale_topo]
        self.topo_progression_stats= [np.unique(zero_scale_topo, return_counts=True)]
        self._current_topology = np.arange(self.n_vertices, dtype=int)
        self.persistence_pairs = None
        self.scales = None
        self.distance_of_persistence_pairs = None
        self.sanity_checker = []

    def calculate_connectivity(self):
        tri_strict_upper_indices = np.triu_indices_from(self.distance_matrix,k=1)
        edge_weights = self.distance_matrix[tri_strict_upper_indices]
        edge_indices = np.argsort(edge_weights, kind='stable')

        persistence_pairs = []
        edge_distances = []

        for edge_index, edge_weight in zip(edge_indices, edge_weights[edge_indices]):

            u = tri_strict_upper_indices[0][edge_index]
            v = tri_strict_upper_indices[1][edge_index]

            u_group = self.get_point_current_group(u)
            v_group = self.get_point_current_group(v)
            if u_group == v_group :
                continue # no 0 order topological feature created since connectivity remains unchanged

            self.merge_groups(u_group, v_group)
            persistence_pairs.append((min(u, v), max(u, v)))
            edge_distances.append(edge_weight)
            self.save_current_topo_enc()
            if len(persistence_pairs) == self.n_vertices -1:
                break
        self.scales = [edge_distance/edge_distances[-1] for edge_distance in edge_distances]
        self.distance_of_persistence_pairs = edge_distances
        self.persistence_pairs = persistence_pairs


    def save_current_topo_enc(self):
        self.topo_scale_evolution.append(np.copy(self._current_topology))
        x= np.unique(self._current_topology, return_counts=True)
        stats = {}
        stats["cluster_ids"] = x[0]
        stats["cluster_nb_of_memebers"] = x[1]
        stats["number_of_clusters"] = len(x[0])
        self.topo_progression_stats.append(stats)
        self.sanity_checker.append(len(x[0]))


    def get_point_current_group(self, u):
        return self._current_topology[u]

    def merge_groups(self, u_group, v_group):
        if u_group > v_group:
            self._current_topology[np.where(self._current_topology == v_group)] = u_group
        elif u_group < v_group:
            self._current_topology[np.where(self._current_topology == u_group)] = v_group
        else:
            print("WTF u doing idiot")
            assert u_group != v_group

    def what_connected_these_two_points(self,u,v):
        for index,connectivity in enumerate(self.topo_scale_evolution):
            if connectivity[u] == connectivity[v]:
                connecting_index = index - 1 # this is because the s=0 topology encoding is inserted automatically
                break
        connecting_info = {"index": connecting_index,"persistence_pair": self.persistence_pairs[connecting_index],"scale": self.scales[connecting_index],"median_order":connecting_index/len(self.persistence_pairs)}
        return connecting_info