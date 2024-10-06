import numpy as np
import typing
import torch.nn as nn
from dataclasses import dataclass
import torch
from torchmetrics import Accuracy

from models.salome_models.scheduler import  ConstantScheduler
from models.topology_models.topo_tools.topology import PersistentHomologyCalculation
from models.salome_models.mil import CV_MIL,MILOutput
from distance_functions.functions.perceptual_lpsis_distance import PerceptualLoss
from distance_functions.functions.cubical_complex_distance import CubicalComplexImageDistanceFunction
from distance_functions.functions.random_convolutions_distance import RandomProjectionModel
from distance_functions.functions.reconstruction_functions.classic_ae_distance.ae_reconstruction_distance import ReconstructionProjectionModel
from models.topology_models.topo_tools import SignatureLoss, VietorisRipsComplex, WassersteinDistance




@dataclass
class ToporegMILOutput(MILOutput):
    ins_img: torch.Tensor
    bag_size: int
    img_distances: torch.Tensor
    latent_distances: torch.Tensor    

# topo function
class TopologicalSignatureDistance(nn.Module):
    """Topological signature."""

    def __init__(self, sort_selected=False, use_cycles=False,
                 match_edges=None,to_gpu = True):
        """Topological signature computation.

        Args:
            p: Order of norm used for distance computation
            use_cycles: Flag to indicate whether cycles should be used
                or not.
        """
        super().__init__()
        self.use_cycles = use_cycles
        self.match_edges = match_edges
        self.to_gpu = to_gpu
        print('Using python to compute signatures')
        self.signature_calculator = PersistentHomologyCalculation()

    def _get_pairings(self, distances):
        pairs_0, pairs_1 = self.signature_calculator(
            distances.detach().cpu().numpy())

        return pairs_0, pairs_1

    def _select_distances_from_pairs(self, distance_matrix, pairs):
        # Split 0th order and 1st order features (edges and cycles)
        pairs_0, pairs_1 = pairs
        selected_distances = distance_matrix[(pairs_0[:, 0], pairs_0[:, 1])]

        if self.use_cycles:
            edges_1 = distance_matrix[(pairs_1[:, 0], pairs_1[:, 1])]
            edges_2 = distance_matrix[(pairs_1[:, 2], pairs_1[:, 3])]
            edge_differences = edges_2 - edges_1

            selected_distances = torch.cat(
                (selected_distances, edge_differences))

        return selected_distances
    
    
    def sig_error(self,signature1, signature2):
        """Compute distance between two topological signatures."""
        if self.to_gpu:
            return ((signature1.cuda() - signature2.cuda())**2).sum(dim=-1)
        else:
            return ((signature1 - signature2)**2).sum(dim=-1)
    
    @staticmethod
    def _count_matching_pairs(pairs1, pairs2):
        def to_set(array):
            return set(tuple(elements) for elements in array)
        return float(len(to_set(pairs1).intersection(to_set(pairs2))))
    
    @staticmethod
    def _get_nonzero_cycles(pairs):
        all_indices_equal = np.sum(pairs[:, [0]] == pairs[:, 1:], axis=-1) == 3
        return np.sum(np.logical_not(all_indices_equal))
    
    def forward(self, distances1, distances2):
        """Return topological distance of two pairwise distance matrices.

        Args:
            distances1: Distance matrix in space 1
            distances2: Distance matrix in space 2

        Returns:
            distance, dict(additional outputs)
        """
        pairs1 = self._get_pairings(distances1)
        pairs2 = self._get_pairings(distances2)

        distance_components = {
            'metrics.matched_pairs_0D': self._count_matching_pairs(
                pairs1[0], pairs2[0])
        }
        # Also count matched cycles if present
        if self.use_cycles:
            distance_components['metrics.matched_pairs_1D'] = \
                self._count_matching_pairs(pairs1[1], pairs2[1])
            nonzero_cycles_1 = self._get_nonzero_cycles(pairs1[1])
            nonzero_cycles_2 = self._get_nonzero_cycles(pairs2[1])
            distance_components['metrics.non_zero_cycles_1'] = nonzero_cycles_1
            distance_components['metrics.non_zero_cycles_2'] = nonzero_cycles_2

        if self.match_edges is None:
            sig1 = self._select_distances_from_pairs(distances1, pairs1)
            sig2 = self._select_distances_from_pairs(distances2, pairs2)
            distance = self.sig_error(sig1, sig2)

        elif self.match_edges == 'symmetric':
            sig1 = self._select_distances_from_pairs(distances1, pairs1)
            sig2 = self._select_distances_from_pairs(distances2, pairs2)
            sig1_2 = self._select_distances_from_pairs(distances2, pairs1)
            sig2_1 = self._select_distances_from_pairs(distances1, pairs2)

            distance1_2 = self.sig_error(sig1, sig1_2)
            distance2_1 = self.sig_error(sig2, sig2_1)
            

            distance_components['metrics.distance1-2'] = distance1_2
            distance_components['metrics.distance2-1'] = distance2_1

            distance = distance1_2 + distance2_1

        elif self.match_edges == 'random':
            n_instances = len(pairs1[0])
            pairs1 = torch.cat([
                torch.randperm(n_instances)[:, None],
                torch.randperm(n_instances)[:, None]
            ], dim=1)
            pairs2 = torch.cat([
                torch.randperm(n_instances)[:, None],
                torch.randperm(n_instances)[:, None]
            ], dim=1)

            sig1_1 = self._select_distances_from_pairs(
                distances1, (pairs1, None))
            sig1_2 = self._select_distances_from_pairs(
                distances2, (pairs1, None))

            sig2_2 = self._select_distances_from_pairs(
                distances2, (pairs2, None))
            sig2_1 = self._select_distances_from_pairs(
                distances1, (pairs2, None))

            distance1_2 = self.sig_error(sig1_1, sig1_2)
            distance2_1 = self.sig_error(sig2_1, sig2_2)
            distance_components['metrics.distance1-2'] = distance1_2
            distance_components['metrics.distance2-1'] = distance2_1

            distance = distance1_2 + distance2_1

        return distance, distance_components
    

    # Topo Reg MIL model
class CV_TopoRegMIL(CV_MIL):
    def __init__(
        self, 
        init: str = "xavier_normal_", 
        pooling: str = "mean",
        lam: typing.Optional[float] = None,
        classification_acc: float = 0.7,
        class_count: int = 9,
        in_dim: int = 32,
        in_channel: int = 3,
        scheduler_class: typing.Optional[
            typing.Union[ConstantScheduler, typing.Callable[[], float]]
        ] = None,
        scheduler_topo: typing.Optional[
            typing.Union[ConstantScheduler, typing.Callable[[], float]]
        ] = None,
        distance = "euclidean"
    ):
        super().__init__(init=init, pooling=pooling, class_count=class_count, in_dim=in_dim, in_channel=in_channel)

        self.latent_norm = torch.nn.Parameter(data=torch.ones(1), requires_grad=True)
        self.topo_sig = TopologicalSignatureDistance(match_edges='symmetric')
        self.topo_loss = SignatureLoss(p=2)
        self.distance = distance
        self.lam: typing.Optional[float] = lam
        self.lam_class: typing.Optional[float] = None
        self.lam_topo: typing.Optional[float] = None
        self.classification_acc = classification_acc
        self.scheduler_class = scheduler_class
        self.scheduler_topo = scheduler_topo

        if self.distance == "perceptual":
            self.perceptual_distance_calculator = PerceptualLoss(device='cuda:0')
        elif self.distance == "cubical_complex":
            self.cubical_complex_calculator = CubicalComplexImageDistanceFunction()
        elif self.distance == "random_convolutions":
            self.random_convolution_distance_calculator = RandomProjectionModel(input_dim=[1,28,28]) #FOR MNIST
        elif self.distance == "reconstruction":
            self.rec_auto_enc = ReconstructionProjectionModel(path_to_model= "models/topology_models/reconstruction_distance_parameters/MNIST_Reconstruction_model.pth")


        if hasattr(self.scheduler_class, "constraint_bound") and (
            self.scheduler_class.constraint_bound == None
        ):
            self.scheduler_class.constraint_bound = -torch.log(torch.tensor(self.classification_acc, dtype=torch.float32))

        self.accuracy = Accuracy(task='multiclass', num_classes=class_count)

    
    def _compute_input_distance_matrix(self,x):

        if self.distance == "euclidean":
            distances = self._compute_euclidean_distance(x)            
        elif self.distance == "perceptual":
            distances = self._compute_perceptual_input_distances(x)
        elif self.distance == "random_convolutions":
            distances = self._compute_random_convolution_input_distances(x.squeeze(dim=0))
        elif self.distance == "reconstruction":
            distances = self._compute_reconstruction_distance(x)
        elif self.distance == "cubical_complex":
            distances = self._compute_cubical_distance(x)
        return distances

    def _compute_cubical_distance(self,x):
        return self.cubical_complex_calculator(x)

    
    def _compute_reconstruction_distance(self, x):
        latent = self.rec_auto_enc.get_latent_code_for_input(x)
        #x_flat = latent.view(x.size(0), -1)
        distances = torch.norm(latent[:, None] - latent, dim=2, p=2)
        return distances
    
    def _compute_euclidean_distance(self, x):
        x_flat = x.view(x.size(0), -1)
        distances = torch.norm(x_flat[:, None] - x_flat, dim=2, p=2)
        return distances
    #dim 0 of x needs to be bag size
    def _compute_perceptual_input_distances(self,x):
        distance_matrix = self.perceptual_distance_calculator(x)
        return distance_matrix
    
    def _compute_random_convolution_input_distances(self,x):
        rndm_convs = self.random_convolution_distance_calculator(x)
        return self._compute_euclidean_distance(rndm_convs)
    
    def forward(self, x, bag_label) -> ToporegMILOutput:
        """Compute the loss of the Topologically regularized mil.

        Args:
            x: Input image
            bag_label: label of the bag

        Returns:
            TopoMILOutput

        """
        dimensions = x.size()
        bag_size = dimensions[0]

        ins_features = self.ins_encode(x)
        ins_features = ins_features.view(x.shape[0], -1)
        ins_latent = self.activation_fn(self.fc(ins_features))
        if self.pooling == "att":
            ins_attention = self.ins_attention(ins_latent)
            bag_latent = self.pool(a=ins_attention, z=ins_latent)
        else:
            ins_attention = None
            bag_latent = self.pool(ins_latent)
        bag_prediction = self.classifier_head(bag_latent)


        img_distances = self._compute_input_distance_matrix(x)
   
        latent_distances = self._compute_euclidean_distance(ins_latent)
        latent_distances = latent_distances / self.latent_norm

        # print(f"Gradient of CubComplex image side {img_distances.grad_fn}, it requires grad {img_distances.requires_grad}")
        # print(f"Gradient of Euclidean Distance {latent_distances.grad_fn}, it requires grad {latent_distances.requires_grad}")
        # print(f"Gradient of image encoder {ins_latent.grad_fn}, it requires grad {ins_latent.requires_grad}")
        return ToporegMILOutput(
            ins_img = x.squeeze(dim=0),
            ins_latent = ins_latent,
            ins_attention=ins_attention,
            ins_prediction = None,
            bag_latent = bag_latent, 
            bag_label = bag_label,
            bag_prediction = bag_prediction,
            bag_size= bag_size,
            img_distances = img_distances,
            latent_distances = latent_distances,
        )
    
    def loss_function(
            self, 
            topomil_output: ToporegMILOutput, 
            is_training: bool = None,
            epoch_num: int = None
            ) -> typing.Dict[str, torch.Tensor]:
        mil_loss = self.mil_loss_function0(topomil_output, is_training=is_training)['loss']

        classification_metric = self.accuracy(torch.argmax(topomil_output.bag_prediction).unsqueeze(dim=0), topomil_output.bag_label)
        if is_training or (self.lam_class is None):
            self.lam_class = self.scheduler_class(float(mil_loss))
        if(topomil_output.bag_size!=1):  
            topo_loss, topo_loss_components = self.topo_sig(
                topomil_output.img_distances, topomil_output.latent_distances)
            
            topo_loss = topo_loss / float(topomil_output.bag_size)

            if is_training or (self.lam_topo is None):
                self.lam_topo = self.scheduler_topo(float(topo_loss))

            loss = self.lam_class * mil_loss + self.lam_topo * topo_loss
        else:
            loss= self.lam_class * mil_loss

        output = {
            "loss": loss,
            "lam_class": self.lam_class,
            "mil_loss": mil_loss,
            "classification_accuracy": classification_metric,
            "lam_topo": self.lam_topo,
            "topo_loss": topo_loss,
            "topo_loss_components": topo_loss_components,
        }
        return output
