import typing
from dataclasses import dataclass
import torch
from torchmetrics import Accuracy

from models.salome_models.scheduler import  ConstantScheduler

from models.salome_models.mil import CV_MIL,MILOutput
from distance_functions.functions.perceptual_lpsis_distance import PerceptualLoss
from distance_functions.functions.cubical_complex_distance import CubicalComplexImageDistanceFunction
from distance_functions.functions.random_convolutions_distance import RandomProjectionModel
from distance_functions.functions.reconstruction_functions.classic_ae_distance.ae_reconstruction_distance import ReconstructionProjectionModel
from models.topology_models.topo_tools import SignatureLoss, VietorisRipsComplex, WassersteinDistance
from models.topology_models.topo_tools.moor_topo_reg import TopologicalSignatureDistance




@dataclass
class ToporegMILOutput(MILOutput):
    ins_img: torch.Tensor
    bag_size: int
    img_distances: torch.Tensor
    latent_distances: torch.Tensor    



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
