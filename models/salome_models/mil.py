import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import typing


@dataclass
class MILOutput:
    ins_latent: torch.Tensor
    ins_attention: torch.Tensor
    ins_prediction: torch.Tensor
    bag_latent: torch.Tensor
    bag_label: int
    bag_prediction: torch.Tensor
    ins_attention: torch.Tensor
    ins_prediction: torch.Tensor



class CV_MIL(nn.Module): 
    def __init__(self, 
                 init: str = "xavier_normal_", 
                 pooling: str = "mean", 
                 class_count: int = 10, 
                 in_dim: int = 28,
                 in_channel: int = 1,
                 aux_loss: bool = False,
                 hidden_dims:list[int] = [20, 50]
                 ):
        super().__init__()
        self.pooling = pooling
        self.class_count = class_count
        self.latent_dim = 500 # 128
        self.param_initialisation = init
        self.aux = aux_loss
        self.in_dim = in_dim
        self.att_ldim = 128 # 64
        self.sic_ldim = 32
        self.in_channels = in_channel
        self.hidden_dims = hidden_dims
        self.activation_fn = nn.ReLU()

        self.sic_classifier = self.build_instance_classifier()
        self.attention = self.build_attention_head()
        self.ins_encoder = self.build_instance_encoder()
        self.fc = self.build_fully_connected_layer()
        self.bag_classifier = self.build_bag_classifier()
        self.build_loss_functions()
        self.initialize_parameters()

            
    def forward(self, x, bag_label) -> MILOutput:
        """Apply MIL on input bag of instances. 

        Args:
            x: Bag of instances with shape [num_instances x channels x n_row x n_col]
            bag_label: label of the bag

        Returns:
            MILOutput
        """
        instance_features = self.ins_encode(x)
        instance_vectors = instance_features.view(x.shape[0], -1)
        instance_latent = self.activation_fn(self.fc(instance_vectors))
        ins_prediction = self.sic_classifier_head(instance_latent)
        if self.pooling == "att":
            instance_attention = self.ins_attention(instance_latent)
            bag_latent = self.pool(a=instance_attention, z=instance_latent)
        else:
            instance_attention = None
            bag_latent = self.pool(instance_latent)
        prediction = self.classifier_head(bag_latent)
        return MILOutput(
            ins_latent = instance_latent,
            ins_attention = instance_attention,
            ins_prediction = ins_prediction,
            bag_latent = bag_latent, 
            bag_label = bag_label,
            bag_prediction = prediction,
        )
    



    def build_loss_functions(self):
        if self.class_count ==2:
            self.mil_ce_loss = nn.BCEWithLogitsLoss()
            self.sic_ce_loss = nn.BCEWithLogitsLoss()
        else:
            self.mil_ce_loss = nn.CrossEntropyLoss()
            self.sic_ce_loss = nn.CrossEntropyLoss()

    def initialize_parameters(self):
        for p in self.ins_encoder.parameters():
            if p.dim() > 1:
                eval("nn.init.%s(p)" % self.param_initialisation)
        for p in self.bag_classifier.parameters():
            if p.dim() > 1:
                eval("nn.init.%s(p)" % self.param_initialisation)
    
    def build_bag_classifier(self):
        return nn.Sequential(
            nn.Linear(self.latent_dim, self.class_count),
        )
    
    def build_fully_connected_layer(self):
        ## calculate output dim -> each conv:(I-(ks-2))/s, pool:(I-ks)/s
        ## 28x28 -> 5x5, 32x32 -> 6
        if self.in_dim == 28:
            return nn.Linear(self.hidden_dims[-1] * 4 * 4, self.latent_dim) # shallow model :nn.Linear(hidden_dims[-1] * 5 * 5, self.latent_dim)
        elif self.in_dim == 32:
            return nn.Linear(self.hidden_dims[-1] * 6 * 6, self.latent_dim)

    
    def build_attention_head(self):
            return nn.Sequential(
            nn.Linear(self.latent_dim, self.att_ldim),
            nn.Tanh(),
            nn.Linear(self.att_ldim, 1)
        )


    def build_instance_encoder(self):
        modules = []
        in_channels = self.in_channels
        for h_dim in self.hidden_dims:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, h_dim, kernel_size=5, stride=1), # shallow model: kernel_size=5
                self.activation_fn,
                nn.MaxPool2d(kernel_size=2, stride=2),
                ))
            in_channels  = h_dim
        return nn.Sequential(*modules)
    
    
    def build_instance_classifier(self):
        return nn.Sequential(
            nn.Linear(self.latent_dim, self.sic_ldim),
            nn.Linear(self.sic_ldim, self.class_count)
        )

    def ins_encode(self, x) -> torch.Tensor:
        if len(x.shape)==3:
            x = x.unsqueeze(dim=1)
        f = self.ins_encoder(x)
        return f
    
    def ins_attention(self, z) -> torch.Tensor:
        a = self.attention(z)
        a = torch.transpose(a, 1, 0)
        a = F.softmax(a, dim=1)
        return a
    
    def sic_classifier_head(self, z) -> torch.Tensor:
        return self.sic_classifier(z)
    
    def pool(self, z,  a=None) -> torch.Tensor:
        if self.pooling == "mean":
            p = F.avg_pool1d(z.T, kernel_size=z.shape[0]).T
        elif self.pooling == "max":
            p = F.max_pool1d(z.T, kernel_size=z.shape[0]).T
        elif self.pooling == "att":
            if a is None:
                raise RuntimeError("attention is not passed to pooling.")
            p = torch.mm(a, z)
        else:
            raise RuntimeError("pooling technique is not supported.")
        return p

    def classifier_head(self, z) -> torch.Tensor:
        return self.bag_classifier(z)

    

    def mil_loss_function0(
            self, mil_out: MILOutput, 
            is_training: bool = None, 
            epoch_num: int = None,
            ) -> typing.Dict[str, torch.Tensor]:
        # if self.class_count == 2:
        device = mil_out.bag_prediction.device
        labels = torch.eye(self.class_count)[mil_out.bag_label.item()].unsqueeze(dim=0).to(device)
        mil_loss = self.mil_ce_loss(mil_out.bag_prediction, labels)
        # else:
        #     mil_loss = self.mil_ce_loss(mil_out.bag_prediction, mil_out.bag_label)
        if self.aux:
            ins_noisy_label = torch.eye(self.class_count)[mil_out.bag_label.item()].unsqueeze(dim=0).to(device).repeat((mil_out.ins_prediction.shape[0], 1)).squeeze()
            # ins_noisy_label = mil_out.bag_label.repeat((mil_out.ins_prediction.shape[0], 1)).squeeze()
            sic_loss = self.sic_ce_loss(mil_out.ins_prediction, ins_noisy_label)
            sic_ratio = torch.tensor(1 * (1 - .05) ** epoch_num)
            loss = (1 - sic_ratio) * mil_loss + sic_ratio * sic_loss
            output = {
                "loss": loss,
                "mil_att_loss": mil_loss,
                "sic_att_loss": sic_loss,
                "sic_ratio": sic_ratio,
                } 
        else:
            output = {"loss": mil_loss}

        return output
    

