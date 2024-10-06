from models.encoder_models.encoder_model import get_input_encoder
from models.salome_models.topo_reg_mil import TopologicalSignatureDistance
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models



class TopoAMiL(nn.Module):
    def __init__(self, class_count, multicolumn, device,input_type = "RGB_image",label_smoothing = 0.0,lam_topo = 1.0,max_lam = 0.1,lam_topo_per_epoch_decay = 1.0,pretrained_encoder = False,dropout_encoder = None):
        '''Initialize model. Takes in parameters:
        - class_count: int, amount of classes --> relevant for output vector
        - multicolumn: boolean. Defines if multiple attention vectors should be used.
        - device: either 'cuda:0' or the corresponding cpu counterpart.
        '''
        super(TopoAMiL, self).__init__()
        # condense every image into self.L features (further encoding before
        # actual MIL starts)
        self.L = 500
        self.D = 128                    # hidden layer size for attention network
        self.input_type =input_type
        self.lam_topo = lam_topo
        self.lam_topo_per_epoch_decay = lam_topo_per_epoch_decay
        self.max_lam = max_lam
        self.class_count = class_count
        self.label_smoothing = label_smoothing
        self.multicolumn = multicolumn
        self.device = device
        self.cross_entropy_loss = nn.CrossEntropyLoss(label_smoothing = label_smoothing)
        self.topo_sig = TopologicalSignatureDistance(match_edges='symmetric')

        # feature extractor before multiple instance learning starts
        
        self.ftr_proc = get_input_encoder(model = self,input_type=input_type,pretrained=pretrained_encoder,dropout=dropout_encoder)#self.get_encoder_architecture(input_type=input_type)
        self.to(device)


        # Networks for single attention approach
        # attention network (single attention approach)
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, 1)
        )
        # classifier (single attention approach)
        self.classifier = nn.Sequential(
            nn.Linear(self.L, 64),
            nn.ReLU(),
            nn.Linear(64, self.class_count)
        )

        # Networks for multi attention approach
        # attention network (multi attention approach)
        self.attention_multi_column = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.class_count),
        )
        # classifier (multi attention approach)
        self.classifier_multi_column = nn.ModuleList()
        for a in range(self.class_count):
            self.classifier_multi_column.append(nn.Sequential(
                nn.Linear(self.L, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            ))
            
    def remove_smoothing(self):
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def mil_loss_function(self,prediction,label_groundtruth):
        loss = self.cross_entropy_loss(prediction,label_groundtruth.unsqueeze(0))
        label_prediction = torch.argmax(prediction, dim=1).item()
        return {"mil_loss":loss,"correct":int(label_groundtruth==label_prediction)
                ,"prediction":prediction.cpu(),"label":label_groundtruth.item(),"prediction_int": label_prediction}
    
    def topo_loss_function(self,latent_distance_matrix,input_distance_matrix):
        x= self.ensure_square_tensor(input_distance_matrix)
        y= self.ensure_square_tensor(latent_distance_matrix)
        assert x.shape == y.shape
        topo_loss, topo_loss_components = self.topo_sig(x, y)
        
        topo_loss = topo_loss / float(latent_distance_matrix.shape[0])
        return {"topo_loss":topo_loss,"topo_loss_components": topo_loss_components}

    def ensure_square_tensor(self,tensor):
        # Check if tensor is 1 x n x n
        if tensor.dim() == 3 and tensor.shape[0] == 1:
            # Remove the first dimension to make it n x n
            return tensor.squeeze(0)
        # Check if tensor is n x n
        elif tensor.dim() == 2 and tensor.shape[0] == tensor.shape[1]:
            return tensor
        else:
            raise ValueError("Input tensor must be of shape n x n or 1 x n x n.")

    def _compute_euclidean_distance(self, x):
        x_flat = x.view(x.size(0), -1)
        distances = torch.norm(x_flat[:, None] - x_flat, dim=2, p=2)
        return distances
    def forward(self, x):
        '''Forward pass of bag x through network. '''
        if x.shape[0] == 1:
            x = x.squeeze(0)
        ft = self.ftr_proc(x)
        latent_space_dist_matrix = self._compute_euclidean_distance(ft)
        # switch between multi- and single attention classification
        if(self.multicolumn):
            att_raw = self.attention_multi_column(ft)
            att_raw = torch.transpose(att_raw, 1, 0)
            bag_feature_stack = torch.empty((self.class_count, ft.size(1)), device=ft.device, dtype=ft.dtype)
            prediction = torch.empty((1, self.class_count), device=ft.device, dtype=ft.dtype)
            # for every possible class, repeat
            for a in range(self.class_count):
                # softmax + Matrix multiplication
                att_softmax = F.softmax(att_raw[a, ...][None, ...], dim=1)
                bag_features = torch.mm(att_softmax, ft)

                # Store bag features directly
                bag_feature_stack[a, :] = bag_features.squeeze(0)

                # Final classification with one output value (value indicating
                # this specific class to be predicted)
                prediction[0, a] = self.classifier_multi_column[a](bag_features).squeeze()


            return prediction, latent_space_dist_matrix
        #att_raw, F.softmax(att_raw, dim=1), bag_feature_stack,
        else:
            # calculate attention
            att_raw = self.attention(ft)
            att_raw = torch.transpose(att_raw, 1, 0)
            # Softmax + Matrix multiplication
            att_softmax = F.softmax(att_raw, dim=1)
            bag_features = torch.mm(att_softmax, ft)
            # final classification
            prediction = self.classifier(bag_features)
            return prediction,latent_space_dist_matrix
            # att_raw, att_softmax, bag_features,
        
        
    def get_encoder_architecture(self,input_type = "images",pretrained = False):
        encoder_output_dim = 500
        
        if input_type == "images": #this and the one after should be the same, since Hehr supposedly used the same network on the same 3x144x144 input 
            FT_DIM_IN = 512
            if pretrained:
                resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            else:
                resnet18 = models.resnet18()
                
            encoder = nn.Sequential(*list(resnet18.children())[:-2],
                nn.Conv2d(FT_DIM_IN, int(FT_DIM_IN * 1.5), kernel_size=2),
                nn.ReLU(),
                nn.Conv2d(int(FT_DIM_IN * 1.5), int(FT_DIM_IN * 2), kernel_size=2),
                nn.ReLU(),
                nn.AdaptiveMaxPool2d(output_size=(1, 1)),
                nn.Flatten(),
                nn.Linear(int(FT_DIM_IN * 2), encoder_output_dim),
                nn.ReLU(),
            )
        elif input_type == "gray_images": #this and the one after should be the same, since Hehr supposedly used the same network on the same 3x144x144 input 
            FT_DIM_IN = 512
            if pretrained:
                resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            else:
                resnet18 = models.resnet18()
            resnet18.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=resnet18.conv1.out_channels, 
            kernel_size=resnet18.conv1.kernel_size, 
            stride=resnet18.conv1.stride, 
            padding=resnet18.conv1.padding, 
            bias=resnet18.conv1.bias
            )

            # If using pretrained weights, adjust them to work with the new layer
            # The weights from the original 3 channels can be averaged to initialize the 1 channel.
            with torch.no_grad():
                resnet18.conv1.weight[:, 0, :, :] = resnet18.conv1.weight.mean(dim=1)

            encoder = nn.Sequential(*list(resnet18.children())[:-2],
                nn.Conv2d(FT_DIM_IN, int(FT_DIM_IN * 1.5), kernel_size=2),
                nn.ReLU(),
                nn.Conv2d(int(FT_DIM_IN * 1.5), int(FT_DIM_IN * 2), kernel_size=2),
                nn.ReLU(),
                nn.AdaptiveMaxPool2d(output_size=(1, 1)),
                nn.Flatten(),
                nn.Linear(int(FT_DIM_IN * 2), encoder_output_dim),
                nn.ReLU(),
            )
        elif input_type == "fnl34": #this and the one after should be the same, since Hehr supposedly used the same network on the same 3x144x144 input 
            FT_DIM_IN = 512            
            encoder = nn.Sequential(
                nn.Conv2d(FT_DIM_IN, int(FT_DIM_IN * 1.5), kernel_size=2),
                nn.ReLU(),
                nn.Conv2d(int(FT_DIM_IN * 1.5), int(FT_DIM_IN * 2), kernel_size=2),
                nn.ReLU(),
                nn.AdaptiveMaxPool2d(output_size=(1, 1)),
                nn.Flatten(),
                nn.Linear(int(FT_DIM_IN * 2), encoder_output_dim),
                nn.ReLU(),
            )
        elif input_type == "dino_bloom_small":
            FT_DIM_IN = 384
            encoder = nn.Sequential()
            self.L = FT_DIM_IN
        else:
            print(input_type)
            raise Exception("Input type not supported")
        return encoder