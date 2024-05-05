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
    
