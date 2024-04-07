import torch.nn as nn
import torch.nn.functional as F
import torch


F_DIM = 512

class MIL_RGMIL(nn.Module):
    def __init__(self, pooling = "rgp", dataset="MNIST"):
        super().__init__()
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")  # If you have multiple GPUs, change the index accordingly
            print("CUDA is available. Moving the model to GPU.")
        else:
            self.device = torch.device("cpu")
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.classifiers = None
        self.backbone = Backbone(dataset)
        self.dataset = dataset
        self.linear = nn.Parameter(data=torch.FloatTensor(F_DIM, 2)).to(self.device)
        self.pooling = pooling

        nn.init.kaiming_uniform_(self.linear)
        self.softmax = nn.Softmax(dim=0)

        self.attentions = []
        self.ks = 1
        for i in range(self.ks):
            setattr(self, 'a{}'.format(i), nn.Sequential(
                nn.Linear(512, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            ))
            self.attentions.append(getattr(self, 'a{}'.format(i)))

    def ins_encode(self, x):
        if len(x.shape)==3:
            x = x.unsqueeze(dim=1)
        f = self.backbone(x)
        return f
    
    def ins_liniear(self, f):
        return super().ins_liniear(f)
    
    def classifier_head(self, z):
        return super().classifier_head(z)
    
    def pool(self, x,fs):
        if self.pooling == "rgp":
        # nsp
            bn = nn.LayerNorm(x.shape[0]).to(x.device)
            alpha = torch.mm(fs, self.linear)  # [t,ks]
            alpha = self.softmax(bn(alpha[:, 1] - alpha[:, 0]))
            F = torch.matmul(alpha, fs)  # [o]
           
            
        elif self.pooling == "att":
            alpha = torch.stack([a(fs).squeeze(1) for a in self.attentions], dim=0)  # [ks,t]
            alpha = torch.softmax(alpha, dim=1)  # softmax over t
            F = torch.mm(alpha, fs).view(-1)  # [ks,o]
        return F,alpha

    def forward(self, x):
        fs = self.backbone(x)  # [t,o]
        pooled,alpha = self.pool(x,fs)
        
        Y_logits = torch.matmul(pooled, self.linear)  # [ks]
        Y_hat = torch.argmax(Y_logits, dim=0)

        return Y_logits, Y_hat, alpha, pooled, alpha
    
    def mil_loss_function(self,y_pred,y_ground_truth):
        target = torch.zeros(2).to(y_pred.device)
        target[y_ground_truth[0]] = 1
        x=self.cross_entropy(y_pred,target)
        return {"loss":x}



class Backbone(nn.Module):
    
    def __init__(self, dataset="MNIST"):
        super(Backbone, self).__init__()

        self.stem = None

        dim_orig = 0
        if dataset == "MUSK1" or dataset == "MUSK2":
            dim_orig = 166
        elif dataset == "TIGER" or dataset == "FOX" or dataset == "ELEPHANT":
            dim_orig = 230
        

        if dataset == "MNIST":
            # input [t,c,h,w]
            self.stem = nn.Sequential(
                nn.Conv2d(1, 20, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(20, 32, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
                nn.AdaptiveAvgPool2d((4, 4))
                # [t,32,4,4]
            )
        elif(dataset == "TIGER" or dataset == "FOX" or dataset == "ELEPHANT" or dataset == "MUSK1" or dataset == "MUSK2"):
            self.stem = nn.Sequential(
                nn.Linear(dim_orig, F_DIM),
                nn.ReLU(),
                nn.Linear(F_DIM, F_DIM),
                nn.ReLU(),
                # [t,32,4,4]
            )
        

    def forward(self, x):
        x = self.stem(x)
        return x.view(x.shape[0], -1)  # [t,F_DIM]


    def calculate_objective(self, X, Y):
        Y0 = Y.squeeze().long()
        target = torch.zeros(2).to(X.device)
        target[Y0] = 1

        Y_logits, Y_hat, weights, feature, weight = self.forward(X)

        loss = torch.nn.CrossEntropyLoss()
        all_loss = loss(Y_logits, target)

        return all_loss, Y_hat, weight