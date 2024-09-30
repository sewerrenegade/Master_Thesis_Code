import torch.nn as nn
import torchvision.models as models
import torch

input_types = ("images","gray_images","fnl34","dino_bloom_small")

def get_loss_function(function_name = "cross_entropy"):
    pass

def get_input_encoder(model,input_type = "images",pretrained = False, dropout = None):
    encoder_output_dim = 500
    
    if input_type == "images": #this and the one after should be the same, since Hehr supposedly used the same network on the same 3x144x144 input 
        FT_DIM_IN = 512
        if pretrained:
            resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            resnet18 = models.resnet18()
            
        res18 = list(resnet18.children())[:-2]
        if dropout is not None and dropout != 0:
            res18 = [res18[0],res18[1],res18[2],res18[3],res18[4],nn.Dropout(dropout),res18[5],nn.Dropout(dropout),res18[6],nn.Dropout(dropout),res18[7],nn.Dropout(dropout)]
 
        encoder = nn.Sequential(*res18,
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
        res18 = list(resnet18.children())[:-2]
        if dropout is not None and dropout != 0:
            res18 = [res18[0],res18[1],res18[2],res18[3],res18[4],nn.Dropout(dropout),res18[5],nn.Dropout(dropout),res18[6],nn.Dropout(dropout),res18[7],nn.Dropout(dropout)]
 
        encoder = nn.Sequential(*res18,
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
        model.L = FT_DIM_IN
    else:
        print(input_type)
        raise Exception("Input type not supported")
    return encoder
