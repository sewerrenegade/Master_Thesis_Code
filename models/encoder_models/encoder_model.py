import torch.nn as nn
import torchvision.models as models
import torch

input_types = ("images","gray_images","fnl34","dino_bloom_small")
renset_types = [18,34,50]

def get_resnet_version(name):
    if name == "images" or name == "images18":
        return 18
    if name == "images34":
        return 34
    if name == "images50":
        return 50

def get_loss_function(function_name = "cross_entropy"):
    pass
def get_input_encoder(model,input_type = "images",pretrained = False, dropout = None,load_encoder_path = None):
    encoder_output_dim = 500
    if "images" in input_type and not ("gray" in input_type): #this and the one after should be the same, since Hehr supposedly used the same network on the same 3x144x144 input 
        FT_DIM_IN = 512
        RESNET = get_resnet_version(input_type)
        if RESNET == 18:
            FT_DIM_IN = 512
            if pretrained:
                res18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            else:
                res18 = models.resnet18()
            res18 = list(res18.children())[:- 2]#-2
            if dropout is not None and dropout != 0:
                res18 = [res18[0],res18[1],res18[2],res18[3],res18[4],nn.Dropout(dropout),res18[5],nn.Dropout(dropout),res18[6],nn.Dropout(dropout),res18[7],nn.Dropout(dropout)]#
            resnet = res18
        elif RESNET == 34:
            FT_DIM_IN = 512
            if pretrained:
                resnet34 = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            else:
                resnet34 = models.resnet34()

            res34 = list(resnet34.children())[:-2]

            # Add dropout layers, if specified
            if dropout is not None and dropout != 0:
                res34 = [res34[0], res34[1], res34[2], res34[3], res34[4],
                        nn.Dropout(dropout), res34[5], nn.Dropout(dropout), res34[6], nn.Dropout(dropout), res34[7], nn.Dropout(dropout)]
            resnet = res34
        elif RESNET == 50:
            FT_DIM_IN = 2048
            if pretrained:
                resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            else:
                resnet50 = models.resnet50()
            res50 = list(resnet50.children())[:-2]

            # Add dropout layers, if specified
            if dropout is not None and dropout != 0:
                res50 = [res50[0], res50[1], res50[2], res50[3], res50[4],
                        nn.Dropout(dropout), res50[5], nn.Dropout(dropout), res50[6], nn.Dropout(dropout), res50[7], nn.Dropout(dropout)]
            resnet = res50
            


        encoder = nn.Sequential(*resnet,
            nn.Conv2d(FT_DIM_IN, int(FT_DIM_IN * 0.5), kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(int(FT_DIM_IN * 0.5), int(FT_DIM_IN * 0.25), kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(int(FT_DIM_IN * 0.25), encoder_output_dim),
            nn.ReLU(),
        )
    elif input_type == "gray_images": #this and the one after should be the same, since Hehr supposedly used the same network on the same 3x144x144 input 
        FT_DIM_IN = 512
        if pretrained:
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            resnet = models.resnet18()
        resnet.conv1 = nn.Conv2d(
        in_channels=1, 
        out_channels=resnet.conv1.out_channels, 
        kernel_size=resnet.conv1.kernel_size, 
        stride=resnet.conv1.stride, 
        padding=resnet.conv1.padding, 
        bias=resnet.conv1.bias
        )

        # If using pretrained weights, adjust them to work with the new layer
        # The weights from the original 3 channels can be averaged to initialize the 1 channel.
        with torch.no_grad():
            resnet.conv1.weight[:, 0, :, :] = resnet.conv1.weight.mean(dim=1)
        res18 = list(resnet.children())[:-2]
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

    if load_encoder_path is not None:
        import pytorch_lightning as pl
        checkpoint = torch.load(load_encoder_path, map_location=torch.device('cpu'))
        encoder_state_dict = {k.replace("model.ftr_proc.", ""): v for k, v in checkpoint['state_dict'].items() if k.startswith("model.ftr_proc.")}
        missing_keys, unexpected_keys = encoder.load_state_dict(encoder_state_dict, strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")       
        # for param in encoder.parameters():
        #     param.requires_grad = False
    return encoder
