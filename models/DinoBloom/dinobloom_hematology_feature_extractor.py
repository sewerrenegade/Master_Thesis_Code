import sys
sys.path.append('/home/milad/Desktop/Master_Thesis/code/Master_Thesis_Code')
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
# PCA for feature inferred
from sklearn.decomposition import PCA
from skimage.filters import threshold_otsu
import os
import random
from glob import glob
#from datasets.SCEMILA.base_SCEMILA import SCEMILAimage_base

DINOBLOOM_DEFAULT_MEAN = (0.485, 0.456, 0.406)
DINOBLOOM_DEFAULT_STD = (0.229, 0.224, 0.225)
DINOBLOOM_NETWORKS_INFOS = {"small":{"out_dim":384,"weights_filename":"DinoBloom-S.pth","model_name":"dinov2_vits14"},
                 "big":{"out_dim":768,"weights_filename":"DinoBloom-B.pth","model_name":"dinov2_vitb14"},
                 "large":{"out_dim":1024,"weights_filename":"DinoBloom-L.pth","model_name":"dinov2_vitl14"},}
DINOBLOOM_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=DINOBLOOM_DEFAULT_MEAN, std=DINOBLOOM_DEFAULT_STD),
])
DEFAULT_PATCH_NUM=16
DINOBLOOM_DEFAULT_IMAGE_DIM=224
eval_model="dinov2_vits14"

def get_path_to_weights(network_weights_name):
    return f"models/DinoBloom/weights/{network_weights_name}"

def get_dino_bloom(size = "small"):
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!GETTING FRESH DINO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # load the original DINOv2 model with the correct architecture and parameters.
    network_infos= DINOBLOOM_NETWORKS_INFOS[size]
    model=torch.hub.load('facebookresearch/dinov2', network_infos["model_name"])
    # load finetuned weights
    pretrained = torch.load(get_path_to_weights(network_infos["weights_filename"]), map_location=torch.device('cpu'),weights_only= True)
    # make correct state dict for loading
    new_state_dict = {}
    for key, value in pretrained['teacher'].items():
        if 'dino_head' in key or "ibot_head" in key:
            pass
        else:
            new_key = key.replace('backbone.', '')
            new_state_dict[new_key] = value

    #corresponds to 224x224 image. patch size=14x14 => 16*16 patches
    pos_embed = nn.Parameter(torch.zeros(1, 257, network_infos["out_dim"]))
    model.pos_embed = pos_embed
    model.load_state_dict(new_state_dict, strict=True)
    model.cuda()
    return model.forward_features