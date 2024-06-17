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
    transforms.Normalize(mean=DINOBLOOM_DEFAULT_STD, std=DINOBLOOM_DEFAULT_MEAN),
])
DEFAULT_PATCH_NUM=16
DEFAULT_IMAGE_DIM=224
eval_model="dinov2_vits14"

def get_path_to_weights(network_weights_name):
    return f"models/DinoBloom/weights/{network_weights_name}"

def get_dino_bloom(size = "small"):
    # load the original DINOv2 model with the correct architecture and parameters.
    network_infos= DINOBLOOM_NETWORKS_INFOS[size]
    model=torch.hub.load('facebookresearch/dinov2', network_infos["model_name"])
    # load finetuned weights
    pretrained = torch.load(get_path_to_weights(network_infos["weights_filename"]), map_location=torch.device('cpu'))
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

# model=get_dino_bloom()


# #data input
# paths=np.random.choice(list(Path("/content/segmentation_WBC/Dataset 2").glob("*.bmp")),4)
# images_for_plotting = [Image.open(img_path).convert('RGB').resize((img_size, img_size)) for img_path in paths]

# if torch.cuda.is_available():
#   model.cuda()
#   imgs_tensor = torch.stack([transform(Image.open(img_path).convert('RGB').resize((img_size,img_size))).cuda() for img_path in paths])
# else:
#   imgs_tensor = torch.stack([transform(Image.open(img_path).convert('RGB').resize((img_size,img_size)))for img_path in paths])


# data_size = 4

# SCEMILA_images_cpu = SCEMILAimage_base(gpu=False,to_tensor=False)
# images_for_plotting = [SCEMILA_images_cpu[index][0].convert('RGB').resize((DEFAULT_IMAGE_DIM, DEFAULT_IMAGE_DIM)) for index in range(data_size)]
# imgs_tensor = torch.stack([DINOBLOOM_TRANSFORMS(SCEMILA_images_cpu[index][0].convert('RGB').resize((DEFAULT_IMAGE_DIM,DEFAULT_IMAGE_DIM))).cuda() for index in range(data_size)])
# model.cuda()
# #data input
# with torch.no_grad():
#     # Ensure the input tensor is on GPU by calling .cuda() on it
#     features_dict = model.forward_features(imgs_tensor)
#     print(list(features_dict.keys()))
#     features = features_dict['x_norm_patchtokens']
#     image_features = features_dict["x_norm_clstoken"]
# features = features.reshape(data_size * DEFAULT_PATCH_NUM*DEFAULT_PATCH_NUM, embed_sizes[eval_model]).cpu().numpy()
# pca = PCA(n_components=3)
# pca.fit(features)
# pca_features = pca.transform(features)
# for i in range(3):
#     pca_features[:, i] = (pca_features[:, i] - pca_features[:, i].min()) / (pca_features[:, i].max() - pca_features[:, i].min())

# pca_features_rgb = pca_features.copy()
# pca_features_rgb = pca_features_rgb.reshape(data_size, DEFAULT_PATCH_NUM, DEFAULT_PATCH_NUM, 3)

# fig, axs = plt.subplots(2, 2, figsize=(10, 10))
# for i, ax in enumerate(axs.flat):
#     #print(Path(images_for_plotting[i]).stem)
#     ax.imshow(pca_features_rgb[i][..., ::-1])
#     ax.axis('off')  # Remove axis
# plt.savefig('features.png')
# plt.show()
# plt.close()
# fig, axs = plt.subplots(2, 2, figsize=(10, 10))
# for i, ax in enumerate(axs.flat):
#     #print(Path(images_for_plotting[i]).stem)
#     ax.imshow(images_for_plotting[i])
#     ax.axis('off')  # Remove axis
# plt.show()
# plt.close()