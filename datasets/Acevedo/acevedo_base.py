from datasets.base_dataset_abstraction import BaseDataset
from collections.abc import Iterable
import os
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms

class Acevedo_base(BaseDataset):

    def __init__(self,training_mode,encode_with_dino_bloom = False, root_dir = "data/",balance_dataset_classes = None,gpu = True, numpy = False,flatten = False,to_tensor = True,augmentation_settings = None):
        self.encode_with_dino_bloom = encode_with_dino_bloom
        self.root_dir = root_dir
        super().__init__("Acevedo",augmentation_settings=augmentation_settings,balance_dataset_classes=balance_dataset_classes)
        self.preload_transforms,self.transforms = self.get_transform_function(grayscale=False,load_jpg=True ,numpy=numpy,to_gpu=gpu,flatten=flatten,to_tensor=to_tensor,augmentation_settings= augmentation_settings,extra_transforms=[])
        self.path_to_data_folder = self.indexer.get_path_to_data_folder()
        self.supposed_dimension = (360,363)
        self.to_pil = transforms.ToPILImage()
        if self.encode_with_dino_bloom:
            self.dino_enc = self.get_dino_bloom_transform()
            self.get_item_function = self.get_and_dino_encoded_tif_image
        else:
            self.get_item_function = self.get_single_jpg_image
        self.loaded_data = {}
        
    def fix_irregular_dimensions(self,image):
        if not isinstance(image, Image.Image):
            # Convert NumPy array to PIL Image
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            # Convert PyTorch tensor to PIL Image
            elif isinstance(image, torch.Tensor):
                # Check if tensor is on GPU, if so move it to CPU
                if image.is_cuda:
                    image = image.cpu()
                image = self.to_pil(image)
            else:
                raise TypeError("Input must be a PIL Image, a NumPy array, or a PyTorch tensor.")
        return image.resize(self.supposed_dimension)
    
    def __len__(self):
        return len(self.indicies_list)

    def __getitem__(self, idx):
        '''returns specific item from this dataset'''
        if type(idx) is int:
            return self.get_item_function(idx)
        elif isinstance(idx,Iterable):
            images = []
            labels = []
            for index in idx:
                image , label = self.get_item_function(index)
                images.append(self.transforms(image))
                labels.append(label)
            return images,labels
        
    def get_single_jpg_image(self, idx):
        if idx not in self.loaded_data:
            image_path = self.indicies_list[idx][0]
            image_path = os.path.join(self.path_to_data_folder,image_path)
            image = self.transforms(self.preload_transforms(image_path))
            self.loaded_data[idx] =  image,self.indexer.convert_label_from_int_to_str_or_viceversa(self.indicies_list[idx][1])
        return self.loaded_data[idx]

    
    def get_and_dino_encoded_tif_image(self,idx):
        if idx not in self.loaded_data:
            image_path = self.indicies_list[idx][0]
            image_path = os.path.join(self.path_to_data_folder,image_path)
            image = self.preload_transforms(image_path)
            dino_features = self.dino_enc(image)
            image_feaures = self.transforms(dino_features[0].cpu().detach())
            self.loaded_data[idx] = image_feaures,self.indexer.convert_label_from_int_to_str_or_viceversa(self.indicies_list[idx][1])
        return self.loaded_data[idx]