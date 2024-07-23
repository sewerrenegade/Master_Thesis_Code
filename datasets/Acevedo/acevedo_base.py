import sys
sys.path.append('/home/milad/Desktop/Master_Thesis/code/Master_Thesis_Code')
from datasets.base_dataset_abstraction import BaseDataset
from collections.abc import Iterable
import os
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms

from datasets.data_synthesizers.data_sythesizer import SinglePresenceMILSynthesizer
from datasets.image_augmentor import AugmentationSettings
from datasets.mil_dataset_abstraction import BaseMILDataset

class Acevedo_base(BaseDataset):

    def __init__(self,training_mode,encode_with_dino_bloom = False, resize = False,root_dir = "data/",balance_dataset_classes = None,gpu = True, numpy = False,flatten = False,to_tensor = True,augmentation_settings = None,grayscale = False):
        self.encode_with_dino_bloom = encode_with_dino_bloom
        self.root_dir = root_dir
        super().__init__("Acevedo",augmentation_settings=augmentation_settings,balance_dataset_classes=balance_dataset_classes)
        self.preload_transforms,self.transforms = self.get_transform_function(grayscale=grayscale,resize = resize,load_jpg=True ,numpy=numpy,to_gpu=gpu,flatten=flatten,to_tensor=to_tensor,augmentation_settings= augmentation_settings,extra_transforms=[])
        self.path_to_data_folder = self.indexer.get_path_to_data_folder()
        self.supposed_dimension = (360,360)
        self.to_pil = transforms.ToPILImage()
        if self.encode_with_dino_bloom:
            self.dino_enc = self.get_dino_bloom_transform()
            self.get_item_function = self.get_and_dino_encoded_tif_image
        else:
            self.get_item_function = self.get_single_jpg_image
        self.loaded_data = {}
        

    def __len__(self):
        return len(self.indicies_list)
    
    def get_only_pretransform_item(self, idx):
        '''returns specific item from this dataset'''
        if type(idx) is int:
            return self.get_single_jpg_image_without_postrans(idx)
        elif isinstance(idx,Iterable):
            images = []
            labels = []
            for index in idx:
                image , label = self.get_single_jpg_image_without_postrans(index)
                images.append(image)
                labels.append(label)
            return images,labels

    def __getitem__(self, idx):
        '''returns specific item from this dataset'''
        if type(idx) is int:
            return self.get_item_function(idx)
        elif isinstance(idx,Iterable):
            images = []
            labels = []
            for index in idx:
                image , label = self.get_item_function(index)
                images.append(image)
                labels.append(label)
            return images,labels
        
    def get_single_jpg_image(self, idx):
        if idx not in self.loaded_data:
            image_path = self.indicies_list[idx][0]
            image_path = os.path.join(self.path_to_data_folder,image_path)
            image = self.transforms(self.preload_transforms(image_path))
            self.loaded_data[idx] =  image,self.indexer.convert_label_from_int_to_str_or_viceversa(self.indicies_list[idx][1])
        return self.loaded_data[idx]

    def get_single_jpg_image_without_postrans(self, idx):
        image_path = self.indicies_list[idx][0]
        image_path = os.path.join(self.path_to_data_folder,image_path)
        image = self.preload_transforms(image_path)
        return image,self.indexer.convert_label_from_int_to_str_or_viceversa(self.indicies_list[idx][1])
    
    def get_and_dino_encoded_tif_image(self,idx):
        if idx not in self.loaded_data:
            image_path = self.indicies_list[idx][0]
            image_path = os.path.join(self.path_to_data_folder,image_path)
            image = self.preload_transforms(image_path)
            dino_features = self.dino_enc(image)
            image_feaures = self.transforms(dino_features[0].cpu().detach())
            self.loaded_data[idx] = image_feaures,self.indexer.convert_label_from_int_to_str_or_viceversa(self.indicies_list[idx][1])
        return self.loaded_data[idx]
    
    
class Acevedo_MIL_base(BaseMILDataset):
    def __init__(self,training_mode,data_synth,encode_with_dino_bloom=False, root_dir = "data/",balance_dataset_classes = 100,gpu = True, grayscale= False, numpy = False,flatten = False,to_tensor = True,augmentation_settings = None):
        self.encode_with_dino_bloom = encode_with_dino_bloom
        self.root_dir = root_dir
        super().__init__("Acevedo", training=training_mode, data_synth=data_synth,augmentation_settings=augmentation_settings,balance_dataset_classes=balance_dataset_classes)
        self.preload_transforms,self.transforms = self.get_transform_function(grayscale=grayscale,load_jpg=True ,numpy=numpy,to_gpu=gpu,flatten=flatten,to_tensor=to_tensor,augmentation_settings= augmentation_settings,extra_transforms=[])
        self.path_to_data_folder = self.indexer.get_path_to_data_folder()
        self.supposed_dimension = (360,360)
        self.to_pil = transforms.ToPILImage()
        if self.encode_with_dino_bloom:
            self.dino_enc = self.get_dino_bloom_transform()
            self.get_item_function = self.get_and_dino_encoded_tif_image
        else:
            self.get_item_function = self.get_single_jpg_image
        self.loaded_data = {}
        
    
    def __len__(self):
        return len(self.indicies_list)

    def __getitem__(self, idx):
        '''returns specific item from this dataset'''
        if type(idx) is int:
            return self.get_bag_images(idx)
        elif isinstance(idx,Iterable):
            images = []
            labels = []
            for index in idx:
                image , label = self.get_bag_images(index)
                images.append(self.transforms(image))
                labels.append(label)
            return images,labels
        
    def get_bag_images(self,idx):
        data = []
        paths,label = self.indicies_list[idx]
        for path in paths:
            data.append(self.get_item_function(path))
        return data,label
        
    def get_single_jpg_image(self, path):
        if path not in self.loaded_data:
            image_path = os.path.join(self.path_to_data_folder,path)
            image = self.transforms(self.preload_transforms(image_path))
            self.loaded_data[path] = image
        return self.loaded_data[path]

    
    def get_and_dino_encoded_tif_image(self,path):
        if path not in self.loaded_data:
            image_path = os.path.join(self.path_to_data_folder,path)
            image = self.preload_transforms(image_path)
            dino_features = self.dino_enc(image)
            image_feaures = self.transforms(dino_features[0].cpu().detach())
            self.loaded_data[path] = image_feaures
        return self.loaded_data[path]
    
    
if __name__ == '__main__':
    mil_db = Acevedo_MIL_base(True,SinglePresenceMILSynthesizer(postive_classes=["ig"],bag_size=44),augmentation_settings=AugmentationSettings())
    x = mil_db[2]
    pass