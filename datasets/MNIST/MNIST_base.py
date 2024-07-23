
import sys
sys.path.append('/home/milad/Desktop/Master_Thesis/code/Master_Thesis_Code')
from datasets.base_dataset_abstraction import BaseDataset
from datasets.mil_dataset_abstraction import BaseMILDataset
import numpy as np
from datasets.data_synthesizers.data_sythesizer import SinglePresenceMILSynthesizer
import torch
import random
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision import transforms

from datasets.image_augmentor import AugmentationSettings
from collections.abc import Iterable



class MNISTBase(BaseDataset):
    def __init__(self,training_mode, root_dir = "data/",balance_dataset_classes = None,gpu = True, numpy = False,flatten = False,to_tensor = True,resize = False,augmentation_settings = None):
        self.root_dir = root_dir
        super().__init__("MNIST",augmentation_settings=augmentation_settings,balance_dataset_classes=balance_dataset_classes,training=training_mode)
        self.training_mode = training_mode
        self.preload_transforms,self.transform = self.get_transform_function(augmentation_settings= augmentation_settings,grayscale=True,resize = resize,numpy=numpy,to_gpu=gpu,flatten=flatten,to_tensor=to_tensor,extra_transforms=[transforms.Normalize((0.1307,), (0.3081,))],)
        self.data = MNIST(train=training_mode,root=root_dir,transform=self.preload_transforms)
        

    
    def __len__(self):
        return len(self.indicies_list)
    
    def get_only_pretransform_item(self, index):
        if isinstance(index,int):
            x = self.indicies_list[index][0]
            image_data,image_label = self.data[x][0],self.indicies_list[index][1]
            return image_data, image_label
        else:
            image_data =[self.data[self.indicies_list[ind][0]][0] for ind in index]
            image_label =[self.indicies_list[ind][1] for ind in index]
            return image_data, image_label
        
    def __getitem__(self, index):
        if isinstance(index,int):
            x = self.indicies_list[index][0]
            image_data,image_label = self.data[x][0],self.indicies_list[index][1]
            return self.transform(image_data), image_label
        else:
            image_data =[self.transform(self.data[self.indicies_list[ind][0]][0]) for ind in index]
            image_label =[self.indicies_list[ind][1] for ind in index]
            return image_data, image_label
#ToDo update the script to be MNIST or dataset agnostic

class MNIST_MIL_base(BaseMILDataset):

    def __init__(self,training_mode,data_synth, root_dir = "data/",balance_dataset_classes = 100,gpu = True, numpy = False,flatten = False,to_tensor = True,augmentation_settings = None):
        self.root_dir = root_dir
        super().__init__("MIL_MNIST",data_synth=data_synth,augmentation_settings=augmentation_settings,balance_dataset_classes=balance_dataset_classes,training=training_mode)
        self.training = training_mode
        self.synth = data_synth
        self.preload_transforms,self.transform = self.get_transform_function(augmentation_settings= augmentation_settings,grayscale=True,numpy=numpy,to_gpu=gpu,flatten=flatten,to_tensor=to_tensor,extra_transforms=[transforms.Normalize((0.1307,), (0.3081,))],)
        self.data = MNIST(train=training_mode,root=root_dir,transform=self.preload_transforms)

    def __len__(self):
        return len(self.indicies_list)

    def __getitem__(self, index):
        bag_indicies,bag_label = self.indicies_list[index]
        bag_data = self.get_images_from_indicies(bag_indicies)
        return bag_data, bag_label
    
    def get_images_from_indicies(self, indicies):
        data = []
        for index in indicies:
            data.append(self.transform(self.data[index][0]))        
        return data
    

if __name__ == '__main__':
    mil_db = MNIST_MIL_base(True,SinglePresenceMILSynthesizer(postive_classes=[9],bag_size=44),augmentation_settings=AugmentationSettings())
    x = mil_db[2]
    pass
            

            


