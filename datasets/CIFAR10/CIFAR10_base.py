from torch.utils.data import Dataset
import torch
from datasets.base_dataset_abstraction import BaseDataset
from torchvision.datasets import CIFAR10
from torchvision import transforms
import random

from datasets.mil_dataset_abstraction import BaseMILDataset


class CIFAR10_base(BaseDataset):

    def __init__(self,training_mode, root_dir = "data/",balance_dataset_classes = None,gpu = True, grayscale = False, numpy = False,flatten = False,to_tensor = True,augmentation_settings = None):
        self.root_dir = root_dir
        super().__init__("CIFAR10",augmentation_settings=augmentation_settings,balance_dataset_classes=balance_dataset_classes,training=training_mode)
        self.training_mode = training_mode
        self.preload_transforms,self.transform = self.get_transform_function(augmentation_settings= augmentation_settings,grayscale=grayscale,numpy=numpy,to_gpu=gpu,flatten=flatten,to_tensor=to_tensor,extra_transforms=[transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])])
        self.data = CIFAR10(train=training_mode,root=root_dir,transform=self.preload_transforms)
        

    
    def __len__(self):
        return len(self.indicies_list)

    def __getitem__(self, index):
        if isinstance(index,int):
            x = self.indicies_list[index][0]
            image_data,image_label = self.data[x][0],self.indicies_list[index][1]
            return self.transform(image_data), image_label
        else:
            image_data =[self.transform(self.data[self.indicies_list[ind][0]][0]) for ind in index]
            image_label =[self.indicies_list[ind][1] for ind in index]
            return image_data, image_label
#ToDo update the script to be CIFAR10 or dataset agnostic

class CIFAR10_MIL_base(BaseMILDataset):

    def __init__(self,training_mode,data_synth, root_dir = "data/",balance_dataset_classes = 100,gpu = True, numpy = False,flatten = False,to_tensor = True,augmentation_settings = None,grayscale = False):
        self.root_dir = root_dir
        super().__init__("MIL_CIFAR10",data_synth=data_synth,augmentation_settings=augmentation_settings,balance_dataset_classes=balance_dataset_classes,training=training_mode)
        self.training = training_mode
        self.synth = data_synth
        self.preload_transforms,self.transform = self.get_transform_function(augmentation_settings= augmentation_settings,grayscale=grayscale,numpy=numpy,to_gpu=gpu,flatten=flatten,to_tensor=to_tensor,extra_transforms=[transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])])
        self.data = CIFAR10(train=training_mode,root=root_dir,transform=self.preload_transforms)

    def __len__(self):
        return len(self.indicies_list)

    def __getitem__(self, index):
        bag_indicies,bag_label = self.indicies_list[index]
        bag_data = self.get_images_from_indicies(bag_indicies)
        return bag_data, bag_label
    
    def get_images_from_indicies(self, indicies):
        data = []
        element_index = 0
        for index in indicies:
            data.append(self.transform(self.data[index][0]))
            element_index += 1
        
        return data
    


#ToDo update the script to be MNIST or dataset agnostic
#add version on GPU and CPU
# class CIFAR10_Dataset_Referencer:
#     Train_Data = None
#     Test_Data = None
#     GPU_Train_Data = None
#     GPU_Test_Data = None
#     INDEXER = CIFAR10_Indexer()
#     transforms_list = []

#     def get_or_load_datasets(training,root_dir,transforms):
#         return CIFAR10(root_dir, train=training, download=False,transform= transforms)
    


            


