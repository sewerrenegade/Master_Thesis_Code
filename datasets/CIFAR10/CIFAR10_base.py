from torch.utils.data import Dataset, random_split, Subset
import torch
from datasets.base_dataset_abstraction import baseDataset
from torchvision.datasets import CIFAR10
from datasets.CIFAR10.CIFAR10_indexer import CIFAR10_Indexer
from torchvision.transforms import ToTensor
from torchvision import transforms
from collections.abc import Iterable
import random
import numpy as np


class CIFAR10_base(baseDataset):

    def __init__(self,training, root_dir = "data/", dataset_size = None,gpu = True, numpy = False,flatten = False,to_tensor = True):
        self.root_dir = root_dir
        self.name = "CIFAR10"
        self.training = training
        self.dataset_size = dataset_size
        self.classes = CIFAR10_Dataset_Referencer.INDEXER.classes
        self.preload_transforms,self.transform = self.get_transform_function(grayscale=False,numpy=numpy,to_gpu=gpu,flatten=flatten,to_tensor=to_tensor,extra_transforms=[transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])],)
        self.data = CIFAR10_Dataset_Referencer.get_or_load_datasets(training,root_dir,self.preload_transforms)
        self.indicies_list = self.build_smaller_dataset()


    def get_n_random_instances_from_class(self, class_name, number_of_instances):
        indicies = self.class_indicies[class_name]
        random_indicies = random.sample(indicies,number_of_instances)
        return self[random_indicies]
    

    def build_smaller_dataset(self):
        if self.training:
            full_indicies = CIFAR10_Dataset_Referencer.INDEXER.train_indicies
        else:
            full_indicies = CIFAR10_Dataset_Referencer.INDEXER.test_indicies
        if self.dataset_size:
            self.class_indicies = {}
            class_size = int(self.dataset_size/len(self.classes))
            indicies = []
            for cifar10_class in self.classes:
                indicies.extend(full_indicies[cifar10_class][:class_size])
                self.class_indicies[cifar10_class] = full_indicies[cifar10_class][:class_size]
            return indicies
        self.class_indicies = full_indicies
    
    def __len__(self):
        if self.dataset_size:
            return len(self.indicies_list)
        else:
            return len(self.data)

    def __getitem__(self, index):
        if self.dataset_size:
            index = self.indicies_list[index]
        if isinstance(index,int):
            image_data,image_label = self.data[index]
            return self.transform(image_data), image_label
        else:
            image_data = [self.transform(self.data[ind][0]) for ind in index]
            image_label = [self.data[ind][0] for ind in index]
            return image_data, image_label
#ToDo update the script to be CIFAR10 or dataset agnostic

class CIFAR10_MIL_base(Dataset):

    def __init__(self,training, data_synth, root_dir = "data/", transforms=None,gpu = True):
        self.root_dir = root_dir
        self.name = "CIFAR10"
        self.training = training        
        self.data = CIFAR10_Dataset_Referencer.get_or_load_datasets(training,root_dir,gpu)
        self.synth = data_synth
        if training:
            self.data_indicies = self.synth.generate_train_bags(1000)
        else:
            self.data_indicies = self.synth.generate_test_bags(100)
        self.transforms = transforms

    def __len__(self):
        return len(self.data_indicies)

    def __getitem__(self, index):
        bag_indicies,bag_label = self.data_indicies[index]
        bag_data = self.get_images_from_indicies(bag_indicies)
        return bag_data, bag_label
    
    def get_images_from_indicies(self, indicies):
        data = []
        element_index = 0
        for index in indicies:
            data.append(self.transforms(self.data[index][0]))
            element_index += 1
        
        return torch.stack(data, dim=0)
    


#ToDo update the script to be MNIST or dataset agnostic
#add version on GPU and CPU
class CIFAR10_Dataset_Referencer:
    Train_Data = None
    Test_Data = None
    GPU_Train_Data = None
    GPU_Test_Data = None
    INDEXER = CIFAR10_Indexer()
    transforms_list = []

    def get_or_load_datasets(training,root_dir,transforms):
        return CIFAR10(root_dir, train=training, download=False,transform= transforms)
    


            


