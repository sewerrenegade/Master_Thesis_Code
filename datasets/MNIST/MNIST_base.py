from datasets.base_dataset_abstraction import BaseDataset,ToGPUTransform,ToNPTransform
import numpy as np
import torch
import random
from torchvision.datasets import MNIST
from datasets.MNIST.MNIST_indexer import MNIST_Indexer
from torchvision.transforms import ToTensor
from torchvision import transforms
from collections.abc import Iterable



class MNIST_base(BaseDataset):
    def __init__(self,training, root_dir = "data/",dataset_size = None,gpu = True, numpy = False,flatten = False,to_tensor = True,augment_image  = True):
        self.root_dir = root_dir
        self.name = "MNIST"
        self.training = training
        self.dataset_size = dataset_size
        self.classes = MNIST_Dataset_Referencer.INDEXER.classes
        self.preload_transforms,self.transform = self.get_transform_function(augment_image= augment_image,grayscale=True,numpy=numpy,to_gpu=gpu,flatten=flatten,to_tensor=to_tensor,extra_transforms=[transforms.Normalize((0.1307,), (0.3081,))],)
        self.data = MNIST_Dataset_Referencer.get_or_load_datasets(training,root_dir,self.preload_transforms)
        self.indicies_list = self.build_smaller_dataset()
        
    def get_n_random_instances_from_class(self, class_name, number_of_instances):
        indicies = self.class_indicies[class_name]
        random_indicies = random.sample(indicies,number_of_instances)
        return self[random_indicies]

    def build_smaller_dataset(self):
        if self.training:
            full_indicies = MNIST_Dataset_Referencer.INDEXER.train_indicies
        else:
            full_indicies = MNIST_Dataset_Referencer.INDEXER.test_indicies
        if self.dataset_size:
            self.class_indicies = {}
            class_size = int(self.dataset_size/len(self.classes))

            indicies = []
            for mnist_class in self.classes:
                indicies.extend(full_indicies[mnist_class][:class_size])
                self.class_indicies[mnist_class] = full_indicies[mnist_class][:class_size]
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
            image_data =[self.transform(self.data[ind][0]) for ind in index]
            image_label =[self.data[ind][0] for ind in index]

            return image_data, image_label
#ToDo update the script to be MNIST or dataset agnostic

class MNIST_MIL_base(BaseDataset):

    def __init__(self,training, data_synth, root_dir = "data/", transforms=None,gpu = True):
        self.root_dir = root_dir
        self.name = "MNIST"
        self.training = training        
        self.data = MNIST_Dataset_Referencer.get_or_load_datasets(training,root_dir,gpu)
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
class MNIST_Dataset_Referencer:
    Train_Data = None
    Test_Data = None
    GPU_Train_Data = None
    GPU_Test_Data = None
    INDEXER = MNIST_Indexer()
    transforms_list = []

    def get_or_load_datasets(training,root_dir,transforms):
        return MNIST(root_dir, train=training, download=False,transform= transforms)
    
            

            


