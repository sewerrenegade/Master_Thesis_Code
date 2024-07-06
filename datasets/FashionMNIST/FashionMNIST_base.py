from torch.utils.data import Dataset, random_split, Subset
import torch
from datasets.base_dataset_abstraction import BaseDataset
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from torchvision import transforms
from collections.abc import Iterable
import random
import numpy as np

class FashionMNIST_base(BaseDataset):
    
    def __init__(self,training, root_dir = "data/", dataset_size = None,gpu = True, numpy = False,flatten = False,to_tensor = True,augmentation_settings = None):
        self.root_dir = root_dir
        super().__init__("FashionMNIST")
        self.training = training
        self.dataset_size = dataset_size
        self.classes = self.indexer.classes
        self.preload_transforms,self.transform = self.get_transform_function(grayscale=True,numpy=numpy,to_gpu=gpu,flatten=flatten,to_tensor=to_tensor,extra_transforms=[transforms.Normalize((0.5,), (0.5,))],augmentation_settings=augmentation_settings)
        self.data = FashionMNIST(train=training,root=root_dir,transform=self.preload_transforms)
        self.indicies_list = self.build_smaller_dataset()

    def get_random_samples_from_class(self, class_name, number_of_instances):
        indicies = self.class_indicies[class_name]
        random_indicies = random.sample(indicies,number_of_instances)
        return self[random_indicies]
    

    def build_smaller_dataset(self):
        if self.training:
            full_indicies = self.indexer.train_indicies
        else:
            full_indicies = self.indexer.test_indicies
        if self.dataset_size:
            self.class_indicies = {}
            class_size = int(self.dataset_size/len(self.classes))
            indicies = []
            for FashionMNIST_class in self.classes:
                indicies.extend(full_indicies[FashionMNIST_class][:class_size])
                self.class_indicies[FashionMNIST_class] = full_indicies[FashionMNIST_class][:class_size]
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
#ToDo update the script to be FashionMNIST or dataset agnostic

class FashionMNIST_MIL_base(Dataset):

    def __init__(self,training, data_synth, root_dir = "data/", transforms=None,gpu = True):
        self.root_dir = root_dir
        self.name = "FashionMNIST"
        self.training = training        
        self.data = FashionMNIST(train=training,root=root_dir,transform=self.preload_transforms)
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
    

# #ToDo update the script to be FashionMNIST or dataset agnostic
# #add version on GPU and CPU
# class FashionMNIST_Dataset_Referencer:
#     Train_Data = None
#     Test_Data = None
#     GPU_Train_Data = None
#     GPU_Test_Data = None
#     INDEXER = FashionMNIST_Indexer()
#     transfrom =  transforms.Compose([ToTensor()])
#     gpu_transfrom =  transforms.Compose([ToTensor(),ToGPUTrans()])

#     def get_or_load_datasets(training,root_dir,gpu = True):
#         if not gpu:
#             if training:
#                 if FashionMNIST_Dataset_Referencer.Train_Data:
#                     return FashionMNIST_Dataset_Referencer.Train_Data
#                 else:
#                     FashionMNIST_Dataset_Referencer.Train_Data=FashionMNIST(root_dir, train=True, download=False,transform= FashionMNIST_Dataset_Referencer.transfrom)
#                     return FashionMNIST_Dataset_Referencer.Train_Data
#             else:
#                 if FashionMNIST_Dataset_Referencer.Test_Data:
#                     return FashionMNIST_Dataset_Referencer.Test_Data
#                 else:
#                     FashionMNIST_Dataset_Referencer.Test_Data =FashionMNIST(root_dir, train=False, download=False,transform=  FashionMNIST_Dataset_Referencer.transfrom)
#                     return FashionMNIST_Dataset_Referencer.Test_Data
#         else:
#             if training:
#                 if FashionMNIST_Dataset_Referencer.GPU_Train_Data:
#                     return FashionMNIST_Dataset_Referencer.GPU_Train_Data
#                 else:
#                     FashionMNIST_Dataset_Referencer.GPU_Train_Data=FashionMNIST(root_dir, train=True, download=False,transform= FashionMNIST_Dataset_Referencer.gpu_transfrom)
#                     return FashionMNIST_Dataset_Referencer.GPU_Train_Data
#             else:
#                 if FashionMNIST_Dataset_Referencer.GPU_Test_Data:
#                     return FashionMNIST_Dataset_Referencer.GPU_Test_Data
#                 else:
#                     FashionMNIST_Dataset_Referencer.GPU_Test_Data =FashionMNIST(root_dir, train=False, download=False,transform=  FashionMNIST_Dataset_Referencer.gpu_transfrom)
#                     return FashionMNIST_Dataset_Referencer.GPU_Test_Data
            

            


