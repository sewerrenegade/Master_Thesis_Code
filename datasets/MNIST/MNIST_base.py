from torch.utils.data import Dataset, random_split, Subset
import os
import numpy as np
import torch

from torchvision.datasets import MNIST
from datasets.MNIST.MNIST_indexer import MNIST_Indexer
from torchvision.transforms import ToTensor
from torchvision import transforms



class baseDataset(Dataset):

    def __init__(self,training, root_dir = "data/", transform=None):
        self.root_dir = root_dir
        self.training = training        
        self.data = MNIST_Dataset_Referencer.get_or_load_datasets(training,root_dir)
        if transform == None:
            self.transform =transforms.Compose([transforms.Grayscale(),transforms.Normalize((0.1307,), (0.3081,))])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_data,image_label = self.data[index]
        return self.transform(image_data), image_label
#ToDo update the script to be MNIST or dataset agnostic

class baseDatasetMIL(Dataset):

    def __init__(self,training, data_synth, root_dir = "data/", transforms=None):
        self.root_dir = root_dir
        self.training = training        
        self.data = MNIST_Dataset_Referencer.get_or_load_datasets(training,root_dir)
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
    
class ToGPUTrans:
    def __init__(self) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return pic.to(self.device)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


#ToDo update the script to be MNIST or dataset agnostic
class MNIST_Dataset_Referencer:
    Train_Data = None
    Test_Data = None
    INDEXER = MNIST_Indexer()
    transfrom =  transforms.Compose([ToTensor(),])#ToGPUTrans()

    def get_or_load_datasets(training,root_dir):
        if training:
            if MNIST_Dataset_Referencer.Train_Data:
                return MNIST_Dataset_Referencer.Train_Data
            else:
                MNIST_Dataset_Referencer.Train_Data=MNIST(root_dir, train=True, download=False,transform= MNIST_Dataset_Referencer.transfrom)
                return MNIST_Dataset_Referencer.Train_Data
        else:
            if MNIST_Dataset_Referencer.Test_Data:
                return MNIST_Dataset_Referencer.Test_Data
            else:
                MNIST_Dataset_Referencer.Test_Data =MNIST(root_dir, train=False, download=False,transform=  MNIST_Dataset_Referencer.transfrom)
                return MNIST_Dataset_Referencer.Test_Data
            


