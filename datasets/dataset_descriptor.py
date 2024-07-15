import sys

import numpy as np
import torch


sys.path.append('/home/milad/Desktop/Master_Thesis/code/Master_Thesis_Code')
from datasets.data_synthesizers.data_sythesizer import SinglePresenceMILSynthesizer
from datasets.Acevedo.acevedo_base import Acevedo_MIL_base, Acevedo_base
from datasets.CIFAR10.CIFAR10_base import CIFAR10_base
from datasets.FashionMNIST.FashionMNIST_base import FashionMNIST_base
import os
import json
from datasets.base_dataset_abstraction import BaseDataset
from datasets.mil_dataset_abstraction import BaseMILDataset
from datasets.CIFAR10.CIFAR10_indexer import CIFAR10_Indexer
from datasets.FashionMNIST.FashionMNIST_indexer import FashionMNIST_Indexer
from datasets.MNIST.MNIST_indexer import MNIST_Indexer
from datasets.SCEMILA.SEMILA_indexer import SCEMILA_Indexer
from typing import Union, Tuple, Dict
from datasets.image_augmentor import DATASET_AUGMENTABIBILITY, Augmentability, AugmentationSettings

DESCRIPTORS = None

class DatasetDescriptor:
    def __init__(
        self,
        name: str= None,
        multiple_instance_dataset: bool= None,
        number_of_channels: int= None,
        output_dimension: Tuple[int, ...]= None,
        augmentation_scheme:Augmentability = None,
        class_distribution: dict= None,
        classes: list= None,
        augmentation_settings:Union[AugmentationSettings,None]= None,
        dataset:Union[BaseMILDataset, BaseDataset, None] = None
    ):
        assert dataset is not None or all(arg is not None for arg in [
        name,
        multiple_instance_dataset,
        number_of_channels,
        output_dimension,
        augmentation_scheme,
        class_distribution,
        classes,
        augmentation_settings
    ]), "Either 'dataset' should not be None or all other inputs should not be None."

        if dataset is not None:
            self.extract_info_from_live_dataset(dataset)
        else:
            self.name = name
            self.number_of_channels = number_of_channels
            self.output_dimension = output_dimension
            self.augmentation_settings = augmentation_settings
            self.augmentation_scheme = augmentation_scheme
            self.class_distribution = class_distribution
            self.size = sum([value for _,value in class_distribution.items()])
            self.multiple_instance_dataset = multiple_instance_dataset
            self.classes = classes
            
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "multiple_instance_dataset": self.multiple_instance_dataset,
            "number_of_channels": self.number_of_channels,
            "output_dimension": self.output_dimension,
            "class_distribution": self.class_distribution,
            "classes": self.classes,
            "augmentation_settings": self.augmentation_settings.to_dict(),
            "augmentation_scheme": self.augmentation_scheme.to_string(),
            "size": self.size
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'DatasetDescriptor':
        return cls(
            name=data.get("name"),
            multiple_instance_dataset=data.get("multiple_instance_dataset"),
            number_of_channels=data.get("number_of_channels"),
            output_dimension=data.get("output_dimension"),
            class_distribution=data.get("class_distribution"),
            classes=data.get("classes"),
            augmentation_settings=AugmentationSettings.from_dict(data.get("augmentation_settings")),
            augmentation_scheme =  Augmentability.from_string(data.get("augmentation_scheme")),
        )
    def extract_info_from_live_dataset(self,dataset:Union[BaseMILDataset, BaseDataset]):
        self.name = dataset.name
        sample_output = dataset[0][0]
        self.output_dimension = DatasetDescriptor.get_tensor_shape(sample_output)
        if isinstance(dataset,BaseDataset):
            self.multiple_instance_dataset = False
            self.number_of_channels = self.output_dimension[0]
        if isinstance(dataset,BaseMILDataset):
            self.multiple_instance_dataset = True
            self.number_of_channels = self.output_dimension[1]
        self.augmentation_settings = dataset.augmentation_settings
        self.augmentation_scheme = DATASET_AUGMENTABIBILITY[self.name]
        self.class_distribution = dataset.per_class_count
        self.classes = dataset.classes
        self.size = sum([value for _,value in self.class_distribution.items()])
    
    def __str__(self):
        return (f"DatasetDescriptor(\n"
                f"  name={self.name},\n"
                f"  number_of_channels={self.number_of_channels},\n"
                f"  dimensions={self.output_dimension},\n"
                f"  class_distribution={self.class_distribution},\n"
                f"  classes={self.classes},\n"
                f"  augmentation_settings={self.augmentation_settings},\n"
                f"  augmentation_scheme={self.augmentation_scheme},\n"
                f"  size={self.size},\n"
                f"  multiple_instance_dataset={self.multiple_instance_dataset}\n"
                f")")
            
    @staticmethod
    def get_tensor_shape(tensor):
        if isinstance(tensor, torch.Tensor):
            return tuple(tensor.shape)
        elif isinstance(tensor, np.ndarray):
            return tensor.shape
        elif isinstance(tensor, list):
            if len(tensor) > 0:
                first_element = tensor[0]
                if isinstance(first_element, (torch.Tensor, np.ndarray)):
                    return (len(tensor), *DatasetDescriptor.get_tensor_shape(first_element))
                else:
                    raise ValueError("The list elements must be torch tensors or numpy arrays.")
            else:
                raise ValueError("The input list is empty.")
        else:
            raise TypeError("The input must be a torch tensor, numpy array, or a list of such objects.")

    def count_number_of_instances_from_class_dict(self,dict_count):
        return sum([value for _,value in dict_count.items()])
    
    def print_dataset_info(self):
        print(f"Dataset: {self.name}")
        print(f"Number of Channels: {self.number_of_channels}")
        print(f"Dimensions: {self.output_dimension}")
        print(f"Augmentation Scheme: {self.augmentation_scheme}")
        print(f"Total_size: {self.size}")
        print(f"Test Size: {self.test_class_distribution}")
        print(f"Train Size: {self.train_size}")
        print(f"Indexer Type: {type(self.indexer)}")
        

            
    #depricated dataset descriptors ar created directly from datasets
    @staticmethod
    def calculate_dataset_descriptors():
        mnist_indexer = MNIST_Indexer()
        fmnist_indexer = FashionMNIST_Indexer()
        cifar_indexer = CIFAR10_Indexer()
        scemila_indexer = SCEMILA_Indexer()
        list_of_descriptors = []
        list_of_descriptors.append(DatasetDescriptor(
            name="MNIST",
            multiple_instance_dataset=False,
            number_of_channels=1,
            dimensions=(28, 28),
            description="classic torchvision MNSIT dataset",
            is_image=True,
            train_class_distribution=mnist_indexer.train_class_count,
            test_class_distribution=mnist_indexer.test_class_count,
            classes=mnist_indexer.classes,
        ))
        list_of_descriptors.append(DatasetDescriptor(
            name="FashionMNIST",
            multiple_instance_dataset=False,
            number_of_channels=1,
            dimensions=(28, 28),
            description="classic torchvision FNSIT dataset",
            is_image=True,
            train_class_distribution=fmnist_indexer.train_class_count,
            test_class_distribution=fmnist_indexer.test_class_count,
            classes=mnist_indexer.classes,
        ))
        list_of_descriptors.append(DatasetDescriptor(
            name = "CIFAR10",
            multiple_instance_dataset = False,
            number_of_channels = 3,
            dimensions = (32, 32),
            description = "classic torchvision CIFAR10 dataset",
            is_image = True,
            train_class_distribution = cifar_indexer.train_class_count,
            test_class_distribution = cifar_indexer.test_class_count,
            classes = cifar_indexer.classes,
        ))
        list_of_descriptors.append(DatasetDescriptor(
            name = "SCEMILA/image_data",
            multiple_instance_dataset=False,
            number_of_channels=3,
            dimensions=(144, 144),
            description="SCEMILA single cell dataset, its lables were taken from AML Hehr",
            is_image=True,
            train_class_distribution= scemila_indexer.instance_level_class_count,
            classes= scemila_indexer.instance_classes
        ))
        list_of_descriptors.append(DatasetDescriptor(
            name = "SCEMILA/fnl34_feature_data",
            multiple_instance_dataset=False,
            number_of_channels=512,
            dimensions=(5, 5),
            description="features for SCEMILA single cell images, feature extraction trained in fully supervised setting by Matthias Hehr",
            is_image=False,
            train_class_distribution=scemila_indexer.instance_level_class_count,
            classes= scemila_indexer.instance_classes
        ))
        return {desc.name: desc for desc in list_of_descriptors}


if __name__ == '__main__':
    test_db = Acevedo_MIL_base(flatten = False,data_synth=SinglePresenceMILSynthesizer(postive_classes=["ig"],bag_size=44),to_tensor = True,grayscale= False, training_mode = True,balance_dataset_classes=50, augmentation_settings=AugmentationSettings.all_false_except_one("gaussian_noise_aug"))
    desc = DatasetDescriptor(dataset= test_db)
    print(desc)
    descriptor_dict = desc.to_dict()
    json_string = json.dumps(descriptor_dict)
    deserialized_dict = json.loads(json_string)
    new_descriptor = DatasetDescriptor.from_dict(deserialized_dict)
    pass