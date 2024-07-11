import sys
sys.path.append('/home/milad/Desktop/Master_Thesis/code/Master_Thesis_Code')
import os
import json

from datasets.CIFAR10.CIFAR10_indexer import CIFAR10_Indexer
from datasets.FashionMNIST.FashionMNIST_indexer import FashionMNIST_Indexer
from datasets.MNIST.MNIST_indexer import MNIST_Indexer
from datasets.SCEMILA.SEMILA_indexer import SCEMILA_Indexer
from typing import Tuple
from datasets.image_augmentor import DATASET_AUGMENTABIBILITY

DESCRIPTORS = None

class DatasetDescriptor:
    def __init__(
        self,
        name: str,
        is_multi_instance: bool,
        number_of_channels: int,
        dimensions: Tuple[int, ...],
        description: str,
        is_image: bool,
        train_class_distribution: dict,
        classes: list,
        test_class_distribution: dict = {},
    ):
        self.number_of_channels = number_of_channels
        self.dimensions = dimensions
        self.name = name
        self.description = description
        self.augmentation_scheme = DATASET_AUGMENTABIBILITY[name]
        self.is_image = is_image
        self.test_class_distribution = test_class_distribution
        self.train_class_distribution = train_class_distribution
        self.test_size = self.count_number_of_instances_from_class_dict(test_class_distribution)
        self.train_size = self.count_number_of_instances_from_class_dict(train_class_distribution)
        self.total_size = self.test_size + self.train_size
        self.multiple_instance_dataset = is_multi_instance
        self.classes = classes
        
    def count_number_of_instances_from_class_dict(self,dict_count):
        return sum([value for _,value in dict_count.items()])
    
    def print_dataset_info(self):
        print(f"Dataset: {self.name}")
        print(f"Description: {self.description}")
        print(f"Number of Channels: {self.number_of_channels}")
        print(f"Dimensions: {self.dimensions}")
        print(f"Augmentation Scheme: {self.augmentation_scheme}")
        print(f"Is Image: {self.is_image}")
        print(f"Size: {self.size}")
        print(f"Test Size: {self.test_class_distribution}")
        print(f"Train Size: {self.train_size}")
        print(f"Indexer Type: {type(self.indexer)}")
        
    @staticmethod
    def get_serialised_dataset_descriptors():
        path_to_file = "data/serialised_dataset_descriptors.txt"
        if os.path.exists(path_to_file):
            with open(path_to_file, 'r') as file:
                descriptors = json.load(file)
        else:
            descriptors = DatasetDescriptor.get_serialised_dataset_descriptors
            with open(path_to_file, 'w') as file:
                json.dump(descriptors, file)
        global DESCRIPTORS
        DESCRIPTORS = descriptors
            
    
    @staticmethod
    def calculate_dataset_descriptors():
        mnist_indexer = MNIST_Indexer()
        fmnist_indexer = FashionMNIST_Indexer()
        cifar_indexer = CIFAR10_Indexer()
        scemila_indexer = SCEMILA_Indexer()
        list_of_descriptors = []
        list_of_descriptors.append(DatasetDescriptor(
            name="MNIST",
            is_multi_instance=False,
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
            is_multi_instance=False,
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
            is_multi_instance = False,
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
            is_multi_instance=False,
            number_of_channels=3,
            dimensions=(144, 144),
            description="SCEMILA single cell dataset, its lables were taken from AML Hehr",
            is_image=True,
            train_class_distribution= scemila_indexer.instance_level_class_count,
            classes= scemila_indexer.instance_classes
        ))
        list_of_descriptors.append(DatasetDescriptor(
            name = "SCEMILA/fnl34_feature_data",
            is_multi_instance=False,
            number_of_channels=512,
            dimensions=(5, 5),
            description="features for SCEMILA single cell images, feature extraction trained in fully supervised setting by Matthias Hehr",
            is_image=False,
            train_class_distribution=scemila_indexer.instance_level_class_count,
            classes= scemila_indexer.instance_classes
        ))
        return {desc.name: desc for desc in list_of_descriptors}


