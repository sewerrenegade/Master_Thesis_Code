from torch.utils.data import Dataset
from abc import ABC, abstractmethod
import torch
import torchvision.transforms as transforms
from PIL import Image
import tifffile as tiff
from datasets.dataset_transforms import DatasetTransforms,DinoBloomEncodeTransform
import numpy as np
import random

from datasets.image_augmentor import AugmentationSettings
from datasets.indexer_scripts.indexer_abstraction import Indexer


class BaseMILDataset(Dataset, ABC):
    def __init__(self, database_name,augmentation_settings:AugmentationSettings,balance_dataset_classes,data_synth = None,training = True):
        self.name = database_name
        self.number_of_per_class_instances = balance_dataset_classes
        self.augmentation_settings = augmentation_settings
        if self.augmentation_settings is not None:
            self.augmentation_settings.dataset_name = self.name
        self.indexer:Indexer = self.get_indexer()
        if data_synth is None:
            self.per_class_indicies = self.indexer.get_bag_level_indicies(training)
        else:
            self.per_class_indicies = self.indexer.get_bag_level_indicies(training,balance_dataset_classes,data_synth)
        self.per_class_count = BaseMILDataset.count_per_class_samples(self.per_class_indicies)
        self.indicies_list,self.per_class_indicies,self.per_class_count =self.balance_dataset()
        self.classes = list(self.per_class_indicies.keys())
    
    def balance_dataset(self):
        if self.number_of_per_class_instances is None or self.number_of_per_class_instances == 0:
            new_class_indicies = {}
            paths = []
            labels = []
            for key,value in self.per_class_indicies.items():
                paths.extend(value)
                new_class_indicies[key] = list(range(len(labels),len(labels)+len(value)))
                labels.extend([key]*len(value))
            return list(zip(paths,labels)),new_class_indicies, BaseMILDataset.count_per_class_samples(new_class_indicies)
        
        if self.augmentation_settings is None and min(list(self.per_class_count.values())) < self.number_of_per_class_instances:
            raise ValueError(f"You have requested a balanced version of {self.name}, however the smallest class (size of {min(list(self.per_class_count.values()))}) in the natural dataset is smaller than the requested balance {self.per_class_count} and the augmentation of")
        
        new_class_indicies = {}
        paths = []
        labels = []
        for key,value in self.per_class_indicies.items():
            class_list_of_paths = value * (self.number_of_per_class_instances // len(value) + 1)
            class_list_of_paths = class_list_of_paths[:self.number_of_per_class_instances]
            paths.extend(class_list_of_paths)
            new_class_indicies[key] = list(range(len(labels),len(labels)+self.number_of_per_class_instances))
            labels.extend([key]*self.number_of_per_class_instances)
        return list(zip(paths,labels)),new_class_indicies, BaseMILDataset.count_per_class_samples(new_class_indicies)
    
    @staticmethod
    def count_per_class_samples(class_indicies_dict):
        return {key:len(value) for key,value in class_indicies_dict.items()}    
    
    def get_random_samples_from_class(self, class_name, number_of_instances):
        indicies = self.per_class_indicies[class_name]
        random_indicies = random.sample(indicies,number_of_instances)
        return self[random_indicies]

    def get_random_instances_from_all_classes(self, number_of_instances):
        instances_dict = {}
        for class_name in self.classes:
            instances_dict[class_name] = self.get_random_samples_from_class(
                class_name, number_of_instances
            )[0]
        return instances_dict

    def get_indexer(self) -> Indexer:
        from datasets.indexer_scripts. indexer_utils import get_dataset_indexer
        return get_dataset_indexer(self.name)
    
    def get_dino_bloom_transform(self):
        return DinoBloomEncodeTransform()
    
    def get_transform_function(
        self,
        load_tiff=False,
        load_jpg = False,
        augmentation_settings=None,
        numpy=False,
        to_gpu=False,
        flatten=False,
        grayscale=False,
        to_tensor=True,
        extra_transforms=[],
    ):
        self.dataset_transform = DatasetTransforms(
            self.name,
            load_tiff=load_tiff,
            load_jpg=load_jpg,
            augmentation_settings=augmentation_settings,
            numpy=numpy,
            to_gpu=to_gpu,
            flatten=flatten,
            grayscale=grayscale,
            to_tensor=to_tensor,
            extra_transforms=extra_transforms,
        )
        
        return self.dataset_transform.create_preload_and_postload_transforms()
