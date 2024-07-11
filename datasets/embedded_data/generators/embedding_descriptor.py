from dataclasses import dataclass
from torch.utils.data import Dataset
from typing import Callable,Union
import json

from datasets.image_augmentor import AugmentationSettings

@dataclass
class EmbeddingDescriptor:
    name: str
    dataset: Dataset
    downprojection_function_name:str
    downprojection_function: Callable
    downprojection_function_settings: dict


class SerializableEmbeddingDescriptor:
    def __init__(self, name, dataset_name, dataset_sampling, augmentation_settings, dino_bloom, transform_name, transform_settings):
        self.name = name
        self.dataset_name = dataset_name
        self.dataset_sampling = dataset_sampling#if is int then n samples are taken from each sample. otherwise the entire dataset is taken
        self.augmentation_settings = augmentation_settings
        self.dino_bloom = dino_bloom
        self.transform_name = transform_name
        self.transform_settings = transform_settings
        
    def __str__(self):
        return (f"Name: {self.name}\n"
                f"Dataset Name: {self.dataset_name}\n"
                f"Dataset Sampling: {self.dataset_sampling}\n"
                f"Augmentation Settings: {self.augmentation_settings}\n"
                f"DINO Bloom: {self.dino_bloom}\n"
                f"Transform Name: {self.transform_name}\n"
                f"Transform Settings: {self.transform_settings}\n")
        
    def to_dict(self):
        return {
            'name': self.name,
            'dataset_name': self.dataset_name,
            'dataset_sampling': self.dataset_sampling,
            'augmentation_settings': self.augmentation_settings.__dict__,  # assuming augmentation_settings is an object
            'dino_bloom': self.dino_bloom,
            'transform_name': self.transform_name,
            'transform_settings': self.transform_settings,
        }

    @staticmethod
    def from_dict(data):
        augmentation_settings = AugmentationSettings(**data['augmentation_settings'])
        return SerializableEmbeddingDescriptor(
            name=data['name'],
            dataset_name=data['dataset_name'],
            dataset_sampling=data['dataset_sampling'],
            augmentation_settings=augmentation_settings,
            dino_bloom=data['dino_bloom'],
            transform_name=data['transform_name'],
            transform_settings=data['transform_settings'],
        )
    
def create_serialializable_descriptor_from_live_descriptor(emb_decriptor:EmbeddingDescriptor):
    return SerializableEmbeddingDescriptor(
        name=emb_decriptor.name,
        dataset_name=emb_decriptor.dataset.name,
        dataset_sampling=emb_decriptor.dataset.number_of_per_class_instances,
        augmentation_settings=emb_decriptor.dataset.augmentation_settings,
        dino_bloom=getattr(emb_decriptor.dataset, 'encode_with_dino_bloom', False),
        transform_name=emb_decriptor.downprojection_function_name,
        transform_settings=emb_decriptor.downprojection_function_settings
    )