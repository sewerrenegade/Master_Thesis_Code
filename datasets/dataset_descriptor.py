import sys

import numpy as np
import torch

from datasets.embedded_datasets.dataset.embedding_base import EmbeddingBaseDataset
from datasets.data_synthesizers.data_sythesizer import SinglePresenceMILSynthesizer
from datasets.Acevedo.acevedo_base import Acevedo_MIL_base
import json
from datasets.base_dataset_abstraction import BaseDataset
from datasets.mil_dataset_abstraction import BagSizeTypes, BaseMILDataset
from typing import Union, Tuple, Dict
from datasets.image_augmentor import (
    DATASET_AUGMENTABIBILITY,
    Augmentability,
    AugmentationSettings,
)

DESCRIPTORS = None


class SerializableDatasetDescriptor:
    def __init__(
        self,
        name: str = None,
        multiple_instance_dataset: bool = None,
        bag_size: Tuple[int, ...] = None,
        number_of_channels: int = None,
        output_dimension: Tuple[int, ...] = None,
        augmentation_scheme: Augmentability = None,
        class_distribution: dict = None,
        classes: list = None,
        augmentation_settings: Union[AugmentationSettings, None] = None,
        dino_bloom: bool = False,
        dataset: Union[BaseMILDataset, BaseDataset, None] = None,
    ):
        assert dataset is not None or all(
            arg is not None
            for arg in [
                name,
                multiple_instance_dataset,
                number_of_channels,
                output_dimension,
                augmentation_scheme,
                class_distribution,
                classes,
                augmentation_settings,
            ]
        ), "Either 'dataset' should not be None or all other inputs should not be None."

        if dataset is not None:
            self.extract_info_from_live_dataset(dataset)
        else:
            self.name = name
            self.number_of_channels = number_of_channels
            self.output_dimension = output_dimension
            self.augmentation_settings = augmentation_settings
            self.augmentation_scheme = augmentation_scheme
            self.class_distribution = class_distribution
            self.size = sum([value for _, value in class_distribution.items()])
            self.multiple_instance_dataset = multiple_instance_dataset
            self.bag_size = bag_size
            self.classes = classes
            self.dino_bloom = dino_bloom

    def to_dict(self) -> Dict:
        if isinstance(self.bag_size,tuple):
            bag_size_data= []
            for entry in self.bag_size:
                if hasattr(entry,"to_string"):
                    bag_size_data.append(entry.to_string())
                else:
                    bag_size_data.append(str(entry))
        elif hasattr(self.bag_size,"to_string"):
            bag_size_data = self.bag_size.to_string()
        else:
            raise TypeError("Unsupported bag size format")
                    
                
        return {
            "name": self.name,
            "multiple_instance_dataset": self.multiple_instance_dataset,
            "bag_size": bag_size_data,
            "number_of_channels": self.number_of_channels,
            "output_dimension": self.output_dimension,
            "class_distribution": self.class_distribution,
            "classes": self.classes,
            "augmentation_settings": self.augmentation_settings.to_dict(),
            "augmentation_scheme": self.augmentation_scheme.to_string(),
            "size": self.size,
            "dino_bloom": self.dino_bloom,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SerializableDatasetDescriptor":
        return cls(
            name=data.get("name"),
            multiple_instance_dataset=data.get("multiple_instance_dataset"),
            bag_size=BagSizeTypes.from_string(data.get("bag_size")),
            number_of_channels=data.get("number_of_channels"),
            output_dimension=data.get("output_dimension"),
            class_distribution=data.get("class_distribution"),
            classes=data.get("classes"),
            dino_bloom=data.get("dino_bloom"),
            augmentation_settings=AugmentationSettings.from_dict(
                data.get("augmentation_settings")
            ),
            augmentation_scheme=Augmentability.from_string(
                data.get("augmentation_scheme")
            ),
        )

    def extract_info_from_live_dataset(
        self, dataset: Union[BaseMILDataset, BaseDataset]
    ):
        self.name = dataset.name
        sample_output = dataset[0][0]
        self.output_dimension = SerializableDatasetDescriptor.get_output_shape(
            dataset
        )
        if isinstance(dataset, BaseDataset):
            self.multiple_instance_dataset = False
            self.number_of_channels = self.output_dimension[0]
            self.bag_size = (
                BagSizeTypes.NOT_APPLICABLE
            )
        if isinstance(dataset, BaseMILDataset):
            self.multiple_instance_dataset = True
            self.number_of_channels = self.output_dimension[0]
            self.bag_size = dataset.bag_size
        self.augmentation_settings = dataset.augmentation_settings
        self.augmentation_scheme = DATASET_AUGMENTABIBILITY[self.name]
        self.class_distribution = dataset.per_class_count
        self.classes = dataset.classes
        self.dino_bloom = getattr(dataset, "encode_with_dino_bloom", False)
        self.size = sum([value for _, value in self.class_distribution.items()])

    def __str__(self):
        return (
            f"DatasetDescriptor(\n"
            f"  name={self.name},\n"
            f"  number_of_channels={self.number_of_channels},\n"
            f"  dimensions={self.output_dimension},\n"
            f"  class_distribution={self.class_distribution},\n"
            f"  classes={self.classes},\n"
            f"  augmentation_settings={self.augmentation_settings},\n"
            f"  augmentation_scheme={self.augmentation_scheme.to_string()},\n"
            f"  size={self.size},\n"
            f"  multiple_instance_dataset={self.multiple_instance_dataset}\n"
            f"  bag_size={self.bag_size}\n"
            f")"
        )

    @staticmethod
    def get_output_shape(dataset):
        if hasattr(dataset,"encode_with_dino_bloom") and dataset.encode_with_dino_bloom:
            return (384,)
        tensor = dataset[0][0]
        if isinstance(tensor, torch.Tensor):
            return tuple(tensor.shape)
        elif isinstance(tensor, np.ndarray):
            return tensor.shape
        elif isinstance(tensor, list):
            if len(tensor) > 0:
                first_element = tensor[0]
                if isinstance(first_element, (torch.Tensor, np.ndarray)):
                    return (
                        len(tensor),
                        *SerializableDatasetDescriptor.get_output_shape(first_element),
                    )
                else:
                    raise ValueError(
                        "The list elements must be torch tensors or numpy arrays."
                    )
            else:
                raise ValueError("The input list is empty.")
        else:
            raise TypeError(
                "The input must be a torch tensor, numpy array, or a list of such objects."
            )

    def count_number_of_instances_from_class_dict(self, dict_count):
        return sum([value for _, value in dict_count.items()])


# depricated dataset descriptors ar created directly from datasets


def get_data_descriptors(
    dataset: Union[BaseDataset, BaseMILDataset, EmbeddingBaseDataset]
):
    if isinstance(dataset, Union[BaseDataset, BaseMILDataset]):
        return SerializableDatasetDescriptor(dataset=dataset)
    elif isinstance(dataset, EmbeddingBaseDataset):
        from datasets.embedded_datasets.generators.embedding_descriptor import (
            SerializableEmbeddingDescriptor,
        )

        return SerializableEmbeddingDescriptor(dataset)


if __name__ == "__main__":
    test_db = Acevedo_MIL_base(
        flatten=False,
        data_synth=SinglePresenceMILSynthesizer(postive_classes=["ig"], bag_size=44),
        to_tensor=True,
        grayscale=False,
        training_mode=True,
        balance_dataset_classes=50,
        augmentation_settings=AugmentationSettings.create_settings_with_name(
            "gaussian_noise_aug"
        ),
    )
    desc = SerializableDatasetDescriptor(dataset=test_db)
    print(desc)
    descriptor_dict = desc.to_dict()
    json_string = json.dumps(descriptor_dict)
    deserialized_dict = json.loads(json_string)
    new_descriptor = SerializableDatasetDescriptor.from_dict(deserialized_dict)
    pass
