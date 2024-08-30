from typing import Callable, Union
from datasets.base_dataset_abstraction import BaseDataset
from datasets.dataset_descriptor import SerializableDatasetDescriptor
from datasets.embedded_datasets.dataset.embedding_base import EmbeddingBaseDataset
from datasets.mil_dataset_abstraction import BaseMILDataset
from datasets.topological_datasets.create_bag_distance_matrix_dataset_from_MIL_dataset import (
    calculate_distance_matrix_of_MIL_dataset,
)
from distance_functions.functions.embedding_functions import EmbeddingFunction
import numpy as np


class TopoDatasetDescriptor:
    def __init__(
        self,
        name: str,
        dataset: Union[BaseDataset, BaseMILDataset],
        distance_function: object,
        nb_of_grouped_bags: int,
        embedding_function: EmbeddingFunction
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.distance_function = distance_function
        self.nb_of_grouped_bags = nb_of_grouped_bags
        self.embedding_function = embedding_function

    def to_dict(self):
        return (
            self.create_serialializable_descriptor_from_live_topo_ds_descriptor().to_dict()
        )

    def create_serialializable_descriptor_from_live_topo_ds_descriptor(self):
        topo_ds_decriptor = self
        return SerializableTopoDatasetDescriptor(
            name=topo_ds_decriptor.name,
            dataset=topo_ds_decriptor.dataset,
            distance_function_name=topo_ds_decriptor.distance_function.name,
            nb_of_grouped_bags=topo_ds_decriptor.nb_of_grouped_bags,
            transform_name=topo_ds_decriptor.embedding_function.name,
            transform_settings=topo_ds_decriptor.embedding_function.settings,
        )
    @staticmethod
    def ensure_float32(dict_array):
        for key,value in dict_array.items():
            if value.dtype != np.float32:
                dict_array[key] = value.astype(np.float32)
        return dict_array
    def generate_or_get_topo_dataset_from_descriptor(self, recalculate=False):
        from results.results_manager import ResultsManager

        results_manager = ResultsManager.get_manager()
        if not recalculate and results_manager.check_if_result_already_exists(self):
            pass
        else:
            bag_distance_matrix, bag_instance_order = calculate_distance_matrix_of_MIL_dataset(
                    dataset=self.dataset,
                    distance_function=self.distance_function,
                    embedding_function=self.embedding_function,
                    nb_of_grouped_bags=self.nb_of_grouped_bags,
                )
            
            results_manager.save_results(
                descriptor=self,
                results={
                    "bag_distance_matrix": bag_distance_matrix,
                    "bag_instance_order": bag_instance_order,
                },
            )
        distance_matrix_bag_dict,instance_order_in_bag_dict = results_manager.load_topo_dataset(self)
        return TopoDatasetDescriptor.ensure_float32(distance_matrix_bag_dict),instance_order_in_bag_dict


class SerializableTopoDatasetDescriptor:
    def __init__(
        self,
        name,
        dataset,
        distance_function_name,
        nb_of_grouped_bags,
        transform_name,
        transform_settings
    ):
        self.name = name
        self.dataset_serialisable_descriptor = SerializableDatasetDescriptor(
            dataset=dataset
        )
        self.number_of_bags_to_calc_dist = nb_of_grouped_bags
        self.distance_function_name = distance_function_name
        self.transform_name = transform_name
        self.transform_settings = transform_settings

    def __str__(self):
        return (
            f"Name: {self.name}\n"
            f"Dataset Name: {self.dataset_serialisable_descriptor.name}\n"
            f"Augmentation Settings: {self.dataset_serialisable_descriptor.augmentation_settings}\n"
            f"DINO Bloom: {self.dataset_serialisable_descriptor.dino_bloom}\n"
            f"Distance Function Name: {self.distance_function_name}\n"
            f"Nb of bags for dist calc: {self.number_of_bags_to_calc_dist}\n"
            f"Transform Name: {self.transform_name}"
            f"Transform Settings: {self.transform_settings}\n"
        )

    def to_dict(self):
        return {
            "dataset_dict": (
                self.dataset_serialisable_descriptor.to_dict()
                if hasattr(self.dataset_serialisable_descriptor, "to_dict")
                else self.dataset_dict
            ),
            "name": self.name,
            "number_of_bags_to_calc_dist": self.number_of_bags_to_calc_dist,
            "distance_function_name": self.distance_function_name,
            "transform_name": self.transform_name,
            "transform_settings": self.transform_settings,
        }

    @staticmethod
    def from_dict(data):
        return SerializableTopoDatasetDescriptor(
            name=data["name"],
            dataset_serialisable_descriptor=SerializableDatasetDescriptor.from_dict(data["dataset_dict"]),
            distance_function_name=data["distance_function_name"],
            transform_name=data["transform_name"],
            transform_settings=data["transform_settings"],
        )


def create_serialializable_descriptor_from_live_embedding_dataset(
    emb_dataset: EmbeddingBaseDataset,
) -> SerializableTopoDatasetDescriptor:
    return emb_dataset.get_serializable_embedding_descriptor()
