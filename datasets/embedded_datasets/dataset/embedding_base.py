from torch.utils.data import Dataset, random_split, Subset
import os
import numpy as np
import random
from datasets.base_dataset_abstraction import BaseDataset
from configs.global_config import GlobalConfig
from collections.abc import Iterable
from datasets.embedded_datasets.embeddings_manager import EMBEDDING_MANAGER_INSTANCE

class EmbeddingBaseDataset(Dataset):
    def __init__(self, embedding_id,balance_dataset_classes:int = None):
        self.embedding_id = embedding_id
        self.balance_dataset_classes = balance_dataset_classes
        self.embeddings, self.embedding_labels, self.embedding_descriptor = EMBEDDING_MANAGER_INSTANCE.load_embedding(self.embedding_id)
        assert self.embedding_labels is not None
        self.name = self.embedding_descriptor.name
        self.data_origin =   self.embedding_descriptor.dataset_name
        self.classes = []
        self.class_indicies = self.get_class_indicies()
        self.indicies_list = self.build_smaller_dataset()
    
    def get_random_samples_from_class(self, class_name, number_of_instances):
        indicies = self.class_indicies[class_name]
        random_indicies = random.sample(indicies,number_of_instances)
        return self[random_indicies]

    
    def get_class_indicies(self):
        class_indicies  = {}
        for data_index in range(self.embedding_labels.shape[0]):
            try:
                class_indicies[self.embedding_labels[data_index]].append(data_index)
            except Exception as e:
                class_indicies[self.embedding_labels[data_index]] = [data_index]
                self.classes.append(self.embedding_labels[data_index])
        return class_indicies
    
    def build_smaller_dataset(self):
        if self.balance_dataset_classes:
            class_size = self.balance_dataset_classes
            indicies = []
            for class_name in self.classes:
                indicies = self.class_indicies[class_name][:class_size]
                indicies.extend(indicies)
                self.class_indicies[class_name] = indicies
            return indicies

    def load_file_data(self):
        data = np.load(f"{self.root_dir}{GlobalConfig.NAME_OF_LABELED_EMBEDDED_FEATURES}.npy",allow_pickle=True)
        stats = np.load(f"{self.root_dir}{GlobalConfig.NAME_OF_STATS_OF_EMBEDDED_FEATURES}.npz",allow_pickle=True)
        return data,stats

    def __len__(self):
        if self.balance_dataset_classes:
            return len(self.indicies_list)
        else:
            return len(self.embedding_labels)

    def __getitem__(self, index):
        if not isinstance(index,Iterable):
            index = [index]
        if self.balance_dataset_classes:
            embedded_data,image_label = self.embeddings[self.indicies_list[index],:],self.embedding_labels[self.indicies_list[index]]
            return embedded_data, image_label
        else:
            embedded_data,image_label =self.embeddings[index,:],self.embedding_labels[index]
            return embedded_data, image_label

