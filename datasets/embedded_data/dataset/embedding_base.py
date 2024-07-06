from torch.utils.data import Dataset, random_split, Subset
import os
import numpy as np
import random
from datasets.base_dataset_abstraction import BaseDataset
from configs.global_config import GlobalConfig
from collections.abc import Iterable


class EmbeddingBaseDataset(BaseDataset):
    def __init__(self, root_dir,training = True,dataset_size = None):
        self.root_dir = root_dir
        self.training = training
        self.dataset_size = dataset_size
        self.data,self.stats = self.load_file_data()
        self.name = str(self.stats["name"])
        self.data_origin =  str(self.stats["generating_data"])
        self.classes = []
        self.class_indicies = self.get_class_indicies(self.data)
        self.indicies_list = self.build_smaller_dataset()
    
    def get_random_samples_from_class(self, class_name, number_of_instances):
        indicies = self.class_indicies[class_name]
        random_indicies = random.sample(indicies,number_of_instances)
        return self[random_indicies]

    
    def get_class_indicies(self,data):
        class_indicies  = {}
        for data_index in range(len(data)):
            x = int(data[data_index][1])
            try:
                class_indicies[x].append(data_index)
            except Exception as e:
                class_indicies[x] = [data_index]
                self.classes.append(x)
        return class_indicies
    
    def build_smaller_dataset(self):
        if self.dataset_size:
            class_size = int(self.dataset_size/len(self.classes))
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
        if self.dataset_size:
            return len(self.indicies_list)
        else:
            return len(self.data)

    def __getitem__(self, index):
        if not isinstance(index,Iterable):
            index = [index]
        if self.dataset_size:
            x = self.data[self.indicies_list[index]]
            embedded_data,image_label = x[:,0],x[:,1]
            return embedded_data, image_label
        else:
            x = self.data[index]
            embedded_data,image_label = x[:,0],x[:,1]
            return embedded_data, image_label

