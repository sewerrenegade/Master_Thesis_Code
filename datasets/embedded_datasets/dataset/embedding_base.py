from torch.utils.data import Dataset
import numpy as np
import random
from configs.global_config import GlobalConfig
from collections.abc import Iterable

from datasets.dataset_factory import BASE_MODULES as DATA_SET_MODULES

class EmbeddingBaseDataset(Dataset):
    def __init__(self, embedding_id,balance_dataset_classes:int = None):
        self.embedding_id = embedding_id
        self.balance_dataset_classes = balance_dataset_classes
        from results.results_manager import ResultsManager
        self.embeddings, self.embedding_labels, self.embedding_descriptor = ResultsManager.get_manager().load_embedding(self.embedding_id)
        assert self.embedding_labels is not None
        from datasets.embedded_datasets.generators.embedding_descriptor import SerializableEmbeddingDescriptor
        self.embedding_descriptor = SerializableEmbeddingDescriptor.from_dict(self.embedding_descriptor)
        self.name = self.embedding_descriptor.name
        self.data_origin =   self.embedding_descriptor.dataset_name
        self.augmentation_settings = self.embedding_descriptor.augmentation_settings
        self.classes = []
        self.class_indicies = self.get_class_indicies()
        self.indicies_list = self.build_smaller_dataset()
    
    def get_serializable_embedding_descriptor(self):
        return self.embedding_descriptor

    def get_dataset_origin(self,flatten):
        emb_descriptor = self.embedding_descriptor
        og_db = DATA_SET_MODULES.get(emb_descriptor.dataset_name,None)
        return og_db(training_mode = True,gpu= False,numpy = True,flatten = flatten,balance_dataset_classes = emb_descriptor.dataset_sampling)
        
    def get_random_samples_from_class(self, class_name, number_of_instances):
        indicies = self.class_indicies[class_name]
        random_indicies = random.sample(indicies,number_of_instances)
        return self[random_indicies]
    
    def get_random_instances_from_all_classes(self, number_of_instances):
        instances_dict = {}
        for class_name in self.classes:
            instances_dict[class_name] = self.get_random_samples_from_class(
                class_name, number_of_instances
            )[0]
        return instances_dict
    
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

