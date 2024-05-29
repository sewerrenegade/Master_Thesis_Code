from torch.utils.data import Dataset, random_split, Subset
import os
import numpy as np
from configs.global_config import GlobalConfig



class baseDataset(Dataset):
    def __init__(self, root_dir,training = True,dataset_size = None):
        self.root_dir = root_dir
        self.name = "EMBEDDING"
        self.training = training
        self.dataset_size = dataset_size
        self.data,self.stats = self.load_file_data()
        self.classes = []
        self.class_indicies = self.get_class_indicies(self.data)
        self.indicies_list = self.build_smaller_dataset()
    
    def get_class_indicies(self,data):
        class_indicies  = {}
        for data_index in range(len(data)):
            x=int(data[data_index][1])
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
            for mnist_class in self.classes:
                indicies.extend(self.class_indicies[mnist_class][:class_size])
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
        if self.dataset_size:
            image_data,image_label = self.data[self.indicies_list[index]]
            return image_data, image_label
        else:
            data,image_label = self.data[index]
            return data, image_label

