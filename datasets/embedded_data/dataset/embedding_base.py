from torch.utils.data import Dataset, random_split, Subset
import os
import numpy as np
from configs.global_config import GlobalConfig



class baseDataset(Dataset):
    def __init__(self, root_dir,training = True):
        self.root_dir = root_dir
        self.name = "EMBEDDING"
        self.training = training
        self.data,self.stats = self.load_file_data()
        self.class_indicies = self.get_class_indicies(self.data)
    
    def get_class_indicies(self,data):
        class_indicies  = {}
        for data_index in range(len(self)):
            x=int(data[data_index][1])
            try:
                class_indicies[x].append(data_index)
            except Exception as e:
                class_indicies[x] = [data_index]

        return class_indicies
    
    def load_file_data(self):
        data = np.load(f"{self.root_dir}{GlobalConfig.NAME_OF_LABELED_EMBEDDED_FEATURES}.npy",allow_pickle=True)
        stats = np.load(f"{self.root_dir}{GlobalConfig.NAME_OF_STATS_OF_EMBEDDED_FEATURES}.npz",allow_pickle=True)
        return data,stats

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data,image_label = self.data[index]
        return data, image_label
