from torch.utils.data import Dataset, random_split, Subset
import os
import numpy as np
from configs.global_config import GlobalConfig



class baseDataset(Dataset):
    def __init__(self, root_dir,training = True):
        self.root_dir = root_dir
        self.name = "EMBEDDING"
        self.training = training
        self.data,self.stats = self.load_file_data(root_dir)
    
    def load_file_data(self):
        data = np.load(f"{self.root_dir}{GlobalConfig.NAME_OF_LABELED_EMBEDDED_FEATURES}.npy")
        stats = np.load(f"{self.root_dir}{GlobalConfig.NAME_OF_STATS_OF_EMBEDDED_FEATURES}.npz")
        return data,stats

    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data,image_label = self.data[index]
        return data, image_label
