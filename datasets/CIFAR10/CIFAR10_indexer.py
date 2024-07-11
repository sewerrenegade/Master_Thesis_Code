import sys
sys.path.append('/home/milad/Desktop/Master_Thesis/code/Master_Thesis_Code')
import os
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import CIFAR10
import json
from collections.abc import Iterable
import random
from datasets.indexer_scripts.indexer_abstraction import Indexer
SINGLETON_INSTANCE  = None

class CIFAR10_Indexer(Indexer):
    def __init__(self,CIFAR10_Path = "data/CIFAR10", perform_reindexing = False) -> None:
        self.path = CIFAR10_Path
        self.class_names = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.dict_path = f"{CIFAR10_Path}/metadata.json"
        if perform_reindexing or not self.does_if_meta_file_exists():
            self.create_meta_file()
        with open(self.dict_path, 'r') as json_file:
            loaded_dict = json.load(json_file)    
        from datasets.indexer_scripts.indexer_utils import process_deserialized_json
        self.train_per_class_indicies = process_deserialized_json(loaded_dict["train"])
        self.test_per_class_indicies = process_deserialized_json(loaded_dict["test"])
        self.classes= sorted(loaded_dict["classes"])
        self.train_class_count ,self.test_class_count = process_deserialized_json(loaded_dict["class_count"])
        self.train_size,self.test_size = loaded_dict["total_count"]
        
    def get_instance_level_indicies(self,training):
        indicies_list = []
        labels_list = []
        if training:
            indicies = self.train_per_class_indicies
            class_count = self.train_class_count
        else:
            indicies = self.test_per_class_indicies
            class_count = self.test_class_count
        for key, value in indicies.items():
            indicies_list.extend(value)
            labels_list.extend([key]*len(value))
                
        return list(zip(indicies_list,labels_list)),indicies,class_count
        
    def get_bag_level_indicies(self):
        raise NotImplementedError    
    @staticmethod
    def get_indexer():
        global SINGLETON_INSTANCE
        if SINGLETON_INSTANCE is None:
            SINGLETON_INSTANCE = CIFAR10_Indexer()
        return SINGLETON_INSTANCE
    
    def get_random_samples_of_class(self,target_class_s,training):
        if training:
            class_count = self.train_class_count
            indicies = self.train_per_class_indicies
        else:
            class_count = self.test_class_count
            indicies = self.test_per_class_indicies

        if not isinstance(target_class_s, Iterable):
            target_class_s = [target_class_s]
        instances = []
        for target_class in target_class_s:
            x1 = random.randint(0, class_count[target_class] - 1)
            x2 = target_class
            instances.append(indicies[x2][x1])
        return instances

    
    def get_per_class_count(self,classes,train_indicies,test_indicies):
        train_class_count={}
        test_class_count={}
        for class_name in classes:
            train_class_count[class_name] = len(train_indicies[class_name])
            test_class_count[class_name] = len(test_indicies[class_name])
        return train_class_count,test_class_count
    
    def get_classes(self,set_indicies):
        return list(set_indicies.keys())
    
    def does_if_meta_file_exists(self):
        return os.path.exists(self.dict_path)
    
    def create_meta_file(self):
        train_data = CIFAR10("data/", train=True, download=True)
        test_data= CIFAR10("data/", train=False, download=True)
        train_data_indicies = self.get_indices(train_data)
        test_data_indicies = self.get_indices(test_data)
        classes = self.get_classes(train_data_indicies)
        class_count = self.get_per_class_count(classes,train_data_indicies,test_data_indicies)
        train_test_indeices = {'train':train_data_indicies,'test':test_data_indicies,'class_count':class_count,'classes':classes, 'total_count':(len(train_data),len(test_data))}
        with open(self.dict_path, 'w+') as json_file:
            json.dump(train_test_indeices, json_file)
            

    def get_indices(self,dataset):
        set_size = len(dataset)
        class_indices = {}
        for data_index in range(set_size):
            _,label = dataset[data_index]
            try:
                class_indices[label].append(data_index)
            except KeyError:
                class_indices[label]=[data_index]
        return class_indices

if __name__ == "__main__":
    print(list(CIFAR10_Indexer().classes)[2])


