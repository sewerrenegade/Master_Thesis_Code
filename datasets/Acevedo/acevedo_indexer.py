
import sys
sys.path.append('/home/milad/Desktop/Master_Thesis/code/Master_Thesis_Code')
from datasets.indexer_scripts.indexer_abstraction import Indexer
import os
import json

SINGLETON_INSTANCE= None

class AcevedoIndexer(Indexer):
    def __init__(self,path_to_acevedo = "data/Acevedo/processed_images/") -> None:
        super().__init__()
        self.path_to_acevedo = path_to_acevedo
        self.dict_path = f"{path_to_acevedo}/metadata.json"
        self.get_meta_data()
        self.per_class_paths_dict, self.per_class_count, self.classes,self.total_size = self.get_meta_data()
        pass
    def get_path_to_data_folder(self):
        return self.path_to_acevedo
    def convert_label_from_int_to_str_or_viceversa(self,label):
        if isinstance(label, str):
            return self.classes.index(label)
        elif isinstance(label, int):
            return self.classes[label]
        else:
            raise ValueError
        
    def get_instance_level_indicies(self,training=True):
        indicies_list = []
        labels_list = []            
        for key, value in self.per_class_paths_dict.items():
            indicies_list.extend(value)
            labels_list.extend([key]*len(value))
        return list(zip(indicies_list,labels_list)), self.per_class_paths_dict,self.per_class_count
    
    def get_bag_level_indicies(self):
        raise NotImplementedError  
      
    def get_meta_data(self):
        if os.path.exists(self.dict_path):
            with open(self.dict_path, 'r') as json_file:
                loaded_dict = json.load(json_file)
            meta_dict =  loaded_dict
        else:
            meta_dict = self.create_metadata_file()
        return meta_dict['indicies'],meta_dict['class_count'],meta_dict['classes'],meta_dict['total_count']


    def create_metadata_file(self):
        class_to_paths = {}
        total_count = 0
        for class_name in os.listdir(self.path_to_acevedo):
            class_folder = os.path.join(self.path_to_acevedo, class_name)
            if os.path.isdir(class_folder):
                image_paths = []
                for root, _, files in os.walk(class_folder):
                    for file in files:
                        if file.endswith('.jpg'):
                            relative_path = os.path.relpath(os.path.join(root, file), self.path_to_acevedo)
                            image_paths.append(relative_path)
                            total_count = total_count + 1
                class_to_paths[class_name] = image_paths
        classes = list(class_to_paths.keys())
        class_count = self.calc_class_count(classes,class_to_paths)
        meta_data = {'indicies':class_to_paths,'class_count':class_count,'classes':classes, 'total_count':total_count}
        with open(self.dict_path, 'w+') as json_file:
            json.dump(meta_data, json_file)
        return meta_data
            
    def calc_class_count(self,classes,train_indicies):
        train_class_count={}
        for class_name in classes:
            train_class_count[class_name] = len(train_indicies[class_name])
        return train_class_count
    @staticmethod
    def get_indexer():
        global SINGLETON_INSTANCE
        if SINGLETON_INSTANCE is None:
            SINGLETON_INSTANCE = AcevedoIndexer()
        return SINGLETON_INSTANCE 
          
    def get_per_class_count(self,classes,train_indicies,test_indicies = []):
        train_class_count={}
        for class_name in classes:
            train_class_count[class_name] = len(train_indicies[class_name])
        
        return train_class_count
            
    def get_random_samples_of_class(self, class_name, number_of_instances):
        
        raise NotImplementedError("Subclass must implement abstract method")
        return
if __name__ == "__main__":
    x = AcevedoIndexer()
    x.get_instance_level_indicies()