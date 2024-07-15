import os
import pickle
import json
import hashlib
import sys
from typing import Union

import numpy as np
sys.path.append('/home/milad/Desktop/Master_Thesis/code/Master_Thesis_Code')
from configs.global_config import GlobalConfig
from datasets.embedded_datasets.generators.embedding_descriptor import EmbeddingDescriptor, SerializableEmbeddingDescriptor, create_serialializable_descriptor_from_live_descriptor
from datasets.image_augmentor import AugmentationSettings
from datetime import datetime



class EmbeddingManager:
    def __init__(self):
        self.storage_dir = GlobalConfig.EMBEDDING_DATA_FOLDER_PATH
        os.makedirs(self.storage_dir, exist_ok=True)
        self.descriptor_dir = os.path.join(self.storage_dir, 'descriptors')
        self.embedding_dir = os.path.join(self.storage_dir, 'embeddings')
        self.creation_time_tracker = os.path.join(self.storage_dir, 'creation_times')
        os.makedirs(self.descriptor_dir, exist_ok=True)
        os.makedirs(self.embedding_dir, exist_ok=True)
    _instance = None
    
    @classmethod
    def get_manager(cls) -> 'EmbeddingManager':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    def check_if_embedding_already_exists(self,descriptor:Union[EmbeddingDescriptor,SerializableEmbeddingDescriptor,dict,str]):
        if isinstance(descriptor,Union[EmbeddingDescriptor,SerializableEmbeddingDescriptor,dict]):
            _descriptor_id = self._calculate_descriptor_id(descriptor)
        else:
            _descriptor_id = descriptor
        descriptor_path = os.path.join(self.descriptor_dir, f"{_descriptor_id}.pkl")
        return os.path.exists(descriptor_path)
    
    def save_embedding(self, descriptor:EmbeddingDescriptor, embedding,embedding_label = None,embedding_stats = None):
        
        descriptor_id = self._calculate_descriptor_id(descriptor)
        descriptor_path = os.path.join(self.descriptor_dir, f"{descriptor_id}.pkl")
        embedding_path = os.path.join(self.embedding_dir, f"{descriptor_id}.npy")        

        descriptor_dict = descriptor.to_dict()
        with open(descriptor_path, 'wb') as f:
            pickle.dump(descriptor_dict, f)
        np.save(embedding_path, embedding)
        if embedding_label is not None:
            assert len(embedding) == len(embedding_label)
            embedding_label_path = os.path.join(self.embedding_dir, f"{descriptor_id}_label.npy")
            labels = np.array(embedding_label)
            np.save(embedding_label_path, labels)
              
        self.add_creation_time(descriptor_id)
        self.generate_descriptor_list_file()
        
    def load_embedding(self, descriptor):
        descriptor_dict = descriptor.to_dict()
        descriptor_id = self._calculate_descriptor_id(descriptor_dict)
        descriptor_path = os.path.join(self.descriptor_dir, f"{descriptor_id}.pkl")
        embedding_path = os.path.join(self.embedding_dir, f"{descriptor_id}.npy")
        embedding_label_path = os.path.join(self.embedding_dir, f"{descriptor_id}.npy")

        if os.path.exists(descriptor_path) and os.path.exists(embedding_path):
            with open(descriptor_path, 'rb') as f:
                embedding_descriptor = pickle.load(f)
            embedding = np.load(embedding_path)
            if os.path.exists(embedding_label_path):
                embedding_label = np.load(embedding_label_path)
            else:
                embedding_label = None
            
            return embedding,embedding_label,embedding_descriptor
        else:
            print(f"Warning! The embedding or descriptor you are looking for does not exist in the file system.")
            return None

    
    def get_creation_times(self):
        if os.path.exists(self.creation_time_tracker):
            with open(self.creation_time_tracker, 'rb') as f:
                creation_time_dict = pickle.load(f)
        return creation_time_dict
        
    def add_creation_time(self,descriptor_id):
        creation_time = datetime.now()
        creation_time = str(creation_time.replace(microsecond=0))
        
        if os.path.exists(self.creation_time_tracker):
            with open(self.creation_time_tracker, 'rb') as f:
                descriptor_dict = pickle.load(f)
        else:
            with open(self.creation_time_tracker, 'w'):
                pass 
            descriptor_dict = {descriptor_id:creation_time}
            
        descriptor_dict[descriptor_id] = creation_time
        with open(self.creation_time_tracker, 'wb') as f:
            pickle.dump(descriptor_dict, f)
            
    def remove_creation_time(self,descriptor_id):
        creation_time = datetime.now()
        creation_time = str(creation_time.replace(microsecond=0))
        if not os.path.exists(self.creation_time_tracker):
            raise FileExistsError("The time of creation file does not exist")
        else:
            with open(os.path.join(self.descriptor_dir, self.creation_time_tracker), 'rb') as f:
                descriptor_dict = pickle.load(f)
            del descriptor_dict[descriptor_id]
        with open(self.creation_time_tracker, 'wb') as f:
            pickle.dump(descriptor_dict, f)    

        
    def delete_embedding(self, descriptor):
        descriptor_dict = descriptor.to_dict()
        descriptor_id = self._calculate_descriptor_id(descriptor_dict)
        descriptor_path = os.path.join(self.descriptor_dir, f"{descriptor_id}.pkl")
        embedding_path = os.path.join(self.embedding_dir, f"{descriptor_id}.npy")
        self.remove_creation_time(descriptor_id)
        if os.path.exists(descriptor_path):
            os.remove(descriptor_path)
        else:
            print(f"Descriptor file not found: {descriptor_path}")

        if os.path.exists(embedding_path):
            os.remove(embedding_path)
            self.generate_descriptor_list_file()
        else:
            print(f"Embedding file not found: {embedding_path}")

    
    def query_descriptors_lambda(self, condition):
        matching_descriptors = []
        for filename in os.listdir(self.descriptor_dir):
            try:
                if filename.endswith(".pkl"):
                    with open(os.path.join(self.descriptor_dir, filename), 'rb') as f:
                        descriptor_dict = pickle.load(f)
                        descriptor = SerializableEmbeddingDescriptor.from_dict(descriptor_dict)
                        if condition(descriptor):
                            matching_descriptors.append(descriptor)
            except Exception as e:
                print(f"Was not able to query embedding {filename} not adding it to search results")
                print(e)
                
        return matching_descriptors

    def query_descriptors(self, **query):
        matching_descriptors = []
        for filename in os.listdir(self.descriptor_dir):
            if filename.endswith(".pkl"):
                with open(os.path.join(self.descriptor_dir, filename), 'rb') as f:
                    descriptor_dict = pickle.load(f)
                    if all(descriptor_dict.get(k) == v for k, v in query.items()):
                        matching_descriptors.append(SerializableEmbeddingDescriptor.from_dict(descriptor_dict))
        return matching_descriptors
    
    def generate_descriptor_list_file(self):
        creation_time_dict = self.get_creation_times()
        """Generate a text file listing all descriptors."""
        list_file_path = os.path.join(self.storage_dir, 'descriptor_list.txt')

        with open(list_file_path, 'w') as f:
            f.write("List of SerializableEmbeddingDescriptors:\n")
            f.write("----------------------------------------\n\n")

            for filename in os.listdir(self.descriptor_dir):
                if filename.endswith(".pkl"):
                    with open(os.path.join(self.descriptor_dir, filename), 'rb') as f_desc:
                        descriptor_dict = pickle.load(f_desc)
                        descriptor = SerializableEmbeddingDescriptor.from_dict(descriptor_dict)
                        embedding_id = self._calculate_descriptor_id(descriptor)
                        f.write(f"Descriptor ID: {filename}\n")
                        f.write(f"Time of Creation: {creation_time_dict[embedding_id]}\n")
                        f.write(f"{descriptor}\n")
                        f.write("----------------------------------------\n\n")

    def _calculate_descriptor_id(self, descriptor: Union[EmbeddingDescriptor,SerializableEmbeddingDescriptor,dict]):
        """Generate a unique ID based on the descriptor's content."""
        if isinstance(descriptor,EmbeddingDescriptor):
            descriptor= create_serialializable_descriptor_from_live_descriptor(descriptor).to_dict()
        elif isinstance(descriptor,SerializableEmbeddingDescriptor):
            descriptor = descriptor.to_dict()
        elif isinstance(descriptor,dict):
            descriptor = descriptor
        else:
            raise TypeError(f"Descriptor should be of type Union[EmbeddingDescriptor,SerializableEmbeddingDescriptor,dict]")
        
        descriptor_str = json.dumps(descriptor, sort_keys=True)
        return hashlib.sha256(descriptor_str.encode('utf-8')).hexdigest()
    