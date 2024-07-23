import os
import pickle
import json
import hashlib
import sys
from typing import Union
from datetime import datetime
import numpy as np


sys.path.append('/home/milad/Desktop/Master_Thesis/code/Master_Thesis_Code')
from configs.global_config import GlobalConfig
from datasets.embedded_datasets.generators.embedding_descriptor import EmbeddingDescriptor, SerializableEmbeddingDescriptor

from results.metrics_descriptor import MetricsDescriptor, SerializableMetricsDescriptor



class ResultsManager:
    _instance = None
    def __init__(self):
        self.ensure_folder_heirarchy_exists()
        self.id_dict = self.get_serialized_id()
        self.tracked_object_types = Union[EmbeddingDescriptor,SerializableEmbeddingDescriptor,MetricsDescriptor,SerializableMetricsDescriptor,dict,str]
    
    def ensure_folder_heirarchy_exists(self):
        self.storage_dir = GlobalConfig.RESULTS_DATA_FOLDER_PATH
        self.descriptor_dir = os.path.join(self.storage_dir, 'descriptors')
        self.metrics_dir = os.path.join(self.storage_dir, 'metrics')
        self.embeddings_dir = os.path.join(self.storage_dir, 'embeddings')
        self.creation_time_tracker = os.path.join(self.storage_dir, 'creation_times')
        self.path_to_saved_file = os.path.join(self.storage_dir, 'saved_id_info.json')
        os.makedirs(self.storage_dir, exist_ok=True)
        os.makedirs(self.descriptor_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.embeddings_dir, exist_ok=True)

            
    def get_relavent_dir(self,identifier):
        assert isinstance(identifier,self.tracked_object_types)
        if isinstance(identifier,(EmbeddingDescriptor,SerializableEmbeddingDescriptor)):
            return self.embeddings_dir
        if isinstance(identifier,(MetricsDescriptor,SerializableMetricsDescriptor)):
            return self.metrics_dir
        if isinstance(identifier,dict):
            id = self.calculate_descriptor_id(identifier)
            result_type = self.id_dict[id]
            if result_type == "metric":
                return self.metrics_dir
            elif result_type == "embeddings":
                return self.embeddings_dir
            else:
                raise ValueError("could not find discriptor directors")
        raise ValueError("could not find discriptor directors")
    
    def save_results(self,descriptor,results): 
        assert isinstance(descriptor,self.tracked_object_types)       
        if isinstance(descriptor,(EmbeddingDescriptor)):
            id = self.save_embedding(descriptor=descriptor,**results)
            result_type = "embedding"
        if isinstance(descriptor,MetricsDescriptor):
            id = self.save_metric(descriptor=descriptor,**results)
            result_type = "metric"
        self.add_results_to_tracker(id,result_type)
        
    def save_metric(self, descriptor, metrics_dict):
        assert isinstance(descriptor,self.tracked_object_types)   
        descriptor_id = self.calculate_descriptor_id(descriptor)
        descriptor_path = os.path.join(self.descriptor_dir, f"{descriptor_id}.pkl")
        metric_path = os.path.join(self.metrics_dir, f"{descriptor_id}.npz")        
        descriptor_dict = descriptor.to_dict()
        with open(descriptor_path, 'wb') as f:
            pickle.dump(descriptor_dict, f)  
        np.savez(metric_path, **metrics_dict)
        # with open(metric_path,'w') as f:
        #     f.write(self.serialize_metrics(metrics_dict))            
        self.add_creation_time(descriptor_id)
        self.generate_descriptor_list_file()
        return descriptor_id
      
    def load_metric(self,descriptor):
        assert isinstance(descriptor,self.tracked_object_types)   
        descriptor_id = self.calculate_descriptor_id(descriptor)
        metric_path = os.path.join(self.metrics_dir, f"{descriptor_id}.npz")
        # with open(metric_path, "r") as f:
        #     serialized_metrics = f.read()
        # deserialized_metrics = self.deserialize_metrics(serialized_metrics)
        deserialized_metrics = np.load(metric_path)
        return deserialized_metrics
        
    def serialize_metrics(self,metrics_dict):
        serializable_dict = {}
        
        for key, value in metrics_dict.items():
            str_key = str(key)  # Ensure the key is a string
            
            if isinstance(value, np.ndarray):
                serializable_dict[str_key] = {
                    "_type": "ndarray",
                    "value": value.tolist()
                }
            elif isinstance(value, (np.integer, np.floating)):
                serializable_dict[str_key] = {
                    "_type": "np_scalar",
                    "value": value.item()
                }
            else:
                serializable_dict[str_key] = value

        return json.dumps(serializable_dict)

    def deserialize_metrics(self,serialized_str):
        metrics_dict = json.loads(serialized_str)
        deserialized_dict = {}

        for key, value in metrics_dict.items():
            if isinstance(value, dict) and "_type" in value:
                if value["_type"] == "ndarray":
                    deserialized_dict[key] = np.array(value["value"])
                elif value["_type"] == "np_scalar":
                    deserialized_dict[key] = np.array(value["value"])
            else:
                deserialized_dict[key] = value

        return deserialized_dict
    
    def save_embedding(self, descriptor, embedding,embedding_label = None,embedding_stats = None):
        assert isinstance(descriptor,self.tracked_object_types)   
        descriptor_id = self.calculate_descriptor_id(descriptor)
        descriptor_path = os.path.join(self.descriptor_dir, f"{descriptor_id}.pkl")
        embedding_path = os.path.join(self.embeddings_dir, f"{descriptor_id}.npy")        

        descriptor_dict = descriptor.to_dict()
        with open(descriptor_path, 'wb') as f:
            pickle.dump(descriptor_dict, f)
        np.save(embedding_path, embedding)
        if embedding_label is not None:
            assert len(embedding) == len(embedding_label)
            embedding_label_path = os.path.join(self.embeddings_dir, f"{descriptor_id}_label.npy")
            labels = np.array(embedding_label)
            np.save(embedding_label_path, labels)
              
        self.add_creation_time(descriptor_id)
        self.generate_descriptor_list_file()
        return descriptor_id
    
    def add_results_to_tracker(self,id,result_type):
        self.id_dict[id] = result_type
        self.save_live_ids()
        
    def remove_id_from_tracker(self,id,result_type):
        del self.id_dict[id]
        self.save_live_ids()
        
    @classmethod
    def get_manager(cls) -> 'ResultsManager':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def check_if_result_already_exists(self,descriptor:Union[EmbeddingDescriptor,SerializableEmbeddingDescriptor,MetricsDescriptor,SerializableMetricsDescriptor,dict,str]):
        assert isinstance(descriptor,self.tracked_object_types)   
        if isinstance(descriptor,Union[EmbeddingDescriptor,SerializableEmbeddingDescriptor,MetricsDescriptor,SerializableMetricsDescriptor,EmbeddingDescriptor,SerializableEmbeddingDescriptor,dict]):
            _descriptor_id = self.calculate_descriptor_id(descriptor)
        else:
            _descriptor_id = descriptor
        descriptor_path = os.path.join(self.descriptor_dir, f"{_descriptor_id}.pkl")
        return os.path.exists(descriptor_path)
    
   
        
    def load_embedding(self, descriptor):
        assert isinstance(descriptor,self.tracked_object_types)
        if isinstance(descriptor,str):
            descriptor_id = descriptor
        else:
            descriptor_dict = descriptor.to_dict()
            descriptor_id = self.calculate_descriptor_id(descriptor_dict)
        descriptor_path = os.path.join(self.descriptor_dir, f"{descriptor_id}.pkl")
        embedding_path = os.path.join(self.embeddings_dir, f"{descriptor_id}.npy")
        embedding_label_path = os.path.join(self.embeddings_dir, f"{descriptor_id}_label.npy")

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
    
    def get_serialized_id(self):
        path_to_saved_file = os.path.join(self.storage_dir, 'saved_id_info.json')
        if os.path.exists(path_to_saved_file):
            with open(path_to_saved_file,'r') as file:
                id_dict = json.load(file)
        else:
            print("Could not find the meta id tracker file")    
            id_dict = {}
            with open(path_to_saved_file, 'w') as file:
                json.dump(id_dict, file, indent=4)
        return id_dict
    
    def save_live_ids(self):
        with open(self.path_to_saved_file, 'w') as file:
            json.dump(self.id_dict, file, indent=4)
            
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
        descriptor_id = self.calculate_descriptor_id(descriptor_dict)
        descriptor_path = os.path.join(self.descriptor_dir, f"{descriptor_id}.pkl")
        embedding_path = os.path.join(self.embeddings_dir, f"{descriptor_id}.npy")
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

    
    def query_metrics_lambda(self, condition):
        matching_descriptors = []
        for filename in os.listdir(self.descriptor_dir):
            try:
                if filename.endswith(".pkl"):
                    if self.id_dict[os.path.splitext(filename)[0]] == "metric":
                        with open(os.path.join(self.descriptor_dir, filename), 'rb') as f:
                            descriptor_dict = pickle.load(f)
                            if condition(descriptor_dict):
                                matching_descriptors.append(descriptor_dict)
            except Exception as e:
                print(f"Was not able to query embedding {filename} not adding it to search results")
                if filename.endswith(".pkl"):
                    if self.id_dict[os.path.splitext(filename)[0]] == "metric":
                        with open(os.path.join(self.descriptor_dir, filename), 'rb') as f:
                            descriptor_dict = pickle.load(f)
                            print(descriptor_dict)
                print(e)
            
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
                        embedding_id = os.path.splitext(filename)[0]
                        f.write(f"Descriptor ID: {filename}\n")
                        f.write(f"Time of Creation: {creation_time_dict[embedding_id]}\n")
                        f.write(f"{descriptor_dict}\n")
                        f.write("----------------------------------------\n\n")

    def calculate_descriptor_id(self, descriptor: Union[EmbeddingDescriptor,SerializableEmbeddingDescriptor,MetricsDescriptor,SerializableMetricsDescriptor,dict,str]):
        """Generate a unique ID based on the descriptor's content."""
        if isinstance(descriptor,(EmbeddingDescriptor,SerializableEmbeddingDescriptor,MetricsDescriptor,SerializableMetricsDescriptor)):
            descriptor= descriptor.to_dict()
        elif isinstance(descriptor,dict):
            descriptor = descriptor
        elif isinstance(descriptor,str):
            return descriptor
        else:
            raise TypeError(f"Descriptor should be of type (EmbeddingDescriptor,SerializableEmbeddingDescriptor,MetricsDescriptor,SerializableMetricsDescriptor)")
        
        descriptor_str = json.dumps(descriptor, sort_keys=True)
        return hashlib.sha256(descriptor_str.encode('utf-8')).hexdigest()
    