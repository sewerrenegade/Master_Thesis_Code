import os
import pickle
import json
import hashlib
import sys
import numpy as np
sys.path.append('/home/milad/Desktop/Master_Thesis/code/Master_Thesis_Code')
from datasets.embedded_data.generators.embedding_descriptor import SerializableEmbeddingDescriptor
from datasets.image_augmentor import AugmentationSettings
from datetime import datetime

class EmbeddingManager:
    def __init__(self, storage_dir = "data/EMBEDDING/"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self.descriptor_dir = os.path.join(storage_dir, 'descriptors')
        self.embedding_dir = os.path.join(storage_dir, 'embeddings')
        self.creation_time_tracker = os.path.join(storage_dir, 'creation_times')
        os.makedirs(self.descriptor_dir, exist_ok=True)
        os.makedirs(self.embedding_dir, exist_ok=True)

    def save_embedding(self, descriptor, embedding):
        descriptor_dict = descriptor.to_dict()
        descriptor_id = self._calculate_descriptor_id(descriptor_dict)
        descriptor_path = os.path.join(self.descriptor_dir, f"{descriptor_id}.pkl")
        embedding_path = os.path.join(self.embedding_dir, f"{descriptor_id}.npy")

        
        with open(descriptor_path, 'wb') as f:
            pickle.dump(descriptor_dict, f)

        np.save(embedding_path, embedding)
        self.add_creation_time(descriptor_id)
        self.generate_descriptor_list_file()
    
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

    def load_embedding(self, descriptor):
        descriptor_dict = descriptor.to_dict()
        descriptor_id = self._calculate_descriptor_id(descriptor_dict)
        descriptor_path = os.path.join(self.descriptor_dir, f"{descriptor_id}.pkl")
        embedding_path = os.path.join(self.embedding_dir, f"{descriptor_id}.npy")

        if os.path.exists(descriptor_path) and os.path.exists(embedding_path):
            with open(descriptor_path, 'rb') as f:
                descriptor_data = pickle.load(f)
            embedding = np.load(embedding_path)
            return embedding
        else:
            print(f"Warning! The embedding or descriptor you are looking for does not exist in the file system.")
            return None

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

    def _calculate_descriptor_id(self, descriptor_dict):
        """Generate a unique ID based on the descriptor's content."""
        if isinstance(descriptor_dict,SerializableEmbeddingDescriptor):
            descriptor_dict = descriptor_dict.to_dict()
        descriptor_str = json.dumps(descriptor_dict, sort_keys=True)
        return hashlib.sha256(descriptor_str.encode('utf-8')).hexdigest()

# # Example usage
# storage_dir = 'embeddings'
# manager = EmbeddingManager(storage_dir)

# descriptor = SerializableEmbeddingDescriptor(
#     name='PHATE_2',
#     dataset_name='SCEMILA/image_data',
#     dataset_sampling=100,
#     augmentation_settings=AugmentationSettings(
#         dataset_name='SCEMILA/image_data',
#         color_jitter=False,
#         sharpness_aug=False,
#         horizontal_flip_aug=False,
#         vertical_flip_aug=False,
#         rotation_aug=True,
#         translation_aug=False,
#         gaussian_blur_aug=False,
#         gaussian_noise_aug=False,
#         auto_generated_notes='+/-180 degree rotation augmentation applied'
#     ),
#     dino_bloom=False,
#     transform_name=None,
#     transform_settings={'knn': 110, 't': 'auto', 'n_components': 2, 'decay': 40}
# )

# embedding = np.random.rand(100, 40)  # Example embedding

# # Save the embedding
# manager.save_embedding(descriptor, embedding)

# # Load the embedding
# loaded_embedding = manager.load_embedding(descriptor)
# print(loaded_embedding)

# # Query descriptors using a lambda expression
# query_result = manager.query_descriptors_lambda(lambda desc: desc.transform_name == "PHATE" and desc.transform_settings['n_components'] == 2)
# print("Query result:", query_result)
