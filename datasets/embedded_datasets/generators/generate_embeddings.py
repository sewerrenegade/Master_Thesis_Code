import sys
import os

from datasets.embedded_datasets.embeddings_manager import EmbeddingManager
sys.path.append('/home/milad/Desktop/Master_Thesis/code/Master_Thesis_Code')
from configs.global_config import GlobalConfig
import umap
import numpy as np
from datasets.MNIST.MNIST_base import BaseDataset as MNIST_baseDataset
import time
import random
from datasets.embedded_datasets.generators.embedding_descriptor import EmbeddingDescriptor



def save_embeddings(embeddings,labels, stats,base_path):
    assert len(embeddings) == len(labels)
    os.makedirs(base_path, exist_ok=True)
    x = list(zip(embeddings,np.array(labels)))
    random.shuffle(x)
    to_serialise = np.array(x,dtype=object)
    save_path = f"{base_path}{GlobalConfig.NAME_OF_LABELED_EMBEDDED_FEATURES}.npy"
    np.save(save_path,to_serialise)
    stats_path = f"{base_path}{GlobalConfig.NAME_OF_STATS_OF_EMBEDDED_FEATURES}.npz"
    np.savez(stats_path, **stats)


def generate_embeddings_for_dataset(name, base_database,transform):
    start_time = time.time()
    all_data = []
    labels = []
    for data in base_database:
        tmp_data = data[0]
        all_data.append(tmp_data)
        labels.append(data[1])
    expected_shape = all_data[0].shape
    for i, vec in enumerate(all_data):
        if vec.shape != expected_shape:
            print(f"Vector at index {i} has shape {vec.shape}, expected {expected_shape}")
    x = np.array(all_data)
    embeddings = transform(x)
    stats_dic = get_stats_from_embedding(name,embeddings,base_database.name)
    end_time = time.time()
    stats_dic["time_taken_seconds"] = end_time - start_time
    return embeddings,labels,stats_dic


def get_stats_from_embedding(name,embeddings,generating_data_name):#gets stats on each feature
    return {"min_values" : np.min(embeddings, axis=0),
    "max_values" : np.max(embeddings, axis=0),
    "mean_values" : np.mean(embeddings, axis=0),
    "std_values" : np.std(embeddings, axis=0),
    "name": name,
    "generating_data": generating_data_name}


def generate_embedding_from_descriptor(descriptor : EmbeddingDescriptor):
    embeddings,labels,stats_dic = generate_embeddings_for_dataset(descriptor.name,descriptor.dataset,descriptor.downprojection_function(**descriptor.downprojection_function_settings).fit_transform)
    embeddings_manager = EmbeddingManager.get_manager()
    embeddings_manager.save_embedding(descriptor,embeddings,embedding_label = labels,embedding_stats = stats_dic)

if __name__ == '__main__':
    #UMAP EXAMPLE
    down_projection_factor = 25
    dataset = MNIST_baseDataset(True)
    data_dim = np.prod(np.array(dataset[0][0].shape))
    down_proj_dim = int(data_dim/down_projection_factor)
    print(f"downprojecting from {data_dim} to {down_proj_dim} dimensions")
    embeddings = generate_embeddings_for_dataset(dataset,umap.UMAP(n_components=down_proj_dim).fit_transform,name = f"PCA_{down_proj_dim}")
