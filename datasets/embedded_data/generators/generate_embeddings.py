import sys
import os
sys.path.append('/home/milad/Desktop/Master_Thesis/code/Master_Thesis_Code')
from configs.global_config import GlobalConfig
import umap
import numpy as np
from datasets.MNIST.MNIST_base import BaseDataset as MNIST_baseDataset
from typing import Callable,Union
from torch.utils.data import Dataset
import torch
from dataclasses import dataclass
import time
import random

PATH_TO_MNIST_EMBEDDINGS  = "data/MINST/embeddings/"

@dataclass
class EmbeddingDescriptor:
    name: str
    dataset: Dataset
    downprojection_function: Union[Callable,str]

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
        if isinstance(tmp_data, torch.Tensor):
            tmp_data = tmp_data.numpy()
        all_data.append(tmp_data.flatten())
        labels.append(data[1])
    x = np.array(all_data)
    embeddings = transform(x)
    stats_dic = get_stats_from_embedding(name,embeddings,base_database.name)
    folder_path = f"data/{base_database.name}/embeddings/{name}/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    end_time = time.time()
    stats_dic["time_taken_seconds"] = end_time - start_time
    save_embeddings(embeddings,labels,stats_dic,folder_path)
    return folder_path


def get_stats_from_embedding(name,embeddings,generating_data_name):#gets stats on each feature
    return {"min_values" : np.min(embeddings, axis=0),
    "max_values" : np.max(embeddings, axis=0),
    "mean_values" : np.mean(embeddings, axis=0),
    "std_values" : np.std(embeddings, axis=0),
    "name": name,
    "generating_data": generating_data_name}


def generate_embedding_from_descriptor(decriptor : EmbeddingDescriptor):
    return generate_embeddings_for_dataset(decriptor.name,decriptor.dataset,decriptor.downprojection_function)


if __name__ == '__main__':
    #UMAP EXAMPLE
    down_projection_factor = 25
    dataset = MNIST_baseDataset(True)
    data_dim = np.prod(np.array(dataset[0][0].shape))
    down_proj_dim = int(data_dim/down_projection_factor)
    print(f"downprojecting from {data_dim} to {down_proj_dim} dimensions")
    embeddings = generate_embeddings_for_dataset(dataset,umap.UMAP(n_components=down_proj_dim).fit_transform,name = f"PCA_{down_proj_dim}")
