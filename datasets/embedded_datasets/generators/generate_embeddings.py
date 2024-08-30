import numpy as np
import time

from datasets.dataset_transforms import ToNPTransform

def generate_embeddings_for_dataset(name, base_database,transform):
    start_time = time.time()
    all_data = []
    labels = []
    size = len(base_database)
    for data_index in range(size):
        tmp_data,tmp_label = base_database[data_index]
        if hasattr(tmp_data,"numpy"):
           tmp_data = tmp_data.numpy()
        all_data.append(tmp_data)
        labels.append(tmp_label)
        print(f"{100*(data_index+1)/size}%")
    expected_shape = all_data[0].shape
    for i, vec in enumerate(all_data):
        if vec.shape != expected_shape:
            print(f"Vector at index {i} has shape {vec.shape}, expected {expected_shape}")
    
    x = ToNPTransform()(all_data)
    #x = x.reshape(x.shape[0], -1)
    if transform is not None:
        embeddings = transform(x)
    else:
        embeddings = x
    stats_dic = {}#get_stats_from_embedding(name,embeddings,base_database.name)
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
