from datasets.MNIST.MNIST_base import MNIST_Dataset_Referencer
from datasets.MNIST.MNIST_base import BaseDataset as MNISTbase
from datasets.embedded_data.dataset.embedding_base import EmbeddingBaseDataset as EMNIST
from models.topology_models.mahalanobis import MahalanobisDistanceCalculator
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import os
from configs.global_config import GlobalConfig

order_of_embeddings = ["ISOMAP","PCA","TSNE","UMAP"]
MSNIT_DATASET= MNISTbase(training= True)
EMBEDDED_MNIST_DATASET = [EMNIST(f"data/MNIST/embeddings/{embeddings}/") for embeddings in order_of_embeddings]
NUMBER_OF_SAMPLES_PER_CLASS = 100

def _compute_euclidean_distance(x):
    x_flat = x#x.view(x.size(0), -1)
    distances =  np.linalg.norm(x_flat[0] - x_flat[1])
    return distances

    
def compute_all_distances(x):
    assert len(x)==2
    out_dic =_compute_euclidean_distance(x)
    return out_dic

def compute_mean_var(class1_samples,class2_samples):
    samples = zip(class1_samples,class2_samples)
    distances= []
    for input_pair in samples:
        distances.append(compute_all_distances(input_pair))
    mean = np.mean(distances,axis=0)
    var = np.var(distances,axis=0)
    return mean,var
    
def compute_distance_between_class_i_with_all_others(data,class_key):
    distance_calculator = MahalanobisDistanceCalculator(data[class_key])
    random.shuffle(class2)
    return compute_mean_var(class1,class2)




def calculate_class_mahalanobis_distance_embedded_MNIST():
    number_of_embeddings = len(order_of_embeddings)
    classes = MNIST_Dataset_Referencer.INDEXER.classes
    class_count = len(classes)
    mean_distance_matrix = np.zeros((number_of_embeddings,class_count,class_count))
    var_distance_matrix = np.zeros((number_of_embeddings,class_count,class_count))
    for embedding_index in range(number_of_embeddings):

        data = load_n_samples_from_EMNIST(NUMBER_OF_SAMPLES_PER_CLASS,embedding_index)
        for i in range(class_count):
                mean_distance_matrix[embedding_index,i,:],var_distance_matrix[embedding_index,i,:] =  compute_distance_between_class_i_with_all_others(data,classes[i])
                    
    return mean_distance_matrix,var_distance_matrix

def load_n_samples_from_EMNIST(NUMBER_OF_SAMPLES_PER_CLASS,embedding_index):
    samples = {}
    for class_label in MNIST_Dataset_Referencer.INDEXER.classes:
        indicies = EMBEDDED_MNIST_DATASET[embedding_index].class_indicies[class_label][:NUMBER_OF_SAMPLES_PER_CLASS]
        instances_of_class = [EMBEDDED_MNIST_DATASET[embedding_index][index][0] for index in indicies]

        samples[class_label] = instances_of_class
    return  samples

def standardize_array(array):
    avg = np.mean(array)
    std = np.std(array)
    return (array-avg)/std
#the input looks like this [d1[mean,var],d2[mean,var]]

def visualize_array(array):
    distance_index = 0
    fig, axs = plt.subplots(4, 2, figsize=(10, 15))
    fig.suptitle('Interclass Distances')
    for mean,var in array:

        # normalized_mean = (mean - np.min(mean)) / (np.max(mean) - np.min(mean))
        # normalized_var = (var - np.min(var)) / (np.max(var) - np.min(var))   
        normalized_mean = standardize_array(mean)
        normalized_var = standardize_array(var)
        im1 = axs[distance_index,0].imshow(normalized_mean, cmap='viridis')
        axs[distance_index,0].set_title(f"{order_of_embeddings[distance_index]}")
        im2 = axs[distance_index,1].imshow(normalized_var, cmap='viridis')
        axs[distance_index,1].set_title('Variance')
        fig.colorbar(im1, ax=axs[distance_index,0], fraction=0.046, pad=0.04)
        fig.colorbar(im2, ax=axs[distance_index,1], fraction=0.046, pad=0.04)
        #axs[1].axis('off
        distance_index+=1
    
    save_path = os.path.join(GlobalConfig.RESULTS_FOLDER_PATH,GlobalConfig.MNIST_INTER_CLASS_DIST,"interclass_mahalanobis_distance.png")
    plt.savefig(save_path)
    plt.close(fig)