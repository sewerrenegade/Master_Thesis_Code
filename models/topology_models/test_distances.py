from datasets.MNIST.MNIST_base import baseDataset,MNIST_Dataset_Referencer
from models.topology_models.distances import PerceptualLoss,CubicalComplexImageEncoder,ReconstructionProjectionModel,RandomProjectionModel
import torch
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import os
from configs.global_config import GlobalConfig

DATASET= baseDataset(training= True)
NUMBER_OF_SAMPLES_PER_CLASS = 50

#perceptual_distance_calculator = PerceptualLoss(device='cpu')
cubical_complex_calculator = CubicalComplexImageEncoder()
random_convolution_distance_calculator = RandomProjectionModel(input_dim=[1,28,28]) #FOR MNIST
rec_auto_enc = ReconstructionProjectionModel(path_to_model= "models/topology_models/reconstruction_distance_parameters/MNIST_Reconstruction_model.pth")

order_of_distances = ["Euclidean_Distance","Reconstruction_Disance","Cubical_Complex"]


def load_n_samples_from_MNIST(n_samples):
    samples = {}
    for class_label in MNIST_Dataset_Referencer.INDEXER.classes:
        indicies = MNIST_Dataset_Referencer.INDEXER.get_random_instance_of_class([class_label]*n_samples,training=True)
        instances_of_class = [DATASET[index][0] for index in indicies]

        samples[class_label] = instances_of_class
    return  samples


def _compute_cubical_distance(x):
    return cubical_complex_calculator(x)[0,1].cpu()


def _compute_reconstruction_distance(x):
    latent = rec_auto_enc.get_latent_code_for_input(x)
    #x_flat = latent.view(x.size(0), -1)
    distances = torch.norm(latent[0] - latent[1], dim=0, p=2).cpu()
    return distances

def _compute_euclidean_distance(x):
    x_flat = x#x.view(x.size(0), -1)
    distances = torch.norm(x_flat[0] - x_flat[1], p=2)
    return distances
#dim 0 of x needs to be bag size

def _compute_perceptual_input_distances(x):
    distance_matrix = perceptual_distance_calculator(x)
    return distance_matrix

def compute_all_distances(x):
    assert len(x)==2
    # out_dic ={}
    # out_dic["rec_distance"] = _compute_reconstruction_distance(x)
    # out_dic["cubical_distance"] = _compute_cubical_distance(x)
    # out_dic["euc_distance"] = _compute_euclidean_distance(x)
    # out_dic["percept_distance"] = _compute_perceptual_input_distances(x)
    out_dic = np.array([_compute_euclidean_distance(x).detach().numpy(),_compute_reconstruction_distance(x).detach().numpy(),_compute_cubical_distance(x).detach().numpy()])#, _compute_perceptual_input_distances(x).detach().numpy()
    return out_dic

def compute_mean_var(class1_samples,class2_samples):
    samples = zip(class1_samples,class2_samples)
    distances= []
    for input_pair in samples:
        distances.append(compute_all_distances(torch.stack(input_pair)))
    mean = np.mean(distances,axis=0)
    var = np.var(distances,axis=0)
    return mean,var
    
def compute_distance_between_classes(class1,class2=None):
    if class2 is None:
        class2=class1.copy()
        random.shuffle(class2)
    return compute_mean_var(class1,class2)


def calculate_class_distance():
    classes = MNIST_Dataset_Referencer.INDEXER.classes
    class_count = len(classes)
    mean_distance_matrix = np.zeros((3,class_count,class_count))
    var_distance_matrix = np.zeros((3,class_count,class_count))
    data = load_n_samples_from_MNIST(NUMBER_OF_SAMPLES_PER_CLASS)
    for i in range(class_count):
        for j in range(i, class_count):
            if i != j:
                mean_distance_matrix[:,i,j],var_distance_matrix[:,i,j] =  compute_distance_between_classes(data[classes[i]],data[classes[j]])
                mean_distance_matrix[:,j,i],var_distance_matrix[:,j,i] = mean_distance_matrix[:,i,j],var_distance_matrix[:,i,j]
            else:
                mean_distance_matrix[:,i,i],var_distance_matrix[:,i,i]= compute_distance_between_classes(data[classes[i]])
    return mean_distance_matrix,var_distance_matrix

def standardize_array(array):
    avg = np.mean(array)
    std = np.std(array)
    return (array-avg)/std
#the input looks like this [d1[mean,var],d2[mean,var]]
def visualize_array(array):
    distance_index = 0
    fig, axs = plt.subplots(3, 2, figsize=(10, 15))
    fig.suptitle('Interclass Distances')
    for mean,var in array:

        # normalized_mean = (mean - np.min(mean)) / (np.max(mean) - np.min(mean))
        # normalized_var = (var - np.min(var)) / (np.max(var) - np.min(var))   
        normalized_mean = standardize_array(mean)
        normalized_var = standardize_array(var)
        im1 = axs[distance_index,0].imshow(normalized_mean, cmap='viridis')
        axs[distance_index,0].set_title(f"{order_of_distances[distance_index]}")
        im2 = axs[distance_index,1].imshow(normalized_var, cmap='viridis')
        axs[distance_index,1].set_title('Variance')
        fig.colorbar(im1, ax=axs[distance_index,0], fraction=0.046, pad=0.04)
        fig.colorbar(im2, ax=axs[distance_index,1], fraction=0.046, pad=0.04)
        #axs[1].axis('off')
        distance_index+=1
    
    save_path = os.path.join(GlobalConfig.RESULTS_FOLDER_PATH,"interclass_distance.png")
    plt.savefig(save_path)