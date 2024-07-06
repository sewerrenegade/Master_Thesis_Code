import sys
sys.path.append('/home/milad/Desktop/Master_Thesis/code/Master_Thesis_Code')
import numpy as np
from datasets.MNIST.MNIST_base import MNIST_Dataset_Referencer
from datasets.MNIST.MNIST_base import BaseDataset as MNISTbase
import random
import matplotlib.pyplot as plt
from models.topology_models.tardis.api import calculate_euclidicity


def calculate_class_dim_and_eucl(data,sub_set,max_dim):
    euclidicity,dimesionality = calculate_euclidicity(
        data, sub_set[0],max_dim= max_dim,return_dimensions=True,k=110,n_jobs=12
    )# i have 20 threads
    return euclidicity,dimesionality
    
def load_n_samples_and_m_subsamples_from_MNIST(per_class_n_samples,per_class_subsamples):
    samples = []
    sub_samples = []
    sub_sample_labels = []
    mnist_dataset= MNISTbase(training= True,gpu=False)
    for class_label in MNIST_Dataset_Referencer.INDEXER.classes:
        indicies = MNIST_Dataset_Referencer.INDEXER.get_random_instance_of_class([class_label]*per_class_n_samples,training=True)
        instances_of_class = [mnist_dataset[index][0].reshape(-1).numpy() for index in indicies]
        sub_sample_instances = random.sample(range(per_class_n_samples), per_class_subsamples)
        sub_sample_instances = [instances_of_class[sub_sample] for sub_sample in sub_sample_instances]
        sub_samples.extend(sub_sample_instances)
        sub_sample_labels.extend([class_label]*per_class_subsamples)
        samples.extend(instances_of_class)
    return  np.array(samples), [np.array(sub_samples),sub_sample_labels]

def create_dictionary_from_output(eucl,dim,label):
    assert len(eucl) == len(dim) and len(dim) == len(label)
    eucl_dict= {}
    dim_dict = {}
    for sample in range(len(label)):
        try:
            eucl_dict[label[sample]].append(eucl[sample])
            dim_dict[label[sample]].append(dim[sample])
        except KeyError:
            eucl_dict[label[sample]]=[eucl[sample]]
            dim_dict[label[sample]]=[dim[sample]]
    return eucl_dict,dim_dict

def produce_box_plot(data_dict, title, path):
    data = [values for values in data_dict.values()]
    labels = [key for key in data_dict.keys()]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(data, patch_artist=True)
    ax.set_title(title, fontsize=16)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=12)
    ax.set_ylabel('Values', fontsize=14)
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)


def test():
    samples,subsamples = load_n_samples_and_m_subsamples_from_MNIST(100,10)
    eucl,dim = calculate_class_dim_and_eucl(samples,subsamples,10)
    eucl,dim = create_dictionary_from_output(eucl,dim,subsamples[1])
    produce_box_plot(dim,"TARDIS computed dimensionality","results/MNIST_interinstance_distances/dimensionality_results/TARDIS_100sample_110k_dimensionality_boxplot.png")
    produce_box_plot(eucl,"TARDIS computed euclidicity","results/MNIST_interinstance_distances/dimensionality_results/TARDIS_100sample_110k_euclidicity_boxplot.png")
if __name__ == '__main__':
    test()