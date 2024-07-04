import sys
sys.path.append('/home/milad/Desktop/Master_Thesis/code/Master_Thesis_Code')

from datasets.MNIST.MNIST_base import MNIST_base
from datasets.FMNIST.FMNIST_base import FMNIST_base
from datasets.CIFAR10.CIFAR10_base import CIFAR10_base
from datasets.embedded_data.generators.generate_embeddings import generate_embedding_from_descriptor, EmbeddingDescriptor
from configs.global_config import GlobalConfig
import umap
import phate
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap,TSNE
from datasets.SCEMILA.base_SCEMILA import SCEMILAimage_base,SCEMILA_fnl34_feature_base,SCEMILA_DinoBloom_feature_base


DOWN_PROJECTION_DIM = 32
SWEEP_PORJECTION_DIM = GlobalConfig.DOWNPROJECTION_TEST_DIMENSIONS
FMNIST_BASE_DS = FMNIST_base(True,dataset_size= 1000,gpu=False)
MNIST_BASE_DS = MNIST_base(True,dataset_size= 1000,gpu=False)
CIFAR10_BASE_DS = CIFAR10_base(True,dataset_size= 1000,gpu=False)
SCEMILA_IMAGE_BASE_DS = SCEMILAimage_base(transforms_list=None,gpu=False, numpy= True,number_of_per_class_instances=100)
#SCEMILA_FEATURE_BASE_DS = SCEMILA_fnl34_feature_base(transforms_list=None,gpu=False)
SCEMILA_DINO_BLOOM_BASE_DS = SCEMILA_DinoBloom_feature_base(transforms_list=None,gpu=False, numpy= True,number_of_per_class_instances=100)
#REONSTRUCTION_BASE_DS = embedded_baseDataset(dataset_size= 1000,root_dir="data/MNIST/embeddings/reconstruction_autoencoder_v2/")

descriptors = []
#dataset = SCEMILA_DINO_BLOOM_BASE_DS
for dataset in [MNIST_BASE_DS,FMNIST_BASE_DS,CIFAR10_BASE_DS,SCEMILA_IMAGE_BASE_DS]:
    for dim in SWEEP_PORJECTION_DIM:
        descriptors.append(EmbeddingDescriptor(f"PHATE_{dim}",dataset,phate.PHATE(n_components=dim,knn=10, decay=40, t='auto').fit_transform))#n_components=dim, knn=10, decay=40, t='auto'
        descriptors.append(EmbeddingDescriptor(f"TSNE_{dim}",dataset,TSNE(n_components=dim,method='exact').fit_transform))
        descriptors.append(EmbeddingDescriptor(f"ISOMAP_{dim}",dataset,Isomap(n_components=dim).fit_transform))
        descriptors.append(EmbeddingDescriptor(f"UMAP_{dim}",dataset,umap.UMAP(n_components=dim).fit_transform))
        descriptors.append(EmbeddingDescriptor(f"PCA_{dim}",dataset,PCA(n_components=dim).fit_transform))

if __name__ == '__main__':
    for descriptor in descriptors:
        print(f"started generating embeddings for {descriptor.name}")
        print(f"finished generating embeddings for {descriptor.name} saved in path {generate_embedding_from_descriptor(descriptor)}")

