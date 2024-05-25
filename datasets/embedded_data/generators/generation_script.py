import sys
sys.path.append('/home/milad/Desktop/Master_Thesis/code/Master_Thesis_Code')

from datasets.MNIST.MNIST_base import baseDataset
from datasets.embedded_data.generators.generate_embeddings import generate_embedding_from_descriptor, EmbeddingDescriptor

import umap
import phate
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap,TSNE


DOWN_PROJECTION_DIM = 32
SWEEP_PORJECTION_DIM = [1,2,3,8,16,32,64]
MNIST_BASE_DS = baseDataset(True,dataset_size= 1000,gpu=False)
descriptors = []
for dim in SWEEP_PORJECTION_DIM:
    descriptors.append(EmbeddingDescriptor(f"PHATE_{dim}",MNIST_BASE_DS,phate.PHATE(n_components=dim).fit_transform))
    descriptors.append(EmbeddingDescriptor(f"TSNE_{dim}",MNIST_BASE_DS,TSNE(n_components=dim,method='exact').fit_transform))
    descriptors.append(EmbeddingDescriptor(f"ISOMAP_{dim}",MNIST_BASE_DS,Isomap(n_components=dim).fit_transform))
    descriptors.append(EmbeddingDescriptor(f"UMAP_{dim}",MNIST_BASE_DS,umap.UMAP(n_components=dim).fit_transform))
    descriptors.append(EmbeddingDescriptor(f"PCA_{dim}",MNIST_BASE_DS,PCA(n_components=dim).fit_transform))

if __name__ == '__main__':
    for descriptor in descriptors:
        print(f"started generating embeddings for {descriptor.name}")
        print(f"finished generating embeddings for {descriptor.name} saved in path {generate_embedding_from_descriptor(descriptor)}")

