import sys
sys.path.append('/home/milad/Desktop/Master_Thesis/code/Master_Thesis_Code')

from datasets.MNIST.MNIST_base import baseDataset
from datasets.embedded_data.generators.generate_embeddings import generate_embedding_from_descriptor, EmbeddingDescriptor

import umap
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap,TSNE





DOWN_PROJECTION_DIM = 32
MNIST_BASE_DS = baseDataset(True,dataset_size= 1000)
descriptors = [
    EmbeddingDescriptor("TSNE",MNIST_BASE_DS,TSNE(n_components=DOWN_PROJECTION_DIM,method='exact').fit_transform),
    EmbeddingDescriptor("ISOMAP",MNIST_BASE_DS,Isomap(n_components=DOWN_PROJECTION_DIM).fit_transform),
    EmbeddingDescriptor("UMAP",MNIST_BASE_DS,umap.UMAP(n_components=DOWN_PROJECTION_DIM).fit_transform),
    EmbeddingDescriptor("PCA",MNIST_BASE_DS,PCA(n_components=DOWN_PROJECTION_DIM,).fit_transform),
    
]

if __name__ == '__main__':
    for descriptor in descriptors:
        print(f"started generating embeddings for {descriptor.name}")
        print(f"finished generating embeddings for {descriptor.name} saved in path {generate_embedding_from_descriptor(descriptor)}")

