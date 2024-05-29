import sys
sys.path.append('/home/milad/Desktop/Master_Thesis/code/Master_Thesis_Code')

from datasets.MNIST.MNIST_base import baseDataset as MNIST_baseDataset
from datasets.embedded_data.dataset.embedding_base import baseDataset as embedded_baseDataset
from datasets.embedded_data.generators.generate_embeddings import generate_embedding_from_descriptor, EmbeddingDescriptor

import umap
import phate
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap,TSNE


DOWN_PROJECTION_DIM = 32
SWEEP_PORJECTION_DIM = [1,2,3,8,16,32,64]
MNIST_BASE_DS = MNIST_baseDataset(True,dataset_size= 1000,gpu=False)
REONSTRUCTION_BASE_DS = embedded_baseDataset(dataset_size= 1000,root_dir="data/MNIST/embeddings/reconstruction_autoencoder_v2/")
descriptors = []
dataset = REONSTRUCTION_BASE_DS
for dim in SWEEP_PORJECTION_DIM:
    descriptors.append(EmbeddingDescriptor(f"rec_v1_PHATE_{dim}",dataset,phate.PHATE(n_components=dim, knn=5, decay=40, n_pca=100).fit_transform))
    descriptors.append(EmbeddingDescriptor(f"rec_v1_TSNE_{dim}",dataset,TSNE(n_components=dim,method='exact').fit_transform))
    descriptors.append(EmbeddingDescriptor(f"rec_v1_ISOMAP_{dim}",dataset,Isomap(n_components=dim).fit_transform))
    descriptors.append(EmbeddingDescriptor(f"rec_v1_UMAP_{dim}",dataset,umap.UMAP(n_components=dim).fit_transform))
    descriptors.append(EmbeddingDescriptor(f"rec_v1_PCA_{dim}",dataset,PCA(n_components=dim).fit_transform))

if __name__ == '__main__':
    for descriptor in descriptors:
        print(f"started generating embeddings for {descriptor.name}")
        print(f"finished generating embeddings for {descriptor.name} saved in path {generate_embedding_from_descriptor(descriptor)}")

