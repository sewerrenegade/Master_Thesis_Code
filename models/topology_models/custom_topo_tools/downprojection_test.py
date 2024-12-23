import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import phate
from sklearn.datasets import fetch_openml,make_swiss_roll
import sys
sys.path.append('C:/Users\MiladBassil/Desktop/Master_Thesis/code\Master_Thesis_Code')
from datasets.SCEMILA.SEMILA_indexer import SCEMILA_Indexer
from datasets.SCEMILA.base_image_SCEMILA import SCEMILA_MIL_base
from models.topology_models.custom_topo_tools.downprojection_tool import ConnectivityDP

DATASETS = ["MNIST","SWISS_ROLL","DinoBloom"]
DATASET = DATASETS[2]
NB_SAMPLES = 200

# Load MNIST dataset
def perform_mnist_test():
    if DATASET == DATASETS[0]:
        print("Loading MNIST dataset...")
        mnist = fetch_openml("mnist_784", version=1)
        X, y = mnist.data, mnist.target.astype(int)
        np.random.seed(42)
        indices = np.random.choice(X.shape[0], NB_SAMPLES, replace=False)
        X, y = X.to_numpy(), y.to_numpy()
        X, y = X[indices], y[indices]
        X = X / 255.0
    elif DATASET == DATASETS[1]:
        print("Generating Swiss Roll dataset...")
        X, y = make_swiss_roll(n_samples=NB_SAMPLES, random_state=42)
    elif DATASET == DATASETS[2]:
        print("Loading AML dataset...")
        data = np.load(r"C:\Users\MiladBassil\Desktop\dinbloomS_labeled1.npz")
        X = data["embedding"]
        y_string = data["labels"]
        indices = np.arange(1500)  # Create an array of indices
        np.random.shuffle(indices)  # Shuffle the indices
        x_shuffled = X[indices]  # Shuffle x using the shuffled indices
        y_shuffled = y_string[indices]  # Shuffle y using the same indices

        # Extract the first 200 data points and their labels
        X = x_shuffled[:NB_SAMPLES]
        y_string = y_shuffled[:NB_SAMPLES]
        y = np.zeros(y_string.shape)
        indexer = SCEMILA_Indexer()
        
        for index in range(NB_SAMPLES):
            y[index] = indexer.convert_from_int_to_label_instance_level(y_string[index])
            
    connectivity_operator = ConnectivityDP(n_iter=300, learning_rate=1,normalize_input= True,loss_calculation_timeout=10,optimizer_name="sgd",importance_weighting=True,augmentation_scheme={"name":"uniform","p":0.01},dev_settings={"labels":y,"create_vid":None,"+++moor_method":None})
    connectivity_embedding = connectivity_operator.fit_transform(X)
    # Dimensionality reduction methods
    print("Calculating PCA...")
    pca_embedding = PCA(n_components=2).fit_transform(X)

    print("Calculating t-SNE...")
    tsne_embedding = TSNE(n_components=2, init="random", random_state=42).fit_transform(X)

    print("Calculating UMAP...")
    umap_embedding = umap.UMAP(n_components=2, random_state=42).fit_transform(X)

    print("Calculating PHATE...")
    phate_operator = phate.PHATE(n_components=2,random_state=42)
    phate_embedding = phate_operator.fit_transform(X)


    # Plot the results
    def plot_embeddings(embeddings, titles, labels, figsize=(12, 8)):
        """Helper function to visualize embeddings."""
        plt.figure(figsize=figsize)
        for i, (embedding, title) in enumerate(zip(embeddings, titles)):
            plt.subplot(2, 3, i + 1)
            plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap="tab10", s=5)
            plt.colorbar()
            plt.title(title)
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
        plt.tight_layout()
        plt.show()
        #plt.close()


    # Visualize
    plot_embeddings(
        embeddings=[pca_embedding, tsne_embedding, umap_embedding, phate_embedding,connectivity_embedding],
        titles=["PCA", "t-SNE", "UMAP", "PHATE","Connectivity"],
        labels=y,
    )
if __name__ == "__main__":
    perform_mnist_test()