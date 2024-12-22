import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import phate
from sklearn.datasets import fetch_openml,make_swiss_roll
import sys
sys.path.append('C:/Users\MiladBassil/Desktop/Master_Thesis/code\Master_Thesis_Code')
from datasets.SCEMILA.base_image_SCEMILA import SCEMILA_MIL_base
from models.topology_models.custom_topo_tools.downprojection_tool import ConnectivityDP

DATASETS = ["MNIST","SWISS_ROLL","DinoBloom"]
DATASET = DATASETS[0]
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
        X, y = make_swiss_roll(n_samples=NB_SAMPLES, random_state=42)
    elif DATASET == DATASETS[2]:
        dataset = SCEMILA_MIL_base(gpu = False,encode_with_dino_bloom=True,training_mode=True)
        indexer = dataset.indexer
        instance_level_annotations_by_class,instance_level_class_count,instance_classes = indexer.get_instance_level_annotations(as_is = True)

        _,per_class_test_patients_paths = indexer.seperate_test_train_data()


        single_cell_embeddings = []  
        single_cell_labels = []     
        for cell_label,cells in instance_level_annotations_by_class.items():
            for cell_path in cells:
                x  = dataset.get_single_tiff_image_using_path(cell_path,cell_label)
                single_cell_embeddings.append(x[0].numpy())
                single_cell_labels.append(cell_label)

        single_cell_embeddings, single_cell_labels = zip(*[(d, l) for d, l in zip(single_cell_embeddings, single_cell_labels) if l not in LABELS_TO_REMOVE_FROM_VIZ])
        single_cell_embeddings, single_cell_labels = list(single_cell_embeddings), list(single_cell_labels)
        single_cell_embeddings = np.array(single_cell_embeddings)    
    connectivity_operator = ConnectivityDP(n_iter=300, learning_rate=5,normalize_input= True,loss_calculation_timeout=10,optimizer_name="adam",importance_weighting=True,augmentation_scheme={"name":"uniform","p":0.01},dev_settings={"labels":y,"create_vid":None,"+++moor_method":None})
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