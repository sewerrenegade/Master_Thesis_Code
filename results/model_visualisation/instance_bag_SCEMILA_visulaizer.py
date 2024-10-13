import numpy as np
import matplotlib.pyplot as plt
import umap  # You can use PHATE instead if you prefer
from sklearn.preprocessing import LabelEncoder
from matplotlib.colors import ListedColormap
import wandb
from torch import flatten



def get_bag_and_instance_level_2D_embeddings(model,dataset):

    import numpy as np
    indexer = dataset.indexer
    instance_level_annotations_by_class,instance_level_class_count,instance_classes = indexer.get_instance_level_annotations(as_is = True)

    _,per_class_test_patients_paths = indexer.seperate_test_train_data()


    single_cell_embeddings = []  
    single_cell_labels = []     
    for cell_label,cells in instance_level_annotations_by_class.items():
        for cell_path in cells:
            x  = dataset.test_dataset.get_single_tiff_image_using_path(cell_path,cell_label)
            single_cell_embeddings.append(flatten(model.get_instance_level_encoding(x[0])).numpy())
            single_cell_labels.append(cell_label)
    single_cell_embeddings = np.array(single_cell_embeddings)
    plot_and_log_2D_embedding(embedding=single_cell_embeddings,labels=single_cell_labels,plot_name = "Single Cell Encoder Embeddings")

    patient_embeddings = []
    patient_labels = []
    for patient_label,patients in per_class_test_patients_paths.items():
        for patient_path in patients:
            x  = dataset.test_dataset.get_single_tiff_bag_using_path(patient_path,patient_label)
            patient_embeddings.append(flatten(model.get_bag_level_encoding(x[0])).numpy())
            patient_labels.append(patient_label)
    patient_embeddings = np.array(patient_embeddings)
    plot_and_log_2D_embedding(embedding= patient_embeddings,labels= patient_labels,plot_name= "Patient Encoder Embeddings")


def plot_and_log_2D_embedding(embedding,labels,plot_name):

    assert len(embedding) == len(labels)
    le = LabelEncoder()
    y = le.fit_transform(labels)  # Convert labels to integers
    label_names = le.classes_  # Get label names for the legend
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding_2D = reducer.fit_transform(embedding)

    # Create a scatter plot with matplotlib
    plt.figure(figsize=(10, 7))

    # Generate color map
    # colors = cm.rainbow(np.linspace(0, 1, len(np.unique(y))))
    cmap = ListedColormap(plt.cm.tab20.colors + plt.cm.tab20b.colors[:5])

    # Plot each label with a specific color
    for i, label in enumerate(np.unique(y)):
        plt.scatter(embedding_2D[y == label, 0], embedding_2D[y == label, 1], 
                    color=cmap(i), label=label_names[label], alpha=0.7, s=40)

    # Add legend to map colors to labels
    plt.legend(title="Labels", bbox_to_anchor=(1.05, 1), loc='upper left')  
    plt.title(plot_name)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.tight_layout()
    wandb.log({plot_name: wandb.Image(plt)})
    plt.close()