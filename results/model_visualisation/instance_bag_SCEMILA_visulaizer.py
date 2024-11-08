import numpy as np
import matplotlib.pyplot as plt
import umap  # You can use PHATE instead if you prefer
import phate
from sklearn.preprocessing import LabelEncoder
from matplotlib.colors import ListedColormap
import wandb
from torch import flatten

from datasets.SCEMILA.SCEMILA_lightning_wrapper import SCEMILA

LABELS_TO_REMOVE_FROM_VIZ = ["other","ambiguous",]
def plot_dinobloob_2D_embeddings():
    import numpy as np
    dataset = SCEMILA(encode_with_dino_bloom = True,gpu= False,num_workers=0)
    indexer = dataset.indexer
    instance_level_annotations_by_class,instance_level_class_count,instance_classes = indexer.get_instance_level_annotations(as_is = True)

    single_cell_embeddings = []  
    single_cell_labels = []     
    for cell_label,cells in instance_level_annotations_by_class.items():
        for cell_path in cells:
            x  = dataset.test_dataset.get_single_tiff_image_using_path(cell_path,cell_label)
            single_cell_embeddings.append(flatten(x[0]).cpu().numpy())
            single_cell_labels.append(cell_label)

    single_cell_embeddings, single_cell_labels = zip(*[(d, l) for d, l in zip(single_cell_embeddings, single_cell_labels) if l not in LABELS_TO_REMOVE_FROM_VIZ])
    single_cell_embeddings, single_cell_labels = list(single_cell_embeddings), list(single_cell_labels)

    single_cell_embeddings = np.array(single_cell_embeddings)
    plot_and_log_2D_embedding(embedding=single_cell_embeddings,labels=single_cell_labels,name = "Single Cell",log_wandb= False)

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
            single_cell_embeddings.append(flatten(model.get_instance_level_encoding(x[0])).cpu().numpy())
            single_cell_labels.append(cell_label)

    single_cell_embeddings, single_cell_labels = zip(*[(d, l) for d, l in zip(single_cell_embeddings, single_cell_labels) if l not in LABELS_TO_REMOVE_FROM_VIZ])
    single_cell_embeddings, single_cell_labels = list(single_cell_embeddings), list(single_cell_labels)
    single_cell_embeddings = np.array(single_cell_embeddings)
    plot_and_log_2D_embedding(embedding=single_cell_embeddings,labels=single_cell_labels,name = "Single Cell")

    patient_embeddings = []
    patient_labels = []
    for patient_label,patients in per_class_test_patients_paths.items():
        for patient_path in patients:
            x  = dataset.test_dataset.get_single_tiff_bag_using_path(patient_path,patient_label)
            patient_embeddings.append(flatten(model.get_bag_level_encoding(x[0])).cpu().numpy())
            patient_labels.append(patient_label)
    patient_embeddings, patient_labels = zip(*[(d, l) for d, l in zip(patient_embeddings, patient_labels) if l not in LABELS_TO_REMOVE_FROM_VIZ])
    patient_embeddings, patient_labels = list(patient_embeddings), list(patient_labels)
    patient_embeddings = np.array(patient_embeddings)
    plot_and_log_2D_embedding(embedding= patient_embeddings,labels= patient_labels,name= "Patient")

custom_colors = [
    '#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', 
    '#984ea3', '#999999', '#e41a1c', '#dede00', '#8dd3c7',
    '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5',
    '#bc80bd', '#ccebc5', '#ffed6f', '#1f78b4', '#33a02c',
    '#ffb3b3', '#b3b3ff', '#ffd700', '#7fc97f', '#beaed4'
]
markers = ['o', '^', 's', 'D', 'P', 'X', '*', 'h', 'v', '<']
def plot_and_log_2D_embedding(embedding,labels,name,log_wandb = True):

    assert len(embedding) == len(labels)
    le = LabelEncoder()
    y = le.fit_transform(labels)  # Convert labels to integers
    label_names = le.classes_  # Get label names for the legend
    reducers = {"UMAP":umap.UMAP(n_components=2),"PHATE": phate.PHATE(n_components=2)}
    for reducer_name,reducer in reducers.items():
        embedding_2D = reducer.fit_transform(embedding)

        # Create a scatter plot with matplotlib
        plt.figure(figsize=(10, 7))

        # Generate color map
        # colors = cm.rainbow(np.linspace(0, 1, len(np.unique(y))))
        #plt.cm.tab20.colors + plt.cm.tab20b.colors[:5]
        cmap = ListedColormap(custom_colors)

        # Plot each label with a specific color
        for i, label in enumerate(np.unique(y)):
            plt.scatter(embedding_2D[y == label, 0], embedding_2D[y == label, 1], marker=markers[i % len(markers)],
                        color=cmap(i), label=label_names[label], alpha=0.7, s=30)

        # Add legend to map colors to labels
        plt.legend(title="Labels", bbox_to_anchor=(1.05, 1), loc='upper left')  
        plt.title(f"{name} Encoder {reducer_name} Embeddings")
        plt.xlabel(f"{reducer_name} Dimension 1")
        plt.ylabel(f"{reducer_name} Dimension 2")
        plt.tight_layout()
        if log_wandb:
            wandb.log({f"{name}_{reducer_name}": wandb.Image(plt)})
        else:
            plt.savefig(f"{name}_{reducer_name}", bbox_inches='tight')
        plt.close()

