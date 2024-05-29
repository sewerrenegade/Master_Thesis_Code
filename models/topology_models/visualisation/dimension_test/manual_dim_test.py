import sys
sys.path.append('/home/milad/Desktop/Master_Thesis/code/Master_Thesis_Code')
import matplotlib.pyplot as plt
import os
from configs.global_config import GlobalConfig
from datasets.MNIST.MNIST_base import MNIST_Dataset_Referencer
from datasets.embedded_data.generators.generation_script import SWEEP_PORJECTION_DIM

from models.topology_models.visualisation.distance_performance_test import load_n_samples_from_EMNIST,test_embedding


def save_method_metrics_wrt_downprojection_dim(methode_name,metrics,down_dim):
    num_metrics = len(metrics[0])
    fig, axs = plt.subplots(num_metrics, 1, figsize=(10, 5 * num_metrics))

    if num_metrics == 1:
        axs = [axs]

    for i, key in enumerate(metrics[0].keys()):
        y_values = [d[key] for d in metrics]
        axs[i].plot(down_dim, y_values, marker='o', linestyle='-', label=key)
        axs[i].set_title(key, fontsize=16)
        axs[i].set_xlabel('Dimension')
        axs[i].set_ylabel('Metric')
        axs[i].legend()

    fig.suptitle(f"Metrics Plots for {methode_name}", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    path = os.path.join(GlobalConfig.RESULTS_FOLDER_PATH,GlobalConfig.MNIST_INTER_CLASS_DIST,f"dimensionality_results/{methode_name}metrics_plots.png")
    plt.savefig(path)


def save_method_metrics_wrt_downprojection_dim_onsame_fig(metrics,down_dim,order_of_embeddings):
    num_metrics = len(metrics[0][0])
    fig, axs = plt.subplots(num_metrics, 1, figsize=(10, 5 * num_metrics))
    assert len(metrics) == len(order_of_embeddings)
    if num_metrics == 1:
        axs = [axs]
    for method_index in range(len(metrics)):
        for i, key in enumerate(metrics[0][0].keys()):
            y_values = [d[key] for d in metrics[method_index]]
            axs[i].plot(down_dim, y_values, marker='o', linestyle='-', label=order_of_embeddings[method_index])
            axs[i].set_title(key, fontsize=16)
            axs[i].set_xlabel('Dimension')
            axs[i].set_ylabel(f"{key} Metric")
            axs[i].legend()

    fig.suptitle(f"Metrics Plots for all embedding Methods", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    path = os.path.join(GlobalConfig.RESULTS_FOLDER_PATH,GlobalConfig.MNIST_INTER_CLASS_DIST,f"dimensionality_results/joint_metrics_plots.png")
    plt.savefig(path)


if __name__ == '__main__':
    down_dim = SWEEP_PORJECTION_DIM
    order_of_embeddings = ["ISOMAP","PCA","TSNE","UMAP"]
    joint_metrics= []
    for method in order_of_embeddings:
        metrics = []
        for dim in down_dim:
            name = f"{method}_{dim}"
            print(f"started metrics calculation for {name}")
            classes  = MNIST_Dataset_Referencer.INDEXER.classes
            data  = load_n_samples_from_EMNIST(100,name)
            metrics.append(test_embedding(name,data, MNIST_Dataset_Referencer.INDEXER.classes,notes = 'second test'))
            print(f"finished generating embeddings for {name} saved in path ")
        save_method_metrics_wrt_downprojection_dim(methode_name=method,metrics= metrics,down_dim=down_dim)
        joint_metrics.append(metrics)    
    save_method_metrics_wrt_downprojection_dim_onsame_fig(joint_metrics,down_dim,order_of_embeddings)