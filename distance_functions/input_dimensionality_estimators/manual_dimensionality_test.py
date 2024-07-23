import sys
sys.path.append('/home/milad/Desktop/Master_Thesis/code/Master_Thesis_Code')
import matplotlib.pyplot as plt
import os
from configs.global_config import GlobalConfig
from datasets.embedded_datasets.dataset.embedding_base import EmbeddingBaseDataset as EmbeddingDatabase
from distance_functions.functions.cubical_complex_distance import CubicalComplexImageDistanceFunction
from distance_functions.distance_function_metrics.distance_function_performance_test import visualize_embedding_performance_wrt_dimension, calculate_origin_dataset_metrics, calculate_distance_function_metrics_on_dataset,image_cubical_complex_distance_function



def visualize_all_embeddings_performance_wrt_dimension(metrics,data_origin,down_dim,order_of_embeddings):
    num_metrics = len(metrics[order_of_embeddings[0]][0])
    fig, axs = plt.subplots(num_metrics, 1, figsize=(15, 5 * num_metrics))
    if not len(metrics) == len(order_of_embeddings):
        temp = ["baseline"]
        temp.extend(order_of_embeddings)
        order_of_embeddings = temp
    assert len(metrics) == len(order_of_embeddings)
    if num_metrics == 1:
        axs = [axs]
    for method_index in range(len(metrics)):
        for i, key in enumerate(metrics[order_of_embeddings[0]][0].keys()):
            y_values = [d[key] for d in metrics[order_of_embeddings[method_index]]]
            axs[i].plot(down_dim, y_values, marker='o', linestyle='-', label=order_of_embeddings[method_index])
            axs[i].set_title(key, fontsize=16)
            axs[i].set_xlabel('Dimension')
            axs[i].set_ylabel(f"{key} Metric")
            axs[i].legend()

    fig.suptitle(f"Metrics Plots for all Embedding Methods originating from {data_origin}", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    path = os.path.join(GlobalConfig.RESULTS_DATA_FOLDER_PATH,f"{data_origin}_interinstance_distances","dimensionality_results/")
    os.makedirs(path,exist_ok= True) 
    path = os.path.join(path,"joint_metrics_plots.png")
    plt.savefig(path)
    plt.close(fig)


def perform_manual_dimensionality_test(original_dataset_name,down_dim = GlobalConfig.DOWNPROJECTION_TEST_DIMENSIONS,order_of_embeddings = GlobalConfig.EMBEDDING_METHODS,per_class_samples = 10):

    joint_metrics = {}
    for method in order_of_embeddings:
        metrics = []
        for dim in down_dim:
            name = f"{method}_{dim}"
            print(f"started metrics calculation for {name}")
            data = EmbeddingDatabase(f"data/{original_dataset_name}/embeddings/{name}/")
            metrics.append(calculate_distance_function_metrics_on_dataset(data,per_class_samples=per_class_samples))
            print(f"finished generating embeddings for {name} saved in path ")
        visualize_embedding_performance_wrt_dimension(methode_name=method,metrics=metrics,down_dim=down_dim,data_origin=original_dataset_name)
        joint_metrics[method] = metrics
    og_dataset_flat = data.get_dataset_origin(flatten= True)
    og_dataset = data.get_dataset_origin(flatten= False)
    cubical_complex_metrics = calculate_distance_function_metrics_on_dataset(og_dataset,per_class_samples=per_class_samples,distance_function= CubicalComplexImageDistanceFunction(),flatten= False)
    base_metrics = calculate_distance_function_metrics_on_dataset(og_dataset_flat,per_class_samples=per_class_samples)
    joint_metrics["baseline"],joint_metrics["cubical_complex"]  = [base_metrics]*len(down_dim),[cubical_complex_metrics]*len(down_dim)
    visualize_all_embeddings_performance_wrt_dimension(joint_metrics,data_origin=original_dataset_name,down_dim=down_dim,order_of_embeddings=order_of_embeddings)

if __name__ == '__main__':
    og_datasets = ["SCEMILA/image_data","FashionMNIST","CIFAR10","MNIST","SCEMILA/fnl34_feature_data",]
    original_dataset_name = "SCEMILA/image_data"
    for dataset_name in og_datasets:
        perform_manual_dimensionality_test(dataset_name)