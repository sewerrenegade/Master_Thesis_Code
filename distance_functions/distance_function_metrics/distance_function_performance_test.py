import numpy as np
import sys
from results.metrics_descriptor import MetricsDescriptor
from results.results_manager import ResultsManager

from datasets.embedded_datasets.dataset.embedding_base import EmbeddingBaseDataset as EmbeddingDatabase
import matplotlib.pyplot as plt
import os
from configs.global_config import GlobalConfig
from distance_functions.distance_function_metrics.distance_matrix_metrics import DistanceMatrixMetricCalculator

from datasets.dataset_factory import BASE_MODULES as DATA_SET_MODULES



def calculate_metric_from_descriptor(descriptor:MetricsDescriptor):
    distance_mat_metrics_calc = DistanceMatrixMetricCalculator(descriptor)
    per_class_samples = descriptor.per_class_samples
    distance_matrix, labels_shuffled, per_class_indicies = distance_mat_metrics_calc.calculate_n_sample_balanced_distance_matrix()
    metrics = distance_mat_metrics_calc.calculate_distance_matrix_metrics(distance_matrix, labels_shuffled)
    ResultsManager.get_manager().save_results(descriptor=descriptor,results={"metrics_dict":metrics})
    return metrics
        
def calculate_origin_dataset_metrics(descriptor: MetricsDescriptor):
    if isinstance(descriptor.dataset,EmbeddingDatabase): 
        emb_descriptor = descriptor.dataset.embedding_descriptor
        og_db = DATA_SET_MODULES.get(emb_descriptor.dataset_name,None)
        flatten = True
        if descriptor.distance_function.name == "Cubical Complex Distance":
            flatten = False
        database = og_db(training_mode = True,gpu= False,numpy = True,flatten = flatten,balance_dataset_classes = emb_descriptor.dataset_sampling) 
        new_descriptor = MetricsDescriptor(descriptor.metrics,dataset=database,distance_function=descriptor.distance_function,per_class_samples=descriptor.per_class_samples)
    else:
        raise TypeError("input needs to be an embedding database or the original dataset itself")
    metrics = calculate_metric_from_descriptor(new_descriptor)
    return metrics    
        

def calculate_interclass_distances(database ,per_class_nb_samples,distance_fn):
    classes  = database.classes
    class_count = len(classes)
    mean_distance_matrix = np.zeros((class_count,class_count))
    var_distance_matrix = np.zeros((class_count,class_count))
    data = database.get_random_instances_from_all_classes(per_class_nb_samples)
    for i in range(class_count):
            for j in range(i, class_count):
                if i != j:
                    mean_distance_matrix[i,j],var_distance_matrix[i,j] =  compute_distance_between_classes(distance_fn,data[classes[i]],data[classes[j]])
                    mean_distance_matrix[j,i],var_distance_matrix[j,i] = mean_distance_matrix[i,j],var_distance_matrix[i,j]
                else:
                    mean_distance_matrix[i,i],var_distance_matrix[i,i]= compute_distance_between_classes(distance_fn,data[classes[i]])
    return mean_distance_matrix,var_distance_matrix,data


def visualize_embedding_performance_wrt_dimension(methode_name,metrics,down_dim,data_origin):
    num_metrics = len(metrics[0])
    fig, axs = plt.subplots(num_metrics, 1, figsize=(10, 5 * num_metrics))

    if num_metrics == 1:
        axs = [axs]

    for i, key in enumerate(metrics[0].keys()):
        y_values = [d[key] for d in metrics]
        axs[i].plot(down_dim, y_values, marker='o', linestyle='-', label=key)
        axs[i].set_title(key, fontsize=16)
        axs[i].set_xlabel('Dimension')
        axs[i].set_ylabel(f"{key} Metric")
        axs[i].legend()

    fig.suptitle(f"Metrics Plots for {methode_name}", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    base_path = os.path.join(GlobalConfig.RESULTS_DATA_FOLDER_PATH,f"{data_origin}_interinstance_distances/","dimensionality_results")
    os.makedirs(base_path,exist_ok=True)
    path = os.path.join(base_path,f"{methode_name}_metrics_plots.png")
    plt.savefig(path)
    plt.close(fig)
