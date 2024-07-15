import sys
sys.path.append('/home/milad/Desktop/Master_Thesis/code/Master_Thesis_Code')
from configs.global_config import GlobalConfig
from distance_functions.distance_function_metrics.embedding_performance_test import visualize_embedding_performance_wrt_dimension, calculate_origin_dataset_metrics, calculate_distance_function_metrics_on_dataset,image_cubical_complex_distance_function
from datasets.embedded_datasets.dataset.embedding_base import EmbeddingBaseDataset as EmbeddingDatabase

def perform_manual_dimensionality_test(original_dataset_name,down_dim = GlobalConfig.DOWNPROJECTION_TEST_DIMENSIONS,order_of_embeddings = GlobalConfig.EMBEDDING_METHODS,per_class_samples = 10):

    cubical_complex_metrics = calculate_origin_dataset_metrics(original_dataset_name,distance_function= image_cubical_complex_distance_function,flatten= False)
    base_metrics = calculate_origin_dataset_metrics(original_dataset_name)
    joint_metrics = {"baseline": [base_metrics]*len(down_dim),"cubical_complex":[cubical_complex_metrics]*len(down_dim)}
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
    #visualize_all_embeddings_performance_wrt_dimension(joint_metrics,data_origin=original_dataset_name,down_dim=down_dim,order_of_embeddings=order_of_embeddings)

if __name__ == '__main__':
    og_datasets = ["SCEMILA/image_data","FashionMNIST","CIFAR10","MNIST","SCEMILA/fnl34_feature_data",]
    original_dataset_name = "SCEMILA/image_data"
    for dataset_name in og_datasets:
        perform_manual_dimensionality_test(dataset_name)