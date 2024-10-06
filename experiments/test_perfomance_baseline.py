from results.report_generators.report_generator import create_pdf_from_dataset_reports, produce_element_from_df, produce_pivot_table_from_dict_lists
from results.report_generators.dataset_report_generator import DatasetReportElements
from results.metrics_descriptor import MetricsDescriptor
from datasets.embedded_datasets.dataset.embedding_base import EmbeddingBaseDataset
from distance_functions.distance_function_metrics.distance_matrix_metrics import DistanceMatrixMetricCalculator
from distance_functions.functions.basic_distance_functions import EuclideanDistance, L1Distance
from distance_functions.functions.cubical_complex_distance import CubicalComplexImageDistanceFunction
from datasets.embedded_datasets.generators.embedding_descriptor import EmbeddingDescriptor
from configs.global_config import GlobalConfig
from datasets.image_augmentor import AugmentationSettings
import numpy as np
import umap
import phate
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, TSNE
import copy
from datasets.dataset_factory import BASE_MODULES as DATA_SET_MODULES

DEFAULT_TRANSFROM_DICT = {
    "PHATE": (phate.PHATE, {'n_components': 8, 'knn': 10, 'decay': 40, 't': 'auto'}),
    "TSNE": (TSNE, {'n_components': 8, 'method': 'exact'}),
    "Isomap": (Isomap, {'n_components': 8}),
    "UMAP": (umap.UMAP, {'n_components': 8}),
    "PCA": (PCA, {'n_components': 8})
}
METRIC = DistanceMatrixMetricCalculator
SWEEP_PORJECTION_DIM = GlobalConfig.DOWNPROJECTION_TEST_DIMENSIONS

DATASET_NAMES_AND_SETTINGS = {
    ("FashionMNIST"): {"training_mode": True, "balance_dataset_classes": 10, "gpu": False, "augmentation_settings": AugmentationSettings(), "numpy": True},
    ("SCEMILA/image_data"): {"training_mode": True, "balance_dataset_classes": 10, "gpu": False, "augmentation_settings": AugmentationSettings(), "numpy": True},
    ("Acevedo"): {"training_mode": True, "balance_dataset_classes": 10, "gpu": False, "augmentation_settings": AugmentationSettings(),"numpy": True},
    
    ("CIFAR10"): {"training_mode": True, "balance_dataset_classes": 10, "gpu": False, "augmentation_settings": AugmentationSettings(), "numpy": True},
    ("MNIST"): {"training_mode": True, "balance_dataset_classes": 10, "gpu": False, "augmentation_settings": AugmentationSettings(), "numpy": True},
}
AUGMENTATIONS_OF_INTEREST = ['all', 'none']
descriptors = []
    

def establish_baseline():
    per_class_samples_for_metric_calc = 10

    experiment_metrics_list = []
    dataset_reports = {dataset_name:DatasetReportElements() for dataset_name in list(DATASET_NAMES_AND_SETTINGS.keys())}
    

    for dataset_name, db_settings in DATASET_NAMES_AND_SETTINGS.items():
        for augmentation in AUGMENTATIONS_OF_INTEREST:
            db_settings["augmentation_settings"] = AugmentationSettings.create_settings_with_name(augmentation)
            dataset_class = DATA_SET_MODULES.get(dataset_name)
            dataset = dataset_class(**db_settings)
            dataset_report = dataset_reports[dataset_name]
            dataset_report.add_variant(dataset)

            euclidean_baseline_metric_desc = MetricsDescriptor(metric_calculator=METRIC, dataset=dataset, distance_function=EuclideanDistance(), per_class_samples=per_class_samples_for_metric_calc)
            euclidean_baseline_metrics = euclidean_baseline_metric_desc.calculate_metric()
            experiment_metrics_list.append({"dataset":dataset_name,"augmentation":augmentation,"distance":"euclidean_distance","metric":euclidean_baseline_metrics})

            L1_baseline_metric_desc = MetricsDescriptor(metric_calculator=METRIC, dataset=dataset, distance_function=L1Distance(), per_class_samples=per_class_samples_for_metric_calc)
            L1_baseline_metrics = L1_baseline_metric_desc.calculate_metric()
            experiment_metrics_list.append({"dataset":dataset_name,"augmentation":augmentation,"distance":"L1_distance","metric":L1_baseline_metrics})

            cubical_complex_baseline_db_settings = copy.deepcopy(db_settings)
            if dataset_name == "SCEMILA/image_data" or dataset_name == "Acevedo":
                cubical_complex_baseline_db_settings["resize"] = True    
            dataset_cub = dataset_class(**cubical_complex_baseline_db_settings)
            cub_complex_metrics_desc = MetricsDescriptor(metric_calculator=METRIC, dataset=dataset_cub, distance_function=CubicalComplexImageDistanceFunction(), per_class_samples=int(per_class_samples_for_metric_calc / 2))
            cub_complex_metrics = cub_complex_metrics_desc.calculate_metric()
            experiment_metrics_list.append({"dataset":dataset_name,"augmentation":augmentation,"distance":"cubical_complex_distance","metric":cub_complex_metrics})

            for transform_name, transform in DEFAULT_TRANSFROM_DICT.items():
                trans_func, trans_settings = transform
                descriptor = EmbeddingDescriptor(f"{dataset_name[0]}_{transform_name}_8", dataset, transform_name, trans_func, trans_settings)
                embd_ds= descriptor.make_sure_embedding_exists()
                metric_desc = MetricsDescriptor(metric_calculator=METRIC, dataset=embd_ds, distance_function=EuclideanDistance(), per_class_samples=per_class_samples_for_metric_calc)
                it_metrics = metric_desc.calculate_metric()
                experiment_metrics_list.append({"dataset":dataset_name,"augmentation":augmentation,"distance":transform_name,"metric":it_metrics})

            print(cub_complex_metrics_desc.to_dict())
    return experiment_metrics_list,dataset_reports

def produce_experiment_elements():
    elements = []
    experiment_metrics_list,dataset_reports = establish_baseline()
    for dataset_report_name,dataset_report in dataset_reports.items():
        dataset_name = dataset_report.name
        relavent_metrics = [metric for metric in experiment_metrics_list if metric["dataset"] == dataset_name]
        relavent_scalar_metrics = METRIC.get_all_scalar_metrics(relavent_metrics)
        pd_pivoted_table = produce_pivot_table_from_dict_lists(relavent_scalar_metrics,["augmentation","distance"],[],["intra_to_inter_class_distance_overall_ratio","knn_acc","knn_acc_std","loocv_knn_acc","loocv_knn_acc_std","silhouette_score"])
        table_element = produce_element_from_df(pd_pivoted_table)
        dataset_report.result_elements.append(table_element)
    create_pdf_from_dataset_reports(dataset_reports)
    

        
if __name__ == '__main__':
    elements = produce_experiment_elements()
    pass