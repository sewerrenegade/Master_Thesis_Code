import sys
sys.path.append("/home/milad/Desktop/Master_Thesis/code/Master_Thesis_Code")
from results.report_generators.dataset_report_generator import DatasetReportElements
from results.report_generators.report_generator import create_pdf_from_dataset_reports, produce_element_from_df, produce_pivot_table_from_dict_lists


from results.metrics_descriptor import MetricsDescriptor
from distance_functions.distance_function_metrics.distance_matrix_metrics import (
    DistanceMatrixMetricCalculator,
)
from distance_functions.functions.basic_distance_functions import EuclideanDistance
from distance_functions.functions.cubical_complex_distance import (
    CubicalComplexImageDistanceFunction,
)
from configs.global_config import GlobalConfig
from datasets.image_augmentor import AugmentationSettings

# from datasets.SCEMILA.base_image_SCEMILA import SCEMILAimage_base,SCEMILA_fnl34_feature_base,SCEMILA_DinoBloom_feature_base
from datasets.SCEMILA import *
from datasets.dataset_factory import BASE_MODULES as DATA_SET_MODULES

AUGMENTATIONS_OF_INTEREST = ['all','none']
SWEEP_PORJECTION_DIM = GlobalConfig.DOWNPROJECTION_TEST_DIMENSIONS

DATASET_NAMES_AND_SETTINGS = {
    ("CIFAR10"): {
        "training_mode": True,
        "balance_dataset_classes": 100,
        "gpu": False,
        "augmentation_settings": AugmentationSettings(),
        "numpy": True,

    },
    ("Acevedo"): {
        "training_mode": True,
        "balance_dataset_classes": 100,
        "gpu": False,
        "augmentation_settings": AugmentationSettings(),
        "numpy": True,
    },
}
METRIC = DistanceMatrixMetricCalculator

descriptors = []


def perform_test():
    per_class_samples_for_metric_calc = 10
    experiment_metrics_list = []
    dataset_reports = {dataset_name:DatasetReportElements() for dataset_name in list(DATASET_NAMES_AND_SETTINGS.keys())}

    for dataset_name, db_settings in DATASET_NAMES_AND_SETTINGS.items():
        for augmentation in AUGMENTATIONS_OF_INTEREST:
            db_settings["augmentation_settings"] = AugmentationSettings.create_settings_with_name(augmentation)
            dataset_class = DATA_SET_MODULES.get(dataset_name)
            assert dataset_class is not None
            dataset = dataset_class(**db_settings)
            dataset_report = dataset_reports[dataset_name]
            dataset_report.add_variant(dataset)


            # Euclidean Baseline metrics
            baseline_metric_desc = MetricsDescriptor(
                metric_calculator=METRIC,
                dataset=dataset,
                distance_function=EuclideanDistance(),
                per_class_samples=per_class_samples_for_metric_calc,
            )
            euclidean_baseline_metrics = baseline_metric_desc.calculate_metric()
            experiment_metrics_list.append({"dataset":dataset_name,"augmentation":augmentation,"distance":"euclidean_distance","metric":euclidean_baseline_metrics})

            # Normal RGB Cubical Complex metrics
            cub_complex_metrics_desc = MetricsDescriptor(
                metric_calculator=METRIC,
                dataset=dataset,
                distance_function=CubicalComplexImageDistanceFunction(grayscale_input= False),
                per_class_samples=int(per_class_samples_for_metric_calc),
            )
            cub_complex_metrics = cub_complex_metrics_desc.calculate_metric()
            experiment_metrics_list.append({"dataset":dataset_name,"augmentation":augmentation,"distance":"normal_cub_complex","metric":cub_complex_metrics})
            
            # Merged Channel Cubical Complex metrics
            cub_complex_join_metrics_desc = MetricsDescriptor(
                metric_calculator=METRIC,
                dataset=dataset,
                distance_function=CubicalComplexImageDistanceFunction(join_channels=True,grayscale_input= False),
                per_class_samples=int(per_class_samples_for_metric_calc),
            )
            joint_cub_complex_metrics = cub_complex_join_metrics_desc.calculate_metric()
            experiment_metrics_list.append({"dataset":dataset_name,"augmentation":augmentation,"distance":"merged_pd_channels_cub_complex","metric":joint_cub_complex_metrics})
            

            grayscale_cub_complex_metrics_description = MetricsDescriptor(
                metric_calculator=METRIC,
                dataset=dataset,
                distance_function=CubicalComplexImageDistanceFunction(grayscale_input=True),
                per_class_samples=int(per_class_samples_for_metric_calc),
            )
            grayscale_cubical_complex_metrics = grayscale_cub_complex_metrics_description.calculate_metric()
            experiment_metrics_list.append({"dataset":dataset_name,"augmentation":augmentation,"distance":"grayscale_cub_complex","metric":grayscale_cubical_complex_metrics})
    return experiment_metrics_list, dataset_reports


def produce_experiment_report():
    experiment_metrics_list,dataset_reports = perform_test()
    for dataset_report_name,dataset_report in dataset_reports.items():
        dataset_name = dataset_report.name
        relavent_metrics = [metric for metric in experiment_metrics_list if metric["dataset"] == dataset_name]
        relavent_scalar_metrics = METRIC.get_all_scalar_metrics(relavent_metrics)
        pd_pivoted_table = produce_pivot_table_from_dict_lists(relavent_scalar_metrics,["augmentation","distance"],[],["loocv_knn_acc","intra_to_inter_class_distance_overall_ratio"])
        table_element = produce_element_from_df(pd_pivoted_table)
        dataset_report.result_elements.append(table_element)
    create_pdf_from_dataset_reports(dataset_reports)
    
    
if __name__ == "__main__":
    elements = produce_experiment_report()
    
