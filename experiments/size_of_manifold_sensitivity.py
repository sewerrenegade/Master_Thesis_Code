import sys
sys.path.append('/home/milad/Desktop/Master_Thesis/code/Master_Thesis_Code')

from results.report_generators.report_generator import create_pdf_from_dataset_reports, produce_element_from_df, produce_pivot_table_from_dict_lists
from results.report_generators.dataset_report_generator import DatasetReportElements
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from results.metrics_descriptor import MetricsDescriptor
from datasets.embedded_datasets.dataset.embedding_base import EmbeddingBaseDataset
from distance_functions.distance_function_metrics.distance_matrix_metrics import DistanceMatrixMetricCalculator
from distance_functions.functions.basic_distance_functions import EuclideanDistance
from datasets.embedded_datasets.generators.embedding_descriptor import EmbeddingDescriptor
from configs.global_config import GlobalConfig
from datasets.image_augmentor import AugmentationSettings
import umap
import phate
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap,TSNE
import pandas as pd
from datasets.SCEMILA import *
from datasets.dataset_factory import BASE_MODULES as DATA_SET_MODULES

DEFAULT_TRANSFROM_DICT = {
    "PHATE": (phate.PHATE,{
    'n_components': 8,
    'knn': 10,
    'decay': 40,
    't': 'auto'}),
    
    "TSNE": (TSNE,{
    'n_components': 8,
    'method':'exact'}),
    
    "Isomap": (Isomap,{
    'n_components': 8}),
    
    "UMAP": (umap.UMAP,{
    'n_components': 8}),
    
    "PCA": (PCA,{
    'n_components': 8})
}
MANIFOLD_CALC_SIZE = [30,60,120,240,480]
PER_CLASS_NB_SAMPLES_FOR_METRIC_CALC = 30
SWEEP_PORJECTION_DIM = GlobalConfig.DOWNPROJECTION_TEST_DIMENSIONS
METRIC = DistanceMatrixMetricCalculator
DATASET_NAMES_AND_SETTINGS = {("SCEMILA/image_data","normal"):{"training_mode":True,"balance_dataset_classes": 100,"gpu":False,"augmentation_settings":AugmentationSettings(),"numpy":True},
                              ("SCEMILA/image_data","dino"):{"training_mode":True,"encode_with_dino_bloom":True,"balance_dataset_classes": 100,"gpu":False,"augmentation_settings":AugmentationSettings(),"numpy":True},
                              ("Acevedo","normal"):{"training_mode":True,"balance_dataset_classes": 100,"gpu":False,"augmentation_settings":AugmentationSettings(),"numpy":True},
                              ("Acevedo","dino"):{"training_mode":True,"encode_with_dino_bloom":True,"balance_dataset_classes": 100,"gpu":False,"augmentation_settings":AugmentationSettings(),"numpy":True},
                              ("FashionMNIST","normal"):{"training_mode":True,"balance_dataset_classes": 100,"gpu":False,"augmentation_settings":AugmentationSettings(),"numpy":True},
                              ("CIFAR10","normal"):{"training_mode":True,"balance_dataset_classes": 100,"gpu":False,"augmentation_settings":AugmentationSettings(),"numpy":True},
                              ("MNIST","normal"):{"training_mode":True,"balance_dataset_classes": 100,"gpu":False,"augmentation_settings":AugmentationSettings(),"numpy":True},
                              }

descriptors = []

def test_manifold_size_sensitivity():
    

    experiment_metrics_list = []
    dataset_reports = {dataset_name[0]:DatasetReportElements() for dataset_name in list(DATASET_NAMES_AND_SETTINGS.keys())}
    

    for dataset_name, db_settings in DATASET_NAMES_AND_SETTINGS.items():
        for nb_points_on_man in MANIFOLD_CALC_SIZE:
            db_settings["balance_dataset_classes"] = nb_points_on_man
            db_settings["augmentation_settings"] = AugmentationSettings.create_settings_with_name("none")
            dataset_class = DATA_SET_MODULES.get(dataset_name[0])
            dataset = dataset_class(**db_settings)
            dataset_report = dataset_reports[dataset_name[0]]
            dataset_report.add_variant(dataset)

            euclidean_baseline_metric_desc = MetricsDescriptor(metric_calculator=METRIC, dataset=dataset, distance_function=EuclideanDistance(), per_class_samples=PER_CLASS_NB_SAMPLES_FOR_METRIC_CALC)
            euclidean_baseline_metrics = euclidean_baseline_metric_desc.calculate_metric()
            experiment_metrics_list.append({"dataset":dataset_name[0],"dinobloom":dataset_name[1],"manifold_calculation_size": nb_points_on_man,"distance":"euclidean_distance","metric":euclidean_baseline_metrics})

            for transform_name, transform in DEFAULT_TRANSFROM_DICT.items():
                trans_func, trans_settings = transform
                descriptor = EmbeddingDescriptor(f"{dataset_name[0]}_{transform_name}_8", dataset, transform_name, trans_func, trans_settings)
                embd_ds= descriptor.make_sure_embedding_exists()
                metric_desc = MetricsDescriptor(metric_calculator=METRIC, dataset=embd_ds, distance_function=EuclideanDistance(), per_class_samples=PER_CLASS_NB_SAMPLES_FOR_METRIC_CALC)
                it_metrics = metric_desc.calculate_metric()
                experiment_metrics_list.append({"dataset":dataset_name[0],"dinobloom":dataset_name[1],"manifold_calculation_size": nb_points_on_man,"distance":transform_name,"metric":it_metrics})

            print(euclidean_baseline_metric_desc.to_dict())
    return experiment_metrics_list,dataset_reports



                        
def produce_experiment_elements():
    experiment_metrics_list,dataset_reports = test_manifold_size_sensitivity()
    for dataset_report_name,dataset_report in dataset_reports.items():
        dataset_name = dataset_report.name
        relavent_metrics = [metric for metric in experiment_metrics_list if metric["dataset"] == dataset_name]
        relavent_scalar_metrics = METRIC.get_all_scalar_metrics(relavent_metrics)
        pd_pivoted_table = produce_pivot_table_from_dict_lists(relavent_scalar_metrics,["dinobloom","distance"],["manifold_calculation_size"],["loocv_knn_acc","intra_to_inter_class_distance_overall_ratio"])
        table_element = produce_element_from_df(pd_pivoted_table)
        dataset_report.result_elements.append(table_element)
    create_pdf_from_dataset_reports(dataset_reports)
    
        

        
if __name__ == '__main__':
    produce_experiment_elements()
    pass