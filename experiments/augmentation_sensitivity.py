import sys



sys.path.append('/home/milad/Desktop/Master_Thesis/code/Master_Thesis_Code')
    
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from results.metrics_descriptor import MetricsDescriptor
from datasets.embedded_datasets.dataset.embedding_base import EmbeddingBaseDataset
from distance_functions.distance_function_metrics.distance_matrix_metrics import DistanceMatrixMetricCalculator
from distance_functions.functions.basic_distance_functions import EuclideanDistance
from distance_functions.functions.cubical_complex_distance import CubicalComplexImageDistanceFunction
from results.results_manager import ResultsManager
from datasets.embedded_datasets.generators.embedding_descriptor import EmbeddingDescriptor
from configs.global_config import GlobalConfig
from datasets.image_augmentor import AugmentationSettings
import umap
import phate
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap,TSNE
import copy
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table
from datasets.SCEMILA import *
from datasets.dataset_factory import BASE_MODULES as DATA_SET_MODULES

DEFAULT_TRANSFROM_DICT = {
    "PHATE": (phate.PHATE,{
    'n_components': 3,
    'knn': 10,
    'decay': 40,
    't': 'auto'}),
    
    "TSNE": (TSNE,{
    'n_components': 3,
    'method':'exact'}),
    
    "Isomap": (Isomap,{
    'n_components': 3}),
    
    "UMAP": (umap.UMAP,{
    'n_components': 3}),
    
    "PCA": (PCA,{
    'n_components': 3})
}

SWEEP_PORJECTION_DIM = GlobalConfig.DOWNPROJECTION_TEST_DIMENSIONS

DATASET_NAMES_AND_SETTINGS = {("SCEMILA/image_data","normal"):{"training_mode":True,"balance_dataset_classes": 100,"gpu":False,"augmentation_settings":AugmentationSettings(),"flatten":True,"numpy":True},
                              ("SCEMILA/image_data","dino"):{"training_mode":True,"encode_with_dino_bloom":True,"balance_dataset_classes": 100,"gpu":False,"augmentation_settings":AugmentationSettings(),"flatten":True,"numpy":True},
                              ("Acevedo","normal"):{"training_mode":True,"balance_dataset_classes": 100,"gpu":False,"augmentation_settings":AugmentationSettings(),"flatten":True,"numpy":True},
                              ("Acevedo","dino"):{"training_mode":True,"encode_with_dino_bloom":True,"balance_dataset_classes": 100,"gpu":False,"augmentation_settings":AugmentationSettings(),"flatten":True,"numpy":True},
                              ("FashionMNIST","normal"):{"training_mode":True,"balance_dataset_classes": 100,"gpu":False,"augmentation_settings":AugmentationSettings(),"flatten":True,"numpy":True},
                              ("CIFAR10","normal"):{"training_mode":True,"balance_dataset_classes": 100,"gpu":False,"augmentation_settings":AugmentationSettings(),"flatten":True,"numpy":True},
                              ("MNIST","normal"):{"training_mode":True,"balance_dataset_classes": 100,"gpu":False,"augmentation_settings":AugmentationSettings(),"flatten":True,"numpy":True},
                              }
AUGMENTATIONS_OF_INTEREST = ['translation_aug','gaussian_noise_aug','all','rotation_aug','none' ]
descriptors = []
    


def create_complete_metrics_table(experiment_metrics, key_metric='accuracy'):
    # Initialize dictionary to store metrics and associated details
    combined_metrics = []

    # Iterate over augmentations of interest
    for augmentation, datasets in experiment_metrics.items():
        for dataset_name, dataset_types in datasets.items():
            for dataset_type, results in dataset_types.items():
                for calc_type, calc_results in results.items():
                    if isinstance(calc_results, dict):
                        for dim, dim_metrics in calc_results.items():
                            metric_value1 = dim_metrics.get(key_metric, float('-inf'))
                            metric_value2 = dim_metrics.get("triplet_loss", float('-inf'))
                            entry = {
                                'Dataset': dataset_name,
                                'Type': dataset_type,
                                'Calculation': f"{calc_type}_{dim}",
                                'Augmentation': augmentation,
                                'KeyMetricValue1': metric_value1,
                                'KeyMetricValue2': metric_value2,
                            }
                            combined_metrics.append(entry)
                    else:
                        metric_value1 = calc_results.get(key_metric, float('-inf'))
                        metric_value2 = calc_results.get("triplet_loss", float('-inf'))
                        entry = {
                            'Dataset': dataset_name,
                            'Type': dataset_type,
                            'Calculation': calc_type,
                            'Augmentation': augmentation,
                            'KeyMetricValue1': metric_value1,
                            'KeyMetricValue2': metric_value2
                        }
                        combined_metrics.append(entry)

    # Create a DataFrame from the combined metrics
    df = pd.DataFrame(combined_metrics)

    # Pivot the DataFrame to have augmentations as columns
    pivot_df = df.pivot_table(index=['Dataset', 'Type', 'Calculation'], columns=['Augmentation'], values=['KeyMetricValue2','KeyMetricValue1']).reset_index()

    # Fill NaN values with a placeholder if necessary
    pivot_df = pivot_df.fillna('N/A')

    return pivot_df


def test_augmentation_sensitivity():
    results_manager = ResultsManager.get_manager()
    per_class_samples_for_metric_calc = 10
    metric = DistanceMatrixMetricCalculator
    experiment_metrics = {}
    for augmentation in AUGMENTATIONS_OF_INTEREST:
        experiment_metrics[augmentation]= {}
        for dataset_name,db_settings in DATASET_NAMES_AND_SETTINGS.items():
            if not dataset_name[0] in experiment_metrics[augmentation]:
                experiment_metrics[augmentation][dataset_name[0]] = {dataset_name[1]:{}}
            else:
                experiment_metrics[augmentation][dataset_name[0]][dataset_name[1]] = {}
            iteration_dict = experiment_metrics[augmentation][dataset_name[0]][dataset_name[1]]
            db_settings["augmentation_settings"] = AugmentationSettings.create_settings_with_name(augmentation)
            dataset_class = DATA_SET_MODULES.get(dataset_name[0])
            assert dataset_class is not None
            dataset = dataset_class(**db_settings)
            
            # Euclidean Baseline metrics
            baseline_metric_desc = MetricsDescriptor(metric_calculator=metric,dataset=dataset,distance_function= EuclideanDistance(),per_class_samples=per_class_samples_for_metric_calc)
            baseline_metrics = baseline_metric_desc.calculate_metric()
            iteration_dict["baseline"] = baseline_metrics
            print(baseline_metric_desc.to_dict())
            if dataset_name[1] == "normal":
                # Cubical Complex metrics
                db_settings_tmp = copy.deepcopy(db_settings)
                db_settings_tmp["flatten"] = False
                dataset_cub = dataset_class(**db_settings_tmp)
                cub_complex_metrics_desc = MetricsDescriptor(metric_calculator=metric,dataset=dataset_cub,distance_function= CubicalComplexImageDistanceFunction(),per_class_samples=int(per_class_samples_for_metric_calc/2))
                cub_complex_metrics = cub_complex_metrics_desc.calculate_metric()
                iteration_dict["cubical_complex"] = cub_complex_metrics
                
            # Embedding functions metrics
            for dim in SWEEP_PORJECTION_DIM:
                for transform_name in list(DEFAULT_TRANSFROM_DICT.keys()):
                    if not transform_name in iteration_dict:
                        iteration_dict[transform_name] = {dim:{}}
                    else:
                        iteration_dict[transform_name][dim] = {}
                    trans_func,trans_settings = DEFAULT_TRANSFROM_DICT[transform_name]
                    trans_settings = copy.deepcopy(trans_settings)
                    trans_settings["n_components"] = dim
                    descriptor = EmbeddingDescriptor(f"{dataset_name[0]}_{transform_name}_{dim}",dataset,transform_name,trans_func,trans_settings)
                    embeddings,labels,stats_dic =descriptor.generate_embedding_from_descriptor()
                    embd_ds = EmbeddingBaseDataset(embedding_id = results_manager.calculate_descriptor_id(descriptor))
                    metric_desc = MetricsDescriptor(metric_calculator=metric,dataset=embd_ds,distance_function= EuclideanDistance(),per_class_samples=per_class_samples_for_metric_calc)
                    metrics = metric_desc.calculate_metric()
                    iteration_dict[transform_name][dim]= metrics
    return experiment_metrics


def df_to_pdf2(df,pdf_file_name = 'metrics_results.pdf'):
    num_rows, num_cols = df.shape
    fig_width = num_cols * 2
    fig_height = num_rows * 0.3

    with PdfPages(pdf_file_name) as pdf:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(df.columns))))

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

                        
if __name__ == '__main__':
    experiment_metrics = test_augmentation_sensitivity()
    metrics_table = create_complete_metrics_table(experiment_metrics, key_metric='knn_acc')
    print(metrics_table)
    # Convert to PDF
    df_to_pdf2(metrics_table, 'output.pdf')